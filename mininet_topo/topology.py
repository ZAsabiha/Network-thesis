"""
topology.py
=============
Parametrized, research-grade Mininet topology for the SDN IDS thesis.

Replaces the old flat script (1 switch, 5 identical hosts) with a proper
``Topo`` subclass that supports several axes of experimental complexity:

  * MULTI-SWITCH   : 3-tier core / aggregation / edge tree (default), or a
                     classic k-ary fat-tree (``--mode fattree``).
  * SCALE          : every tier is parametrized, so the same file builds a
                     2-switch toy or a 20-switch fabric (``--pods``,
                     ``--edges-per-pod``, ``--hosts-per-edge``, or ``--k``).
  * HOST ROLES     : one web server, one DNS server, one database host, N
                     distributed attackers, the rest normal clients.
  * HETEROGENEOUS  : per-tier TCLink profiles (bw / delay / loss / queue),
    LINKS            plus a few "WAN" host links (high delay + loss) so the
                     IDS is tested under noisy conditions, not only clean LAN.

On start-up it writes ``mininet_topo/topology_state.json`` describing every
host (name, IP, MAC, role, switch, link profile). The traffic generator and
the attack scripts read that file, so they keep working when you change scale
or roles without editing them.

--------------------------------------------------------------------------
TWO ARCHITECTURE NOTES (read before using fat-tree / subnet modes)
--------------------------------------------------------------------------
1. LOOPS vs. the current controller.
   ``controller/ids_controller.py`` is a plain L2 learning switch: it FLOODS
   unknown/broadcast traffic and has no spanning tree. The default "tree" mode
   here is loop-free (single path core->agg->edge->host), so it runs with that
   controller unchanged. A real fat-tree has redundant paths (loops); flooding
   on a loop is a broadcast storm. ``--mode fattree`` therefore prints a
   warning and needs either STP or a loop-aware controller. Use it only once
   the controller can handle multipath.

2. SUBNETS need L3.
   With the L2 controller every host lives in one broadcast domain
   (10.0.0.0/8). Real per-pod subnets require a router node; that is a separate
   controller-side change and is intentionally NOT wired in here yet. The IP
   plan below (10.<pod>.<edge>.<host>) is already subnet-shaped so it is a
   small step later.

RUN (from project root):
  1. Start the controller:   ryu-manager controller/ids_controller.py
  2. Start the topology:      sudo python3 mininet_topo/topology.py
     e.g. bigger fabric:      sudo python3 mininet_topo/topology.py --pods 4 \
                                  --edges-per-pod 2 --hosts-per-edge 4 --attackers 5
"""

import argparse
import json
import os

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo

# --------------------------------------------------------------------------
# Roles. The victim for attack scripts defaults to the web server.
# --------------------------------------------------------------------------
ROLE_WEB = "web"
ROLE_DNS = "dns"
ROLE_DB = "db"
ROLE_CLIENT = "client"
ROLE_ATTACKER = "attacker"
SERVER_ROLES = (ROLE_WEB, ROLE_DNS, ROLE_DB)

# --------------------------------------------------------------------------
# Heterogeneous link profiles (item 5). Tweak these for robustness sweeps.
# bw in Mbps, delay as a string Mininet understands, loss in percent,
# max_queue_size in packets. "wan" links are deliberately noisy.
# --------------------------------------------------------------------------
LINK_PROFILES = {
    "backbone": dict(bw=1000, delay="2ms", loss=0, max_queue_size=2000),   # core<->agg
    "distribution": dict(bw=200, delay="5ms", loss=0, max_queue_size=1000),  # agg<->edge
    "lan": dict(bw=100, delay="1ms", loss=0, max_queue_size=1000),          # clean host link
    "wan": dict(bw=20, delay="40ms", loss=2, max_queue_size=100),           # noisy/distant host
}

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "topology_state.json")


class DCTopo(Topo):
    """Datacenter-style topology.

    Two structures share one role/link/addressing scheme:

    * ``mode="tree"`` (default, loop-free):
        1 core switch
        -> ``pods`` aggregation switches
        -> ``edges_per_pod`` edge switches each
        -> ``hosts_per_edge`` hosts each
      Total hosts = pods * edges_per_pod * hosts_per_edge.

    * ``mode="fattree"`` (has loops -- needs a loop-aware controller):
        classic k-ary fat-tree: k pods, (k/2)^2 core switches,
        k/2 aggregation + k/2 edge per pod, k/2 hosts per edge.
      Total hosts = (k^3) / 4.

    DPIDs are assigned by tier (core 1.., agg 101.., edge 201..) so no two
    switches collide on a DPID even though the controller keys on DPID.
    """

    def build(self, mode="tree", pods=2, edges_per_pod=2, hosts_per_edge=3,
              k=4, n_attackers=3):
        # host_meta accumulates per-host placement so the runner can assign
        # roles/IPs and write the state file. Kept on the instance because
        # Topo.build has no return value.
        self.host_meta = []          # list of dicts, filled below
        self.switch_tier = {}        # name -> "core"/"agg"/"edge"
        self._dpid = {"core": 1, "agg": 101, "edge": 201}

        if mode == "fattree":
            self._build_fattree(k)
        else:
            self._build_tree(pods, edges_per_pod, hosts_per_edge)

        self._assign_roles(n_attackers)

    # ---- switch helper -------------------------------------------------
    def _add_switch(self, name, tier):
        dpid = format(self._dpid[tier], "016x")
        self._dpid[tier] += 1
        self.switch_tier[name] = tier
        return self.addSwitch(name, protocols="OpenFlow13", dpid=dpid)

    def _add_host(self, name, edge_switch, pod, profile):
        # IP/MAC are assigned by the runner (needs the final host list); here
        # we only record placement. Host added without an IP for now.
        self.addHost(name)
        self.addLink(name, edge_switch, cls=TCLink, **LINK_PROFILES[profile])
        self.host_meta.append(dict(name=name, edge_switch=edge_switch, pod=pod,
                                   link_profile=profile))

    # ---- loop-free 3-tier tree ----------------------------------------
    def _build_tree(self, pods, edges_per_pod, hosts_per_edge):
        core = self._add_switch("cs1", "core")
        hid = 1
        for p in range(pods):
            agg = self._add_switch("as%d" % (p + 1), "agg")
            self.addLink(core, agg, cls=TCLink, **LINK_PROFILES["backbone"])
            for e in range(edges_per_pod):
                edge = self._add_switch("es%d_%d" % (p + 1, e + 1), "edge")
                self.addLink(agg, edge, cls=TCLink, **LINK_PROFILES["distribution"])
                for _ in range(hosts_per_edge):
                    # First edge of the first pod is the "server rack": clean
                    # LAN links. Elsewhere, one host per edge gets a WAN link.
                    self._add_host("h%d" % hid, edge, p, profile="lan")
                    hid += 1

    # ---- classic k-ary fat-tree (has loops) ---------------------------
    def _build_fattree(self, k):
        if k % 2 != 0:
            raise ValueError("fat-tree k must be even")
        half = k // 2
        cores = [self._add_switch("cs%d" % (i + 1), "core")
                 for i in range(half * half)]
        hid = 1
        for p in range(k):                       # pods
            aggs, edges = [], []
            for a in range(half):
                aggs.append(self._add_switch("as%d_%d" % (p + 1, a + 1), "agg"))
            for e in range(half):
                edges.append(self._add_switch("es%d_%d" % (p + 1, e + 1), "edge"))
            # aggregation j connects to core switches [j*half : (j+1)*half]
            for j, agg in enumerate(aggs):
                for c in range(half):
                    self.addLink(agg, cores[j * half + c], cls=TCLink,
                                 **LINK_PROFILES["backbone"])
            # every edge connects to every aggregation in the pod (the loops)
            for edge in edges:
                for agg in aggs:
                    self.addLink(edge, agg, cls=TCLink,
                                 **LINK_PROFILES["distribution"])
                for _ in range(half):
                    self._add_host("h%d" % hid, edge, p, profile="lan")
                    hid += 1

    # ---- role + noisy-link assignment ---------------------------------
    def _assign_roles(self, n_attackers):
        meta = self.host_meta
        n = len(meta)
        for m in meta:
            m["role"] = ROLE_CLIENT

        # Servers: first three hosts (they sit on the first edge = server rack).
        for m, role in zip(meta[:3], SERVER_ROLES):
            m["role"] = role

        # Attackers: spread across DIFFERENT edge switches, taking the LAST
        # host of each edge, so a DDoS looks distributed rather than one rack.
        by_edge = {}
        for m in meta:
            by_edge.setdefault(m["edge_switch"], []).append(m)
        candidates = [hosts[-1] for hosts in by_edge.values()
                      if hosts[-1]["role"] == ROLE_CLIENT]
        for m in candidates[:max(0, n_attackers)]:
            m["role"] = ROLE_ATTACKER
        # If more attackers than edges were requested, keep filling clients.
        if n_attackers > len(candidates):
            extra = [m for m in meta if m["role"] == ROLE_CLIENT]
            for m in extra[:n_attackers - len(candidates)]:
                m["role"] = ROLE_ATTACKER

        # Make a few links noisy (item 5): every attacker that is not the very
        # first host, plus every 4th client, gets a WAN profile. Servers stay
        # on clean LAN so a legit service is not itself a source of loss.
        for i, m in enumerate(meta):
            if m["role"] == ROLE_CLIENT and i % 4 == 3:
                m["link_profile"] = "wan"
            if m["role"] == ROLE_ATTACKER and i != 0:
                m["link_profile"] = "wan"


# --------------------------------------------------------------------------
# Runner: assign IPs/MACs, apply link profiles chosen per host, start net,
# write the state file, optionally launch services, drop to CLI.
# --------------------------------------------------------------------------
def _assign_addresses(topo):
    """Give every host a deterministic IP (10.<pod>.<edge_idx>.<n>) and MAC.

    Mask is /8 so, under the L2 controller, all hosts share one broadcast
    domain and can reach each other. The address is still subnet-shaped per
    pod for readability and for a later L3 split.
    """
    edge_index = {}   # edge switch name -> small integer, for the 3rd octet
    counters = {}     # edge switch name -> host counter within that edge
    plan = {}
    for m in topo.host_meta:
        es = m["edge_switch"]
        edge_index.setdefault(es, len(edge_index) + 1)
        counters[es] = counters.get(es, 0) + 1
        pod = m["pod"] + 1
        octet3 = edge_index[es]
        octet4 = counters[es]
        ip = "10.%d.%d.%d/8" % (pod, octet3, octet4)
        # MAC derived from the same numbers -> stable and readable.
        mac = "00:00:%02x:%02x:%02x:%02x" % (pod, octet3, octet4, 0)
        plan[m["name"]] = dict(ip=ip, mac=mac)
    return plan


def _reprofile_links(net, topo):
    """Re-apply the per-host link profile chosen during role assignment.

    Links were created with a default profile in Topo.build (which cannot see
    final roles); here we set the WAN parameters on the hosts that ended up
    flagged "wan".
    """
    for m in topo.host_meta:
        if m["link_profile"] != "wan":
            continue
        host = net.get(m["name"])
        intf = host.defaultIntf()
        intf.config(**LINK_PROFILES["wan"])


def build_net(args):
    topo = DCTopo(mode=args.mode, pods=args.pods,
                  edges_per_pod=args.edges_per_pod,
                  hosts_per_edge=args.hosts_per_edge, k=args.k,
                  n_attackers=args.attackers)

    plan = _assign_addresses(topo)
    net = Mininet(topo=topo, controller=None, switch=OVSSwitch, link=TCLink,
                  autoSetMacs=False, build=False)

    info("*** Adding remote controller (expects Ryu on %s:%d)\n"
         % (args.controller_ip, args.controller_port))
    net.addController("c0", controller=RemoteController,
                      ip=args.controller_ip, port=args.controller_port)

    # build() creates the host/switch objects from the topo. With build=False
    # they do not exist yet, so IP/MAC assignment must come AFTER build (but
    # before start, so the interfaces carry the right address when brought up).
    net.build()
    for name, addr in plan.items():
        net.get(name).setIP(addr["ip"])
        net.get(name).setMAC(addr["mac"])

    net.start()
    _reprofile_links(net, topo)

    state = _write_state(net, topo, plan, args)
    info("*** Topology '%s' ready: %d switches, %d hosts (%d attackers)\n"
         % (args.mode, len(net.switches), len(net.hosts),
            sum(1 for h in state["hosts"] if h["role"] == ROLE_ATTACKER)))
    victim = state.get("victim_ip")
    info("*** Victim (web server) = %s. Attackers: %s\n"
         % (victim, ", ".join(h["name"] for h in state["hosts"]
                              if h["role"] == ROLE_ATTACKER)))
    info("*** State written to %s\n" % STATE_PATH)

    if args.mode == "fattree":
        info("*** WARNING: fat-tree has loops. The default L2 learning "
             "controller will broadcast-storm. Enable STP or a loop-aware "
             "controller before trusting results.\n")

    if args.services:
        _start_services(net, state)

    traffic_procs = []
    if args.traffic:
        from traffic import start_background_traffic, stop_background_traffic
        traffic_procs = start_background_traffic(net, state)

    CLI(net)

    if traffic_procs:
        stop_background_traffic(net, traffic_procs)
    net.stop()


def _write_state(net, topo, plan, args):
    hosts = []
    for m in topo.host_meta:
        h = net.get(m["name"])
        hosts.append(dict(name=m["name"], ip=h.IP(), mac=h.MAC(),
                          role=m["role"], edge_switch=m["edge_switch"],
                          pod=m["pod"], link_profile=m["link_profile"]))
    switches = [dict(name=s, dpid=int(topo.nodeInfo(s).get("dpid", "0"), 16)
                     if topo.nodeInfo(s).get("dpid") else net.get(s).dpid,
                     tier=topo.switch_tier[s]) for s in topo.switches()]
    # Switch<->switch links, so the dashboard can draw the fabric for any mode
    # (tree or fat-tree) without re-deriving it from naming conventions.
    switch_names = set(topo.switches())
    fabric_links = [[a, b] for a, b in topo.links()
                    if a in switch_names and b in switch_names]
    victim = next((h["ip"] for h in hosts if h["role"] == ROLE_WEB), None)
    state = dict(
        mode=args.mode,
        params=dict(pods=args.pods, edges_per_pod=args.edges_per_pod,
                    hosts_per_edge=args.hosts_per_edge, k=args.k,
                    attackers=args.attackers),
        victim_ip=victim,
        servers={h["role"]: h["ip"] for h in hosts if h["role"] in SERVER_ROLES},
        hosts=hosts,
        switches=switches,
        fabric_links=fabric_links,
    )
    with open(STATE_PATH, "w") as fh:
        json.dump(state, fh, indent=2)
    return state


def _start_services(net, state):
    """Best-effort launch of the real services. Never fails the topology if a
    binary is missing -- just logs it."""
    for h in state["hosts"]:
        host = net.get(h["name"])
        if h["role"] == ROLE_WEB:
            host.cmd("mkdir -p /tmp/webroot && echo '<h1>IDS test site</h1>' "
                     "> /tmp/webroot/index.html")
            host.cmd("cd /tmp/webroot && nohup python3 -m http.server 80 "
                     "> /tmp/web_%s.log 2>&1 &" % h["name"])
            info("*** %s serving HTTP on %s:80\n" % (h["name"], h["ip"]))
        elif h["role"] == ROLE_DB:
            # Simple always-on TCP listener standing in for a DB port (3306).
            host.cmd("nohup python3 -c \""
                     "import socket;s=socket.socket();"
                     "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                     "s.bind(('0.0.0.0',3306));s.listen(64);\n"
                     "import threading\n"
                     "\nwhile True:\n c,_=s.accept();c.recv(64);c.close()\" "
                     "> /tmp/db_%s.log 2>&1 &" % h["name"])
            info("*** %s listening on %s:3306 (db stub)\n" % (h["name"], h["ip"]))
        elif h["role"] == ROLE_DNS:
            if host.cmd("which dnsmasq").strip():
                host.cmd("nohup dnsmasq -k -p 53 > /tmp/dns_%s.log 2>&1 &"
                         % h["name"])
                info("*** %s running dnsmasq on %s:53\n" % (h["name"], h["ip"]))
            else:
                info("*** dnsmasq not installed; %s (DNS) idle. "
                     "`apt install dnsmasq` to enable.\n" % h["name"])


def parse_args():
    p = argparse.ArgumentParser(description="SDN IDS research topology")
    p.add_argument("--mode", choices=["tree", "fattree"], default="tree")
    p.add_argument("--pods", type=int, default=2, help="tree: aggregation switches")
    p.add_argument("--edges-per-pod", type=int, default=2)
    p.add_argument("--hosts-per-edge", type=int, default=3)
    p.add_argument("--k", type=int, default=4, help="fattree: number of pods (even)")
    p.add_argument("--attackers", type=int, default=3)
    p.add_argument("--controller-ip", default="127.0.0.1")
    p.add_argument("--controller-port", type=int, default=6633)
    p.add_argument("--no-services", dest="services", action="store_false",
                   help="do not auto-start web/dns/db services")
    p.add_argument("--traffic", action="store_true",
                   help="generate benign background traffic on client hosts")
    p.set_defaults(services=True)
    return p.parse_args()


if __name__ == "__main__":
    setLogLevel("info")
    build_net(parse_args())
