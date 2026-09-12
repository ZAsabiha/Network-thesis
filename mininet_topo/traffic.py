"""
traffic.py
============
Legitimate background traffic for the SDN IDS testbed (item 3).

The IDS must be tested against normal traffic, not only attacks, or every
result is a false-positive rate of "we only ever sent attacks". This module
runs benign client behaviour inside each host's namespace using ``host.popen``:

  * clients periodically curl the web server (HTTP GET on a jittered timer),
  * ping the DNS and database hosts,
  * dig the DNS host when dnsmasq is up,
  * a couple of clients run short periodic iperf transfers to the web host,
    giving volumetric-but-legitimate load the IDS should NOT flag.

It is driven from inside the Mininet process (it needs the live ``net`` to
reach namespaces), so ``topology.py --traffic`` calls ``start_background_traffic``
and ``stop_background_traffic`` on shutdown.

Every interval is jittered so background traffic does not become a fixed
signature the model could trivially separate from attacks.
"""

import shlex

# Which roles generate background traffic. Attackers stay silent here; their
# traffic comes from the attack scripts so the two are cleanly separable.
CLIENT_ROLES = ("client",)


def _server_ips(state):
    servers = state.get("servers", {})
    return servers.get("web"), servers.get("dns"), servers.get("db")


def _client_loop_cmd(web, dns, db, min_gap, max_gap, dns_up):
    """A shell loop, one per client, doing benign requests forever.

    Kept as plain busybox-friendly shell so it runs in a bare Mininet host
    without extra packages. Each action is best-effort (``|| true``) so a
    missing curl/dig never kills the loop.
    """
    dig_line = ("dig @%s example.local +time=1 +tries=1 >/dev/null 2>&1 || true"
                % dns) if (dns and dns_up) else ":"
    web_line = ("curl -s -m 3 http://%s/ >/dev/null 2>&1 "
                "|| wget -q -T 3 -O /dev/null http://%s/ 2>/dev/null || true"
                % (web, web)) if web else ":"
    ping_db = ("ping -c 2 -W 1 %s >/dev/null 2>&1 || true" % db) if db else ":"
    ping_dns = ("ping -c 1 -W 1 %s >/dev/null 2>&1 || true" % dns) if dns else ":"
    # $RANDOM % span + min  -> jittered sleep between requests.
    span = max(1, max_gap - min_gap)
    return (
        "while true; do "
        "%s; %s; %s; %s; "
        "sleep $((RANDOM %% %d + %d)); "
        "done" % (web_line, ping_db, ping_dns, dig_line, span, min_gap)
    )


def start_background_traffic(net, state, min_gap=2, max_gap=8, iperf=True):
    """Start benign traffic on every client host. Returns popen handles so the
    caller can terminate them on shutdown."""
    web, dns, db = _server_ips(state)
    procs = []

    # dnsmasq may not be running; probe once on the DNS host.
    dns_up = False
    if dns:
        dns_host = next((net.get(h["name"]) for h in state["hosts"]
                         if h["role"] == "dns"), None)
        if dns_host is not None:
            dns_up = bool(dns_host.cmd("pgrep -x dnsmasq").strip())

    # iperf server on the web host for the volumetric-legit clients.
    iperf_ok = False
    if iperf and web:
        web_host = next((net.get(h["name"]) for h in state["hosts"]
                         if h["role"] == "web"), None)
        if web_host is not None and web_host.cmd("which iperf").strip():
            web_host.cmd("nohup iperf -s > /tmp/iperf_web.log 2>&1 &")
            iperf_ok = True

    clients = [h for h in state["hosts"] if h["role"] in CLIENT_ROLES]
    for i, h in enumerate(clients):
        host = net.get(h["name"])
        loop = _client_loop_cmd(web, dns, db, min_gap, max_gap, dns_up)
        procs.append(host.popen(["bash", "-c", loop]))

        # Every 4th client also runs periodic short iperf transfers to web:
        # legitimate bursts the IDS must not confuse with a flood.
        if iperf_ok and i % 4 == 0:
            burst = ("while true; do iperf -c %s -t 3 >/dev/null 2>&1 || true; "
                     "sleep $((RANDOM %% 20 + 10)); done" % web)
            procs.append(host.popen(["bash", "-c", burst]))

    net_info("Background traffic started on %d clients "
             "(iperf=%s, dns=%s)." % (len(clients), iperf_ok, dns_up))
    return procs


def stop_background_traffic(net, procs):
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    # Kill the server-side helpers we started.
    for h in net.hosts:
        h.cmd("pkill -f 'iperf -s' 2>/dev/null || true")
    net_info("Background traffic stopped.")


def net_info(msg):
    try:
        from mininet.log import info
        info("*** " + msg + "\n")
    except Exception:
        print("*** " + msg)


# Small self-check so `python3 mininet_topo/traffic.py` validates the shell
# generation without needing root/Mininet.
if __name__ == "__main__":
    demo = _client_loop_cmd("10.1.1.1", "10.1.1.2", "10.1.1.3", 2, 8, True)
    print("Sample client loop:\n", demo)
    # Ensure it is syntactically valid shell.
    import subprocess
    r = subprocess.run(["bash", "-n", "-c", demo])
    print("bash -n exit:", r.returncode)
    _ = shlex  # noqa: keep import meaningful if reused later
