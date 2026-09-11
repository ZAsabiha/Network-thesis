#!/bin/bash
# Port scan (Probe class) - run from attacker host xterm (e.g. h5)
# Usage: ./port_scan.sh <victim_ip>
VICTIM=${1:-10.0.0.1}
echo "Running SYN scan against $VICTIM ..."
sudo nmap -sS -p- $VICTIM
