"""
topology.py
=============
Custom Mininet topology: 1 switch, 5 hosts (4 "normal" + 1 designated
attacker), connected to a remote Ryu controller.

RUN (from project root, in a terminal with Mininet installed):
  sudo python3 mininet_topo/topology.py

Then, in a SEPARATE terminal, start the Ryu controller first:
  ryu-manager controller/ids_controller.py

Inside the Mininet CLI that opens:
  pingall                     # verify connectivity
  xterm h1 h5                 # open terminals on host1 (victim) and host5 (attacker)
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


def build_topology():
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink)

    print("*** Adding controller (expects Ryu on 127.0.0.1:6633)")
    c0 = net.addController("c0", controller=RemoteController,
                            ip="127.0.0.1", port=6633)

    print("*** Adding switch")
    s1 = net.addSwitch("s1", protocols="OpenFlow13")

    print("*** Adding hosts")
    h1 = net.addHost("h1", ip="10.0.0.1/24")  # victim / server
    h2 = net.addHost("h2", ip="10.0.0.2/24")  # normal client
    h3 = net.addHost("h3", ip="10.0.0.3/24")  # normal client
    h4 = net.addHost("h4", ip="10.0.0.4/24")  # normal client
    h5 = net.addHost("h5", ip="10.0.0.5/24")  # attacker

    print("*** Creating links")
    for h in [h1, h2, h3, h4, h5]:
        net.addLink(h, s1, bw=100)  # 100 Mbps links

    print("*** Starting network")
    net.build()
    c0.start()
    s1.start([c0])

    print("*** Network ready. Victim=h1 (10.0.0.1), Attacker=h5 (10.0.0.5)")
    print("*** Run 'pingall' to verify connectivity before starting attacks.")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    build_topology()
