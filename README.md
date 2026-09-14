# SDN Intrusion Detection & Mitigation System

A machine-learning intrusion **detection and mitigation** system for
software-defined networks. A single Random Forest, trained offline on the
**InSDN** dataset, runs as the *only* live detector inside a **Ryu** controller
that monitors a multi-switch **Mininet** testbed. When it flags an attack the
controller closes the loop and blocks the attacker in the data plane, and a live
**Streamlit** dashboard shows the topology, alerts, and active blocks in real time.

> The ML model is the sole detector — there are no hand-written threshold rules
> on the live path. The only number that gates an alert is the model's own
> predicted probability.
