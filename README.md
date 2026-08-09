# SDN-Integrated Intrusion Detection System

Closed-loop IDS built on Mininet + Ryu: polls OpenFlow flow stats, classifies
traffic, and pushes alerts to a live dashboard.

Detection runs two paths in parallel. Threshold rules in
`controller/detector.py` are the primary trigger; the ML model is a secondary
signal reported alongside every alert so the two can be compared. See
[Why detection is rule-based right now](#why-detection-is-rule-based-right-now).

Mitigation (installing block / rate-limit flow rules on detection) is a
separate component and is **not** in this tree.

## Environment note (important)

Ryu has compatibility issues with modern Python (3.11+) and modern eventlet.
Recommended setup: a dedicated virtualenv on Python 3.8 or 3.9 specifically
for the `controller/` code, separate from the environment you use for
`model/`, `backend/`, and `dashboard/` (which can use any modern Python 3.10+).

```bash
# Controller env
python3.9 -m venv venv-ryu
source venv-ryu/bin/activate
pip install ryu==4.34 eventlet==0.30.2 joblib scikit-learn requests

# General env (model training, backend, dashboard)
python3 -m venv venv-app
source venv-app/bin/activate
pip install -r requirements.txt
```

## Run order — 4 terminals

Start them in this order. Each one stays running; do not close them.

**Terminal 1 — backend.** Must be started from inside `backend/`; the imports
are flat, so `uvicorn backend.main:app` from the project root will not work.

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — dashboard.**

```bash
cd dashboard
streamlit run streamlit_app.py
```

Opens on http://localhost:8501. It should say *"No alerts yet"* — that is the
correct starting state, not a failure.

**Terminal 3 — Ryu controller** (in the `venv-ryu` environment, from the
project root):

```bash
ryu-manager controller/ids_controller.py
```

Wait for `Loaded IDS model from ...` before starting Mininet.

**Terminal 4 — Mininet** (needs sudo, from the project root):

```bash
sudo python3 mininet_topo/topology.py
```

Inside the Mininet CLI:

```
pingall          # verify connectivity - all 5 hosts should reach each other
xterm h5         # attacker terminal
```

**Then run an attack** from the h5 xterm:

```bash
cd attacks
./syn_flood.sh 10.0.0.1 80                    # -> DoS
./udp_flood.sh 10.0.0.1 53                    # -> DoS
./port_scan.sh 10.0.0.1                       # -> Probe
python3 slowloris_attack.py 10.0.0.1 --port 80    # -> DoS (low-and-slow)
sudo python3 control_plane_saturation.py --victim 10.0.0.1 --rate 500
```

Within one or two poll cycles (3s each) a red **ATTACK DETECTED** banner and a
toast should appear on the dashboard, and Terminal 3 should log an `[ALERT]`
line.

## Why detection is rule-based right now

The shipped `model/ids_model.pkl` was trained on StandardScaler-normalised
features, but only the classifier was pickled — the fitted scaler was thrown
away. The controller feeds raw OpenFlow counters, so every live flow lands far
outside the training distribution and the model answers `Normal` to
everything. Measured: 299 of 315 sampled flow shapes return `Normal`, and the
few `DoS` hits peak at confidence 0.44, below the 0.6 alert threshold. That is
why no alert ever reached the dashboard.

The training script and dataset that produced that pickle are both gone, so
the scaler cannot be recovered. To fix it properly, put a real dataset in
`data/` and retrain:

```bash
python3 model/train_model.py --csv data/InSDN.csv
```

`train_model.py` now saves a `Pipeline(StandardScaler, RandomForest)` so the
transform always travels with the classifier, and it runs a live-scale sanity
check that refuses to save a model which calls an obvious flood `Normal`.
Once retrained, the ML path raises alerts on its own again — the controller
already treats it as an independent trigger.

## Project structure

```
mininet_topo/    Custom topology (1 switch, 5 hosts incl. 1 attacker)
model/           Training pipeline + feature_config.py (shared with controller)
controller/      Ryu app: flow stats polling, detection rules, inference
backend/         FastAPI + SQLite: receives/serves alerts
dashboard/       Streamlit live dashboard with alert banner + toasts
attacks/         Attack simulation scripts
```

## Troubleshooting

**Dashboard stuck on "No alerts yet".** Work backwards along the chain:

1. `curl localhost:8000/alerts` — empty means nothing is reaching the backend.
2. Check Terminal 3 for `[ALERT]` lines. If they appear but the backend is
   empty, the controller cannot reach `BACKEND_URL`.
3. No `[ALERT]` lines means detection is not firing — confirm the attack is
   actually running and that `pingall` worked first.

**Backend errors about missing columns.** The alerts table gained
`detected_by`, `ml_class` and `ml_confidence`. Delete `backend/ids_alerts.db`
and restart the backend; it recreates the schema on startup.

**Alerts show MAC addresses instead of IPs.** Fixed — the controller now
installs IPv4 matches so flow stats carry real `ipv4_src` / `ipv4_dst` /
`ip_proto`. If you still see MACs, you are running a stale controller.

## Corner cases this project specifically targets

1. Flash crowd vs. DDoS — the volumetric rule ignores MTU-filling flows, so a
   legitimate 90 Mbps iperf run does not alert. The trade-off, worth stating
   in the report: a flood built from MTU-sized packets is indistinguishable
   from legitimate bulk traffic at flow-stats granularity.
2. Low-and-slow attacks (Slowloris) — caught by concurrency (many long-lived
   near-idle flows from one source), not by rate. A single idle ssh session
   deliberately does not trigger it.
3. Flow-table exhaustion — randomised-source floods install one flow per
   source; `hard_timeout=30` ages them out.
4. Control-plane saturation — `attacks/control_plane_saturation.py` targets
   the Ryu controller itself via randomised-source PacketIn flooding.
5. Detection latency — every alert is timestamped in the database, so
   end-to-end response time is measurable against the attack start time.

## TODO before final report

- [ ] Replace the broken pickle with a retrained Pipeline (see above)
- [ ] Update `COLUMN_MAP` in `model/train_model.py` to match your dataset headers
- [ ] Run each attack type, record detection latency and false-positive rate
- [ ] Test flash-crowd scenario (iperf between legit hosts during an attack)
- [ ] Compare rule-based vs. ML verdicts using the `detected_by` / `ml_class`
      columns now recorded on every alert
