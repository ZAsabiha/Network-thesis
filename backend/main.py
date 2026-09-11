"""
main.py
=========
FastAPI backend. The Ryu controller POSTs alerts here; the dashboard
polls GET /alerts and GET /stats.

RUN (from the backend/ directory - the imports below are flat, not package
relative, so `uvicorn backend.main:app` from the project root will not work):
  cd backend && uvicorn main:app --reload --port 8000
"""

import time
from collections import Counter
from typing import Optional

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import init_db, clear_alerts, get_db, Alert

app = FastAPI(title="SDN IDS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
# Start every session with a clean feed; old alerts must not linger on the
# dashboard. Run uvicorn WITHOUT --reload so a code edit mid-demo does not
# wipe the alerts you just captured.
clear_alerts()


class AlertIn(BaseModel):
    timestamp: float
    attack_class: str
    confidence: float
    src_ip: str
    dst_ip: str
    dpid: int
    packet_count: int
    byte_count: int
    # Optional so older clients (and the inject tool) can post without them.
    detected_by: str = ""
    ml_class: str = ""
    ml_confidence: float = 0.0
    inference_ms: float = 0.0
    source_count: int = 1


@app.post("/alerts")
def create_alert(alert: AlertIn, db: Session = Depends(get_db)):
    db_alert = Alert(**alert.dict())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return {"status": "ok", "id": db_alert.id}


@app.get("/alerts")
def list_alerts(limit: int = 100, attack_class: Optional[str] = None,
                 db: Session = Depends(get_db)):
    q = db.query(Alert).order_by(desc(Alert.timestamp))
    if attack_class:
        q = q.filter(Alert.attack_class == attack_class)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "attack_class": r.attack_class,
            "confidence": r.confidence,
            "src_ip": r.src_ip,
            "dst_ip": r.dst_ip,
            "dpid": r.dpid,
            "packet_count": r.packet_count,
            "byte_count": r.byte_count,
            "detected_by": r.detected_by or "",
            "ml_class": r.ml_class or "",
            "ml_confidence": r.ml_confidence or 0.0,
            "inference_ms": r.inference_ms or 0.0,
            "source_count": r.source_count or 1,
        }
        for r in rows
    ]


@app.get("/stats")
def get_stats(window_sec: int = 3600, db: Session = Depends(get_db)):
    cutoff = time.time() - window_sec
    rows = db.query(Alert).filter(Alert.timestamp >= cutoff).all()

    class_counts = Counter(r.attack_class for r in rows)
    total = len(rows)
    distinct_attackers = len({r.src_ip for r in rows})

    latencies = [r.inference_ms for r in rows if r.inference_ms]
    confidences = [r.confidence for r in rows if r.confidence is not None]

    return {
        "window_sec": window_sec,
        "total_alerts": total,
        "distinct_attacker_ips": distinct_attackers,
        "by_class": dict(class_counts),
        # Measured on live traffic, so the thesis can quote observed detection
        # latency instead of only the offline benchmark in model_metadata.json.
        "mean_inference_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }


@app.get("/")
def root():
    return {"status": "SDN IDS backend running"}
