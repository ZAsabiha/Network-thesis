"""
database.py
=============
SQLite database via SQLAlchemy. Stores every alert the Ryu controller posts.
"""

import os

from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Anchor the database to this file, not the working directory. With a
# relative "./ids_alerts.db" you silently get a different database depending
# on whether you launch from the project root or from backend/, which looks
# exactly like "the controller posts alerts but the dashboard shows none".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'ids_alerts.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, index=True)
    attack_class = Column(String, index=True)
    confidence = Column(Float)
    src_ip = Column(String, index=True)
    dst_ip = Column(String)
    dpid = Column(Integer)
    packet_count = Column(Integer)
    byte_count = Column(Integer)
    # Detection is ML-only, so detected_by is always "ml-model" and ml_class /
    # ml_confidence mirror attack_class / confidence. They are kept so rows
    # written by the earlier rule-based build stay readable, and so a future
    # second detector can be told apart in the same table.
    detected_by = Column(String, default="")
    ml_class = Column(String, default="")
    ml_confidence = Column(Float, default=0.0)
    # Model inference time for this flow, milliseconds. Recorded per alert so
    # detection latency can be reported from real runs rather than a benchmark.
    inference_ms = Column(Float, default=0.0)
    # Distinct sources this alert represents. 1 for a single-source attack;
    # thousands for a spoofed flood collapsed into one row.
    source_count = Column(Integer, default=1)

class Mitigation(Base):
    """One row per block. Keyed on the attacker's MAC; src_ip is a sample of
    the (possibly spoofed) IPs seen behind it, kept only for readability."""

    __tablename__ = "mitigations"

    id = Column(Integer, primary_key=True, index=True)
    src_mac = Column(String, index=True)
    src_ip = Column(String, default="")
    attack_class = Column(String)
    dpid = Column(Integer)
    blocked_at = Column(Float, index=True)
    expires_at = Column(Float)
    duration_sec = Column(Float, default=0.0)
    reason = Column(String, default="")

class ManualBlock(Base):
    """A block requested from the dashboard. The controller polls for pending
    rows, installs the drop, then marks them applied."""

    __tablename__ = "manual_blocks"

    id = Column(Integer, primary_key=True, index=True)
    src_mac = Column(String, index=True)
    src_ip = Column(String, default="")
    duration_sec = Column(Float, default=120.0)
    victim = Column(String, default="")
    created_at = Column(Float)
    applied = Column(Integer, default=0)


def clear_manual_blocks():
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {ManualBlock.__tablename__}"))

def clear_mitigations():
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {Mitigation.__tablename__}"))


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def clear_alerts():
    """
    Wipe every stored alert. Called at backend startup so each run begins
    with an empty feed - otherwise attacks from a previous session keep
    showing on the dashboard even though nothing new has been detected.
    """
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {Alert.__tablename__}"))


def _add_missing_columns():
    """
    create_all() only creates missing TABLES, never missing columns, so a
    database written by an earlier version keeps its old schema and every
    insert then fails on the new field. SQLite cannot do this through
    SQLAlchemy's DDL, so add the column directly when it is absent.
    """
    existing = {c["name"] for c in inspect(engine).get_columns(Alert.__tablename__)}
    for name, ddl in (("inference_ms", "FLOAT DEFAULT 0.0"),
                      ("source_count", "INTEGER DEFAULT 1")):
        if name not in existing:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {Alert.__tablename__} ADD COLUMN {name} {ddl}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
