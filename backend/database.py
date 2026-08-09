"""
database.py
=============
SQLite database via SQLAlchemy. Stores every alert the Ryu controller posts.
"""

import os

from sqlalchemy import create_engine, Column, Integer, String, Float
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
    # Which path raised this alert (a detector.py rule, or "ml-model"), plus
    # what the model thought regardless - so the report can compare the two.
    detected_by = Column(String, default="")
    ml_class = Column(String, default="")
    ml_confidence = Column(Float, default=0.0)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
