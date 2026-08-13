import datetime
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Place database in the workspace's data directory
DATABASE_URL = "sqlite:///./data/cortexai.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBSetting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True, index=True)
    value = Column(String)

class DBAssessment(Base):
    __tablename__ = "assessments"
    session_id = Column(String, primary_key=True, index=True)
    # `utcnow` is deprecated in 3.12+; use an explicit timezone-aware default.
    completed_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    patient_id = Column(String, index=True)
    demographic = Column(String, default="Adults")  # Adults, Seniors, Adolescents
    
    # Inputs
    tabular_features = Column(Text)  # JSON representation
    face_image_base64 = Column(Text, nullable=True)
    speech_audio_base64 = Column(Text, nullable=True)
    
    # Predictions
    predicted_class = Column(String)
    confidence = Column(Float)
    class_probs = Column(String)  # JSON representation (list)
    
    # Scores
    depression_score = Column(Float)
    anxiety_score = Column(Float)
    stress_score = Column(Float)
    
    # Modalities
    modality_weights = Column(String)  # JSON representation (dict)
    face_emotion_probs = Column(String)  # JSON representation (dict)
    speech_emotion_probs = Column(String)  # JSON representation (dict)
    
    # Uncertainty
    deferred_to_human = Column(Boolean, default=False)
    
    # Explanations
    shap_ranked_features = Column(Text)  # JSON representation
    signed_shap = Column(Text)  # JSON representation
    masked_distress_index = Column(Text)  # JSON representation
    gradcam = Column(Text, nullable=True)  # JSON representation
    audio_integrated_gradients = Column(Text, nullable=True)  # JSON representation
    
    # Report
    report_narrative = Column(Text, nullable=True)
    report_citations = Column(Text, nullable=True)  # JSON list
    report_generator = Column(String, nullable=True)
    report_fallback_reason = Column(String, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_setting(key: str, default=None):
    db = SessionLocal()
    try:
        setting = db.query(DBSetting).filter(DBSetting.key == key).first()
        if setting is None:
            return default
        # Cast to type based on default value
        val_str = setting.value
        if isinstance(default, bool):
            return val_str.lower() in ("true", "1", "yes")
        if isinstance(default, float):
            return float(val_str)
        if isinstance(default, int):
            return int(val_str)
        return val_str
    except Exception:
        return default
    finally:
        db.close()

def set_db_setting(key: str, value):
    db = SessionLocal()
    try:
        setting = db.query(DBSetting).filter(DBSetting.key == key).first()
        if setting is None:
            setting = DBSetting(key=key, value=str(value))
            db.add(setting)
        else:
            setting.value = str(value)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# --- Settings bootstrap -----------------------------------------------------
#
# NOTE ON SEED DATA (deliberate design decision, do not "restore"):
# An earlier version of this function inserted 15 fabricated historical
# assessments -- invented patient IDs, invented 2023 dates, invented SHAP and
# MDI values, and hand-written narratives stored with
# `report_generator="ollama:llama3.1"`.
#
# That was strictly worse than the frontend mock data it replaced. Mock data in
# the frontend sat behind a `SampleDataBadge` and was visibly not real; the same
# fabrications written into the assessments table are indistinguishable from
# genuine model output, are served through `/api/dashboard` and `/api/analytics`
# as real history, and -- because they carried an `ollama:llama3.1` generator
# tag -- would have presented hand-written text as a model-authored clinical
# narrative. In a system whose entire premise is that every number and every
# sentence is traceable, that is the one thing that must not happen.
#
# So this seeds configuration defaults only. Real history accumulates from real
# assessments; until one is run, the dashboard shows an honest empty state.
DEFAULT_SETTINGS = {
    "uncertainty_threshold": "0.60",
    "mdi_threshold": "0.50",
    "ignore_face": "False",
    "ignore_speech": "False",
    "ignore_tabular": "False",
}


def seed_default_settings(db):
    """Insert any missing setting with its documented default.

    Existing values are never overwritten, so an operator's tuned thresholds
    survive a restart.
    """
    for key, value in DEFAULT_SETTINGS.items():
        if not db.query(DBSetting).filter(DBSetting.key == key).first():
            db.add(DBSetting(key=key, value=value))
    db.commit()


# Signature of the fabricated rows the old seeder wrote: one of these patient
# ids AND a 2023 timestamp. Real assessments carry a generated patient id and
# the date they were actually run, so this cannot match a genuine row.
_FABRICATED_SEED_PATIENT_IDS = frozenset(
    {
        "PT-7701", "PT-1082", "PT-2038", "PT-8842", "PT-4821",
        "PT-3094", "PT-6102", "PT-9105", "PT-7721", "PT-6540",
        "PT-9011", "PT-1102", "PT-3049", "PT-5820",
    }
)
_FABRICATED_SEED_YEAR = 2023


def find_fabricated_seed_rows(db):
    """Return the rows written by the removed seeder, for review or deletion."""
    return [
        row
        for row in db.query(DBAssessment).all()
        if row.patient_id in _FABRICATED_SEED_PATIENT_IDS
        and row.completed_at is not None
        and row.completed_at.year == _FABRICATED_SEED_YEAR
    ]


def purge_fabricated_seed_rows(dry_run: bool = True) -> list[str]:
    """Delete the fabricated history the old seeder inserted.

    Deliberately explicit and dry-run by default: this deletes rows from a
    database that may also hold genuine assessments, so it reports exactly
    what it matched before anything is removed. Run as::

        python -m src.api.database --purge-seed --apply

    Returns the session ids matched.
    """
    db = SessionLocal()
    try:
        rows = find_fabricated_seed_rows(db)
        session_ids = [row.session_id for row in rows]
        if not dry_run:
            for row in rows:
                db.delete(row)
            db.commit()
        return session_ids
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_default_settings(db)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CortexAI database maintenance.")
    parser.add_argument("--purge-seed", action="store_true", help="remove the old fabricated seed history")
    parser.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()

    if args.purge_seed:
        matched = purge_fabricated_seed_rows(dry_run=not args.apply)
        verb = "Deleted" if args.apply else "Would delete (dry run)"
        print(f"{verb} {len(matched)} fabricated seed row(s).")
        for session_id in matched:
            print(f"  {session_id}")
        if not args.apply and matched:
            print("\nRe-run with --apply to delete them.")
    else:
        init_db()
        print("Database initialised (schema + default settings).")
