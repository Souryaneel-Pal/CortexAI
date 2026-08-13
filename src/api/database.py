import datetime
import json
import os
import uuid
import random
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
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
    completed_at = Column(DateTime, default=datetime.datetime.utcnow)
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

# Seeding logic
def seed_database(db):
    # Default settings
    default_settings = {
        "uncertainty_threshold": "0.60",
        "mdi_threshold": "0.50",
        "ignore_face": "False",
        "ignore_speech": "False",
        "ignore_tabular": "False"
    }
    for k, v in default_settings.items():
        if not db.query(DBSetting).filter(DBSetting.key == k).first():
            setting = DBSetting(key=k, value=v)
            db.add(setting)
    
    # Check if we already have assessments
    if db.query(DBAssessment).first():
        db.commit()
        return

    print("Seeding database with historical assessments...")
    
    # Generate 15 historical assessments spanning July to December 2023
    months_dates = [
        (7, 10, "PT-7701", "Adults"),
        (7, 24, "PT-1082", "Seniors"),
        (8, 5, "PT-2038", "Adolescents"),
        (8, 19, "PT-8842", "Adults"),
        (9, 2, "PT-4821", "Adults"),
        (9, 15, "PT-3094", "Seniors"),
        (9, 28, "PT-6102", "Adolescents"),
        (10, 4, "PT-9105", "Adults"),
        (10, 12, "PT-7721", "Adults"),
        (10, 22, "PT-6540", "Seniors"),
        (10, 24, "PT-8842", "Adults"),  # This is Dr. Vance's latest case
        (11, 2, "PT-9011", "Adolescents"),
        (11, 15, "PT-1102", "Adults"),
        (12, 1, "PT-3049", "Seniors"),
        (12, 18, "PT-5820", "Adults"),
    ]
    
    # Classes & weights
    classes = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
    
    for month, day, patient_id, demographic in months_dates:
        completed_at = datetime.datetime(2023, month, day, 10, 30, random.randint(0, 59))
        
        # Decide stress class based on patient
        if patient_id in ("PT-8842", "PT-9011"):
            predicted_class = "Severe_Stress"
        elif patient_id in ("PT-9105", "PT-3094"):
            predicted_class = "Moderate_Stress"
        elif patient_id in ("PT-7721", "PT-2038"):
            predicted_class = "Mild_Stress"
        else:
            predicted_class = "Healthy"
            
        class_idx = classes.index(predicted_class)
        probs = [0.05] * 4
        probs[class_idx] = 0.85
        
        # Scores
        if predicted_class == "Severe_Stress":
            dep, anx, stress = 25.0 + random.random()*4, 18.0 + random.random()*3, 30.0 + random.random()*5
            conf = 0.88
            mdi_flag = True
            mdi_val = 0.76
        elif predicted_class == "Moderate_Stress":
            dep, anx, stress = 15.0 + random.random()*4, 11.0 + random.random()*3, 20.0 + random.random()*4
            conf = 0.79
            mdi_flag = False
            mdi_val = 0.35
        elif predicted_class == "Mild_Stress":
            dep, anx, stress = 8.0 + random.random()*3, 6.0 + random.random()*2, 10.0 + random.random()*3
            conf = 0.72
            mdi_flag = False
            mdi_val = 0.18
        else:
            dep, anx, stress = 2.0 + random.random()*2, 1.0 + random.random()*2, 3.0 + random.random()*2
            conf = 0.91
            mdi_flag = False
            mdi_val = 0.05
            
        # Tabular features (ordered representation)
        tab_feats = {
            "Sleep_Quality": 3.0 if class_idx > 1 else 7.0,
            "Social_Engagement": 2.0 if class_idx > 1 else 6.0,
            "Daily_App_Usage_Min": 250.0,
            "Typing_Speed_WPM": 45.0,
            "Session_Frequency": 12.0,
            "Idle_Time_Min": 110.0,
            "Facial_Emotion_Variance": 0.6,
            "Eye_Blink_Rate": 25.0,
            "Smile_Intensity": 0.2 if class_idx > 1 else 0.8,
            "Head_Motion_Index": 0.5,
            "MFCC_Mean": -1.0,
            "MFCC_Variance": 15.0,
            "Pitch_Mean": 180.0,
            "Speech_Rate": 3.8,
            "Heart_Rate_BPM": 95.0 if class_idx > 1 else 70.0,
            "HRV_Index": 35.0 if class_idx > 1 else 65.0,
            "Skin_Temperature": 32.5 if class_idx > 1 else 34.0,
            "GSR_Level": 3.5 if class_idx > 1 else 1.8,
        }
        
        # Explain details
        shap_ranked = [
            {"feature": "Sleep_Quality", "mean_abs_shap": 0.85},
            {"feature": "Heart_Rate_BPM", "mean_abs_shap": 0.75},
            {"feature": "HRV_Index", "mean_abs_shap": 0.65},
            {"feature": "Social_Engagement", "mean_abs_shap": 0.55},
        ]
        signed_shap = [
            {"feature": "Sleep_Quality", "shap": 0.85 if class_idx > 1 else -0.85},
            {"feature": "Heart_Rate_BPM", "shap": 0.75 if class_idx > 1 else -0.75},
            {"feature": "HRV_Index", "shap": 0.65 if class_idx > 1 else -0.65},
            {"feature": "Social_Engagement", "shap": 0.55 if class_idx > 1 else -0.55},
        ]
        
        mdi_data = {
            "mdi": mdi_val,
            "face_calm": 0.80 if class_idx == 3 else 0.20,
            "voice_high_arousal": 0.75 if class_idx > 1 else 0.15,
            "physio_high_arousal": 0.85 if class_idx > 1 else 0.25,
            "flag": mdi_flag,
            "dominant_contradiction": "physiology" if class_idx == 3 else None
        }
        
        # Report narrative (pre-filled for seeding)
        narrative = (
            f"The patient displays signs of elevated {predicted_class.replace('_', ' ').lower()}. "
            f"Somatic factors, including poor sleep quality and high physiological arousal "
            f"(Heart Rate: {tab_feats['Heart_Rate_BPM']} bpm, HRV Index: {tab_feats['HRV_Index']}), "
            f"are prominent drivers of this presentation. Recommend close clinical monitoring."
        )
        citations = ["Clinical Guide to Stress and Autonomic Dysregulation (2022)", "DSM-5 Assessment Framework"]
        
        assessment = DBAssessment(
            session_id=str(uuid.uuid4()),
            completed_at=completed_at,
            patient_id=patient_id,
            demographic=demographic,
            tabular_features=json.dumps(tab_feats),
            predicted_class=predicted_class,
            confidence=conf,
            class_probs=json.dumps(probs),
            depression_score=dep,
            anxiety_score=anx,
            stress_score=stress,
            modality_weights=json.dumps({"face": 0.20, "speech": 0.15, "tabular": 0.65}),
            face_emotion_probs=json.dumps({"Happy": 0.10, "Sad": 0.30, "Neutral": 0.60}),
            speech_emotion_probs=json.dumps({"neutral": 0.50, "sad": 0.30, "fear": 0.20}),
            deferred_to_human=(conf < 0.75 and predicted_class == "Severe_Stress"),
            shap_ranked_features=json.dumps(shap_ranked),
            signed_shap=json.dumps(signed_shap),
            masked_distress_index=json.dumps(mdi_data),
            report_narrative=narrative,
            report_citations=json.dumps(citations),
            report_generator="ollama:llama3.1",
            report_fallback_reason=None
        )
        db.add(assessment)
        
    db.commit()

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
