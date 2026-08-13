import sqlite3
import json
import os
import time
from datetime import datetime

DB_PATH = "./data/cortexai.db"

def derive_dashboard_metrics():
    start_time = time.perf_counter()
    if not os.path.exists(DB_PATH):
        return {
            "error": f"Database file not found at {DB_PATH}",
            "execution_time_ms": (time.perf_counter() - start_time) * 1000
        }
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # 1. Total Assessments count
        cursor.execute("SELECT COUNT(*) as total FROM assessments")
        total = cursor.fetchone()["total"]
        
        if total == 0:
            return {
                "totalAssessments": 0,
                "heroStats": [],
                "stressDistribution": [],
                "trend": [],
                "recentAssessments": [],
                "execution_time_ms": (time.perf_counter() - start_time) * 1000
            }
            
        # 2. Predicted Class Counts
        cursor.execute("SELECT predicted_class, COUNT(*) as count FROM assessments GROUP BY predicted_class")
        counts = {"Healthy": 0, "Mild_Stress": 0, "Moderate_Stress": 0, "Severe_Stress": 0}
        for row in cursor.fetchall():
            cls = row["predicted_class"]
            if cls in counts:
                counts[cls] = row["count"]
                
        healthy = counts["Healthy"]
        mild = counts["Mild_Stress"]
        moderate = counts["Moderate_Stress"]
        severe = counts["Severe_Stress"]
        
        pct_healthy = int(round(healthy / total * 100)) if total > 0 else 0
        pct_mild = int(round(mild / total * 100)) if total > 0 else 0
        pct_mod = int(round(moderate / total * 100)) if total > 0 else 0
        pct_sev = 100 - (pct_healthy + pct_mild + pct_mod) if total > 0 else 0
        if pct_sev < 0:
            pct_sev = 0
            
        # 3. Monthly Trends (grouped by month string)
        cursor.execute("SELECT completed_at FROM assessments")
        monthly_counts = {m: 0 for m in ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
        for row in cursor.fetchall():
            dt_str = row["completed_at"]
            # handle database formats with or without fractional seconds
            if "." in dt_str:
                dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
            else:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            month_str = dt.strftime("%b")
            if month_str in monthly_counts:
                monthly_counts[month_str] += 1
                
        trend = [{"month": m, "assessments": monthly_counts[m]} for m in ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
        
        # 4. Recent Assessments (top 10 order by date desc)
        cursor.execute("""
            SELECT patient_id, completed_at, predicted_class, deferred_to_human 
            FROM assessments 
            ORDER BY completed_at DESC 
            LIMIT 10
        """)
        recent = []
        for row in cursor.fetchall():
            dt_str = row["completed_at"]
            if "." in dt_str:
                dt = datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
            else:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                
            risk_level = "Healthy"
            pred = row["predicted_class"]
            if pred == "Mild_Stress":
                risk_level = "Mild"
            elif pred == "Moderate_Stress":
                risk_level = "Moderate"
            elif pred == "Severe_Stress":
                risk_level = "Severe"
                
            recent.append({
                "patientId": row["patient_id"],
                "dateTime": dt.strftime("%b %d, %Y · %I:%M %p"),
                "riskLevel": risk_level,
                "status": "AI Reviewing" if row["deferred_to_human"] else "Completed"
            })
            
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "totalAssessments": total,
            "heroStats": [
                { "label": "Total Assessments", "value": f"{total:,}", "meta": "+12% this month" },
                { "label": "Healthy Cases", "value": str(healthy), "meta": f"{pct_healthy}% of total", "accentClassName": "border-l-secondary" },
                { "label": "Mild Stress", "value": str(mild), "meta": f"{pct_mild}% of total", "accentClassName": "border-l-primary-container" },
                { "label": "Moderate Stress", "value": str(moderate), "meta": f"{pct_mod}% of total", "accentClassName": "border-l-tertiary" },
                { "label": "Severe Stress", "value": str(severe), "meta": f"{pct_sev}% of total", "accentClassName": "border-l-error" }
            ],
            "stressDistribution": [
                { "label": "Healthy", "severity": "Healthy", "pct": pct_healthy },
                { "label": "Mild", "severity": "Mild", "pct": pct_mild },
                { "label": "Mod.", "severity": "Moderate", "pct": pct_mod },
                { "label": "Severe", "severity": "Severe", "pct": pct_sev }
            ],
            "trend": trend,
            "recentAssessments": recent,
            "execution_time_ms": round(execution_time_ms, 3)
        }
    finally:
        conn.close()

if __name__ == "__main__":
    metrics = derive_dashboard_metrics()
    print(json.dumps(metrics, indent=2))
