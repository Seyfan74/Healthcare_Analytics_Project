from pathlib import Path
import pandas as pd

# ========================================================= 
#          1. Dataset Path 
# # =========================================================
DATA_PATH = Path(
    r"C:\Users\seyfa\OneDrive\Desktop\Datanomicspython"
    r"\Advanced python\Final Capstone Project\module_work"
    r"\healthcare_project_dataset"
)


# ========================================================= 
#             2. Load One CSV File 
#  =========================================================

def load_csv(filename):
    file_path = DATA_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, low_memory=False)


# ========================================================= 
#  3. Load All Healthcare Data 
#  =========================================================

def load_all_data():
    data = {
        "patients": load_csv("patients.csv"),
        "admissions": load_csv("admissions.csv"),
        "dataset_summary": load_csv("dataset_summary.csv"),
        "departments": load_csv("departments.csv"),
        "diagnoses": load_csv("diagnoses.csv"),
        "diagnosis_records": load_csv("diagnosis_records.csv"),
        "doctors": load_csv("doctors.csv"),
        "insurance_claims": load_csv("insurance_claims.csv"),
        "lab_results": load_csv("lab_results.csv"),
        "lab_tests": load_csv("lab_tests.csv"),
        "medication_records": load_csv("medication_records.csv"),
        "medications": load_csv("medications.csv"),
        "payments": load_csv("payments.csv"),
        "procedure_records": load_csv("procedure_records.csv"),
        "procedures": load_csv("procedures.csv")
    }

    return data