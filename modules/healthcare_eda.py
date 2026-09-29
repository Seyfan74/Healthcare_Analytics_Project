# ============================================================
# HEALTHCARE EDA MODULE
# healthcare_eda.py
# ============================================================

import pandas as pd


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_table(data, table_name):
    """
    Safely retrieve a table from the healthcare data dictionary.
    """

    df = data.get(table_name)

    if df is None:
        print(f"Warning: {table_name} not found in data.")

    return df


# ============================================================
# 1. ADMISSIONS EDA
# ============================================================

def admissions_eda(data):

    df = get_table(data, "admissions")

    if df is None or df.empty:
        return {"error": "Admissions data not available"}

    results = {}

    results["total_admissions"] = len(df)

    if "admission_id" in df.columns:
        results["unique_admissions"] = df["admission_id"].nunique()

    if "patient_id" in df.columns:
        results["unique_patients"] = df["patient_id"].nunique()

    if "doctor_id" in df.columns:
        results["unique_doctors"] = df["doctor_id"].nunique()

    if "department_id" in df.columns:
        results["unique_departments"] = df["department_id"].nunique()

    if "admission_type" in df.columns:
        results["admission_type_distribution"] = (
            df["admission_type"]
            .value_counts()
            .to_dict()
        )

    if "discharge_status" in df.columns:
        results["discharge_status_distribution"] = (
            df["discharge_status"]
            .value_counts()
            .to_dict()
        )

    if "room_type" in df.columns:
        results["room_type_distribution"] = (
            df["room_type"]
            .value_counts()
            .to_dict()
        )

    if "length_of_stay" in df.columns:
        results["average_length_of_stay"] = (
            pd.to_numeric(
                df["length_of_stay"],
                errors="coerce"
            ).mean()
        )

        results["median_length_of_stay"] = (
            pd.to_numeric(
                df["length_of_stay"],
                errors="coerce"
            ).median()
        )

        results["maximum_length_of_stay"] = (
            pd.to_numeric(
                df["length_of_stay"],
                errors="coerce"
            ).max()
        )

    if "patient_satisfaction" in df.columns:

        satisfaction = pd.to_numeric(
            df["patient_satisfaction"],
            errors="coerce"
        )

        results["average_patient_satisfaction"] = (
            satisfaction.mean()
        )

        results["median_patient_satisfaction"] = (
            satisfaction.median()
        )

    if "readmission_30_days" in df.columns:

        results["readmission_distribution"] = (
            df["readmission_30_days"]
            .value_counts()
            .to_dict()
        )

        readmission = pd.to_numeric(
            df["readmission_30_days"],
            errors="coerce"
        )

        results["readmission_rate"] = (
            readmission.mean() * 100
        )

    return results


# ============================================================
# 2. PATIENTS EDA
# ============================================================

def patients_eda(data):
    """Perform exploratory analysis on the patients table."""

    df = get_table(data, "patients")

    if df is None or df.empty:
        return {"error": "Patients data not available"}

    # Work on a copy so the EDA function never changes the original data.
    df = df.copy()

    results = {}

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------
    results["total_patients"] = len(df)

    if "patient_id" in df.columns:
        results["unique_patients"] = df["patient_id"].nunique()
        results["duplicate_patient_rows"] = (
            len(df) - df["patient_id"].nunique()
        )

    # --------------------------------------------------------
    # Categorical distributions
    # --------------------------------------------------------
    if "gender" in df.columns:
        results["gender_distribution"] = (
            df["gender"].value_counts(dropna=False).to_dict()
        )

    if "insurance_type" in df.columns:
        results["insurance_distribution"] = (
            df["insurance_type"].value_counts(dropna=False).to_dict()
        )

    if "smoking_status" in df.columns:
        results["smoking_distribution"] = (
            df["smoking_status"].value_counts(dropna=False).to_dict()
        )

    if "blood_type" in df.columns:
        results["blood_type_distribution"] = (
            df["blood_type"].value_counts(dropna=False).to_dict()
        )

    if "state" in df.columns:
        results["state_distribution"] = (
            df["state"].value_counts(dropna=False).to_dict()
        )

    if "city" in df.columns:
        results["city_distribution"] = (
            df["city"].value_counts(dropna=False).to_dict()
        )

    # --------------------------------------------------------
    # Date of birth validation
    # --------------------------------------------------------
    if "date_of_birth" in df.columns:
        dob = pd.to_datetime(
            df["date_of_birth"],
            errors="coerce"
        )

        results["valid_date_of_birth"] = int(dob.notna().sum())
        results["invalid_date_of_birth"] = int(dob.isna().sum())

    # --------------------------------------------------------
    # Missing-value summary
    # --------------------------------------------------------
    results["missing_values"] = (
        df.isnull().sum().to_dict()
    )

    return results


# ============================================================
# 3. DEPARTMENTS EDA
# ============================================================


def departments_eda(data):

    df = get_table(data, "departments")

    if df is None or df.empty:
        return {"error": "Departments data not available"}

    results = {}

    results["total_departments"] = len(df)

    if "department_id" in df.columns:
        results["unique_departments"] = (
            df["department_id"].nunique()
        )

    if "department_name" in df.columns:
        results["department_distribution"] = (
            df["department_name"]
            .value_counts()
            .to_dict()
        )

    if "location" in df.columns:
        results["location_distribution"] = (
            df["location"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 4. DOCTORS EDA
# ============================================================

def doctors_eda(data):

    df = get_table(data, "doctors")

    if df is None or df.empty:
        return {"error": "Doctors data not available"}

    results = {}

    results["total_doctors"] = len(df)

    if "doctor_id" in df.columns:
        results["unique_doctors"] = (
            df["doctor_id"].nunique()
        )

    if "department_id" in df.columns:
        results["doctors_by_department"] = (
            df["department_id"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    if "specialization" in df.columns:
        results["specialization_distribution"] = (
            df["specialization"]
            .value_counts()
            .to_dict()
        )

    if "gender" in df.columns:
        results["gender_distribution"] = (
            df["gender"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 5. DIAGNOSES EDA
# ============================================================

def diagnoses_eda(data):

    df = get_table(data, "diagnoses")

    if df is None or df.empty:
        return {"error": "Diagnoses data not available"}

    results = {}

    results["total_diagnoses"] = len(df)

    if "diagnosis_id" in df.columns:
        results["unique_diagnoses"] = (
            df["diagnosis_id"].nunique()
        )

    if "diagnosis_name" in df.columns:
        results["diagnosis_distribution"] = (
            df["diagnosis_name"]
            .value_counts()
            .to_dict()
        )

    if "category" in df.columns:
        results["category_distribution"] = (
            df["category"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 6. DIAGNOSIS RECORDS EDA
# ============================================================

def diagnosis_records_eda(data):

    df = get_table(data, "diagnosis_records")

    if df is None or df.empty:
        return {"error": "Diagnosis records data not available"}

    results = {}

    results["total_records"] = len(df)

    if "diagnosis_record_id" in df.columns:
        results["unique_records"] = (
            df["diagnosis_record_id"].nunique()
        )

    if "admission_id" in df.columns:
        results["unique_admissions"] = (
            df["admission_id"].nunique()
        )

    if "patient_id" in df.columns:
        results["unique_patients"] = (
            df["patient_id"].nunique()
        )

    if "diagnosis_id" in df.columns:
        results["diagnosis_distribution"] = (
            df["diagnosis_id"]
            .value_counts()
            .to_dict()
        )

    if "is_primary" in df.columns:
        results["primary_diagnosis_distribution"] = (
            df["is_primary"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 7. INSURANCE CLAIMS EDA
# ============================================================

def insurance_claims_eda(df):
    result = {
        "record_count": len(df),
        "unique_claims": df["claim_id"].nunique(),
        "unique_patients": df["patient_id"].nunique(),
    }

    # --------------------------------------------------------
    # Financial columns
    # --------------------------------------------------------

    billed = pd.to_numeric(df["billed_amount"], errors="coerce")
    approved = pd.to_numeric(df["approved_amount"], errors="coerce")
    denied = pd.to_numeric(df["denied_amount"], errors="coerce")

    result["total_billed"] = billed.sum()
    result["total_approved"] = approved.sum()
    result["total_denied"] = denied.sum()

    result["average_billed"] = billed.mean()
    result["average_approved"] = approved.mean()
    result["average_denied"] = denied.mean()

    result["median_billed"] = billed.median()

    # --------------------------------------------------------
    # Approval and denial rates
    # --------------------------------------------------------

    total_billed = billed.sum()

    if total_billed > 0:
        result["approval_rate"] = (
            approved.sum() / total_billed
        ) * 100

        result["denial_rate"] = (
            denied.sum() / total_billed
        ) * 100
    else:
        result["approval_rate"] = 0
        result["denial_rate"] = 0

    # --------------------------------------------------------
    # Claim status
    # --------------------------------------------------------

    if "claim_status" in df.columns:
        result["claim_status_distribution"] = (
            df["claim_status"]
            .value_counts(dropna=False)
            .to_dict()
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    result["missing_values"] = (
        df.isnull().sum().to_dict()
    )

    return result


# ============================================================
# 8. PAYMENTS EDA
# ============================================================

def payments_eda(data):

    df = get_table(data, "payments")

    if df is None or df.empty:
        return {"error": "Payments data not available"}

    results = {}

    results["total_payments"] = len(df)

    if "payment_id" in df.columns:
        results["unique_payments"] = (
            df["payment_id"].nunique()
        )

    if "claim_id" in df.columns:
        results["unique_claims"] = (
            df["claim_id"].nunique()
        )

    if "patient_id" in df.columns:
        results["unique_patients"] = (
            df["patient_id"].nunique()
        )

    if "payment_method" in df.columns:
        results["payment_method_distribution"] = (
            df["payment_method"]
            .value_counts()
            .to_dict()
        )

    if "payment_status" in df.columns:
        results["payment_status_distribution"] = (
            df["payment_status"]
            .value_counts()
            .to_dict()
        )

    if "payment_amount" in df.columns:

        amount = pd.to_numeric(
            df["payment_amount"],
            errors="coerce"
        )

        results["total_payment_amount"] = amount.sum()
        results["average_payment_amount"] = amount.mean()
        results["median_payment_amount"] = amount.median()

    return results


# ============================================================
# 9. LAB TESTS EDA
# ============================================================

def lab_tests_eda(data):

    df = get_table(data, "lab_tests")

    if df is None or df.empty:
        return {"error": "Lab tests data not available"}

    results = {}

    results["total_lab_tests"] = len(df)

    if "lab_test_id" in df.columns:
        results["unique_lab_tests"] = (
            df["lab_test_id"].nunique()
        )

    if "test_name" in df.columns:
        results["test_name_distribution"] = (
            df["test_name"]
            .value_counts()
            .to_dict()
        )

    if "category" in df.columns:
        results["category_distribution"] = (
            df["category"]
            .value_counts()
            .to_dict()
        )

    if "normal_range" in df.columns:
        results["normal_range_distribution"] = (
            df["normal_range"]
            .value_counts()
            .to_dict()
        )

    if "unit" in df.columns:
        results["unit_distribution"] = (
            df["unit"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 10. LAB RESULTS EDA
# ============================================================

def lab_results_eda(data):

    df = get_table(data, "lab_results")

    if df is None or df.empty:
        return {"error": "Lab results data not available"}

    results = {}

    results["total_lab_results"] = len(df)

    if "lab_result_id" in df.columns:
        results["unique_lab_results"] = (
            df["lab_result_id"].nunique()
        )

    if "admission_id" in df.columns:
        results["unique_admissions"] = (
            df["admission_id"].nunique()
        )

    if "lab_test_id" in df.columns:
        results["unique_lab_tests"] = (
            df["lab_test_id"].nunique()
        )

    if "abnormal_flag" in df.columns:
        results["abnormal_distribution"] = (
            df["abnormal_flag"]
            .value_counts()
            .to_dict()
        )

    if "result_value" in df.columns:

        result_value = pd.to_numeric(
            df["result_value"],
            errors="coerce"
        )

        results["average_result_value"] = (
            result_value.mean()
        )

        results["median_result_value"] = (
            result_value.median()
        )

    if "result_unit" in df.columns:
        results["result_unit_distribution"] = (
            df["result_unit"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 11. MEDICATIONS EDA
# ============================================================

def medications_eda(data):

    df = get_table(data, "medications")

    if df is None or df.empty:
        return {"error": "Medications data not available"}

    results = {}

    results["total_medications"] = len(df)

    if "medication_id" in df.columns:
        results["unique_medications"] = (
            df["medication_id"].nunique()
        )

    if "medication_name" in df.columns:
        results["medication_distribution"] = (
            df["medication_name"]
            .value_counts()
            .to_dict()
        )

    if "category" in df.columns:
        results["category_distribution"] = (
            df["category"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 12. MEDICATION RECORDS EDA
# ============================================================

def medication_records_eda(data):

    df = get_table(data, "medication_records")

    if df is None or df.empty:
        return {"error": "Medication records data not available"}

    results = {}

    results["total_records"] = len(df)

    if "medication_record_id" in df.columns:
        results["unique_records"] = (
            df["medication_record_id"].nunique()
        )

    if "admission_id" in df.columns:
        results["unique_admissions"] = (
            df["admission_id"].nunique()
        )

    if "patient_id" in df.columns:
        results["unique_patients"] = (
            df["patient_id"].nunique()
        )

    if "medication_id" in df.columns:
        results["medication_distribution"] = (
            df["medication_id"]
            .value_counts()
            .to_dict()
        )

    if "frequency" in df.columns:
        results["frequency_distribution"] = (
            df["frequency"]
            .value_counts()
            .to_dict()
        )

    if "route" in df.columns:
        results["route_distribution"] = (
            df["route"]
            .value_counts()
            .to_dict()
        )

    if "dosage" in df.columns:

        dosage = pd.to_numeric(
            df["dosage"],
            errors="coerce"
        )

        results["average_dosage"] = dosage.mean()
        results["median_dosage"] = dosage.median()

    return results


# ============================================================
# 13. PROCEDURES EDA
# ============================================================

def procedures_eda(data):

    df = get_table(data, "procedures")

    if df is None or df.empty:
        return {"error": "Procedures data not available"}

    results = {}

    results["total_procedures"] = len(df)

    if "procedure_id" in df.columns:
        results["unique_procedures"] = (
            df["procedure_id"].nunique()
        )

    if "procedure_name" in df.columns:
        results["procedure_distribution"] = (
            df["procedure_name"]
            .value_counts()
            .to_dict()
        )

    if "category" in df.columns:
        results["category_distribution"] = (
            df["category"]
            .value_counts()
            .to_dict()
        )

    return results


# ============================================================
# 14. PROCEDURE RECORDS EDA
# ============================================================

def procedure_records_eda(data):

    df = get_table(data, "procedure_records")

    if df is None or df.empty:
        return {"error": "Procedure records data not available"}

    results = {}

    results["total_records"] = len(df)

    if "procedure_record_id" in df.columns:
        results["unique_records"] = (
            df["procedure_record_id"].nunique()
        )

    if "admission_id" in df.columns:
        results["unique_admissions"] = (
            df["admission_id"].nunique()
        )

    if "procedure_id" in df.columns:
        results["unique_procedures"] = (
            df["procedure_id"].nunique()
        )

    if "outcome" in df.columns:
        results["outcome_distribution"] = (
            df["outcome"]
            .value_counts()
            .to_dict()
        )

    if "actual_cost" in df.columns:

        cost = pd.to_numeric(
            df["actual_cost"],
            errors="coerce"
        )

        results["total_actual_cost"] = cost.sum()
        results["average_actual_cost"] = cost.mean()
        results["median_actual_cost"] = cost.median()

    return results


# ============================================================
# MAIN HEALTHCARE EDA FUNCTION
# ============================================================

def healthcare_eda(data):
    """
    Run EDA across all healthcare dataset tables.

    Parameters
    ----------
    data : dict
        Dictionary containing all healthcare DataFrames.

    Returns
    -------
    dict
        EDA results for each dataset/table.
    """

    results = {

        # ====================================================
        # HOSPITAL OPERATIONS
        # ====================================================

        "admissions": admissions_eda(data),

        "departments": departments_eda(data),

        "doctors": doctors_eda(data),

        # ====================================================
        # PATIENTS
        # ====================================================

        "patients": patients_eda(data),

        # ====================================================
        # DIAGNOSIS
        # ====================================================

        "diagnoses": diagnoses_eda(data),

        "diagnosis_records": diagnosis_records_eda(data),

        # ====================================================
        # INSURANCE AND FINANCIAL
        # ====================================================

        "insurance_claims": insurance_claims_eda(
            data["insurance_claims"]
        ),

        "payments": payments_eda(data),

        # ====================================================
        # LABORATORY
        # ====================================================

        "lab_tests": lab_tests_eda(data),

        "lab_results": lab_results_eda(data),

        # ====================================================
        # MEDICATION
        # ====================================================

        "medications": medications_eda(data),

        "medication_records": medication_records_eda(data),

        # ====================================================
        # PROCEDURES
        # ====================================================

        "procedures": procedures_eda(data),

        "procedure_records": procedure_records_eda(data)
    }

    return results