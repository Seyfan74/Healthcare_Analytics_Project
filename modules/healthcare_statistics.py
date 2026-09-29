import pandas as pd


# ============================================================
# 1. ADMISSIONS STATISTICS
# ============================================================

def admissions_statistics(data):

    admissions = data["admissions"].copy()

    unique_patients = admissions["patient_id"].nunique()

    results = {
        "total_admissions": len(admissions),
        "unique_patients": unique_patients,
        "average_admissions_per_patient": (
            round(len(admissions) / unique_patients, 2)
            if unique_patients > 0
            else 0
        )
    }

    return results


# ============================================================
# 2. PATIENT STATISTICS
# ============================================================

def patient_statistics(data):

    patients = data["patients"].copy()

    # Convert date of birth to datetime
    patients["date_of_birth"] = pd.to_datetime(
        patients["date_of_birth"],
        errors="coerce"
    )

    # Count invalid dates
    invalid_dob_count = patients["date_of_birth"].isna().sum()

    # Calculate age
    today = pd.Timestamp.today()

    patients["age"] = (
        today.year
        - patients["date_of_birth"].dt.year
        - (
            (today.month < patients["date_of_birth"].dt.month)
            |
            (
                (today.month == patients["date_of_birth"].dt.month)
                &
                (today.day < patients["date_of_birth"].dt.day)
            )
        ).astype("Int64")
    )

    results = {
        "total_patients": len(patients),

        "average_age": (
            round(patients["age"].mean(), 2)
            if patients["age"].notna().any()
            else None
        ),

        "median_age": (
            round(patients["age"].median(), 2)
            if patients["age"].notna().any()
            else None
        ),

        "minimum_age": (
            int(patients["age"].min())
            if patients["age"].notna().any()
            else None
        ),

        "maximum_age": (
            int(patients["age"].max())
            if patients["age"].notna().any()
            else None
        ),

        "invalid_date_of_birth": int(invalid_dob_count),

        "gender_distribution": (
            patients["gender"]
            .value_counts(dropna=False)
            .to_dict()
        ),

        "insurance_distribution": (
            patients["insurance_type"]
            .value_counts(dropna=False)
            .to_dict()
        ),

        "smoking_status_distribution": (
            patients["smoking_status"]
            .value_counts(dropna=False)
            .to_dict()
        ),

        "blood_type_distribution": (
            patients["blood_type"]
            .value_counts(dropna=False)
            .to_dict()
        ),

        "state_distribution": (
            patients["state"]
            .value_counts(dropna=False)
            .to_dict()
        )
    }

    return results


# ============================================================
# 3. INSURANCE CLAIMS STATISTICS
# ============================================================

def insurance_statistics(data):

    insurance_claims = data["insurance_claims"].copy()

    for column in [
        "billed_amount",
        "approved_amount",
        "denied_amount"
    ]:

        if column in insurance_claims.columns:

            insurance_claims[column] = pd.to_numeric(
                insurance_claims[column],
                errors="coerce"
            )

    results = {
        "total_claims": len(insurance_claims)
    }

    if "billed_amount" in insurance_claims.columns:

        results["total_billed_amount"] = round(
            insurance_claims["billed_amount"].sum(), 2
        )

        results["average_billed_amount"] = round(
            insurance_claims["billed_amount"].mean(), 2
        )

    if "approved_amount" in insurance_claims.columns:

        results["total_approved_amount"] = round(
            insurance_claims["approved_amount"].sum(), 2
        )

        results["average_approved_amount"] = round(
            insurance_claims["approved_amount"].mean(), 2
        )

    if "denied_amount" in insurance_claims.columns:

        results["total_denied_amount"] = round(
            insurance_claims["denied_amount"].sum(), 2
        )

        results["average_denied_amount"] = round(
            insurance_claims["denied_amount"].mean(), 2
        )

    return results


# ============================================================
# 4. PAYMENT STATISTICS
# ============================================================

def payment_statistics(data):

    payments = data["payments"].copy()

    if "payment_amount" in payments.columns:

        payments["payment_amount"] = pd.to_numeric(
            payments["payment_amount"],
            errors="coerce"
        )

    results = {
        "total_payments": len(payments)
    }

    if "payment_amount" in payments.columns:

        results["total_payment_amount"] = round(
            payments["payment_amount"].sum(), 2
        )

        results["average_payment_amount"] = round(
            payments["payment_amount"].mean(), 2
        )

        results["median_payment_amount"] = round(
            payments["payment_amount"].median(), 2
        )

    if "payment_method" in payments.columns:

        results["payment_method_distribution"] = (
            payments["payment_method"]
            .value_counts(dropna=False)
            .to_dict()
        )

    if "payment_status" in payments.columns:

        results["payment_status_distribution"] = (
            payments["payment_status"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return results


# ============================================================
# 5. LABORATORY STATISTICS
# ============================================================

def laboratory_statistics(data):

    lab_results = data["lab_results"].copy()

    results = {
        "total_lab_results": len(lab_results)
    }

    if "abnormal_flag" in lab_results.columns:

        results["abnormal_flag_distribution"] = (
            lab_results["abnormal_flag"]
            .value_counts(dropna=False)
            .to_dict()
        )

    if "result_value" in lab_results.columns:

        lab_results["result_value"] = pd.to_numeric(
            lab_results["result_value"],
            errors="coerce"
        )

        results["average_result_value"] = round(
            lab_results["result_value"].mean(), 2
        )

        results["median_result_value"] = round(
            lab_results["result_value"].median(), 2
        )

    return results


# ============================================================
# 6. PROCEDURE STATISTICS
# ============================================================

def procedure_statistics(data):

    procedure_records = data["procedure_records"].copy()

    results = {
        "total_procedures": len(procedure_records),

        "unique_admissions": (
            procedure_records["admission_id"].nunique()
            if "admission_id" in procedure_records.columns
            else None
        )
    }

    if "actual_cost" in procedure_records.columns:

        procedure_records["actual_cost"] = pd.to_numeric(
            procedure_records["actual_cost"],
            errors="coerce"
        )

        results["total_procedure_cost"] = round(
            procedure_records["actual_cost"].sum(), 2
        )

        results["average_procedure_cost"] = round(
            procedure_records["actual_cost"].mean(), 2
        )

        results["median_procedure_cost"] = round(
            procedure_records["actual_cost"].median(), 2
        )

    if "outcome" in procedure_records.columns:

        results["outcome_distribution"] = (
            procedure_records["outcome"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return results


# ============================================================
# 7. CALCULATE ALL STATISTICS
# ============================================================

def calculate_statistics(data):

    statistics_results = {

        "admissions":
            admissions_statistics(data),

        "patients":
            patient_statistics(data),

        "insurance_claims":
            insurance_statistics(data),

        "payments":
            payment_statistics(data),

        "lab_results":
            laboratory_statistics(data),

        "procedure_records":
            procedure_statistics(data)
    }

    return statistics_results