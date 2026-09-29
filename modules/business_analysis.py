import pandas as pd


# ============================================================
# HEALTHCARE BUSINESS ANALYSIS
# ============================================================


# ============================================================
# 1. DEPARTMENT PERFORMANCE ANALYSIS
# ============================================================

def department_performance_analysis(data):

    admissions = data["admissions"].copy()
    departments = data["departments"].copy()

    admissions["length_of_stay"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

    admissions["patient_satisfaction"] = pd.to_numeric(
        admissions["patient_satisfaction"],
        errors="coerce"
    )

    department_analysis = (
        admissions
        .groupby("department_id")
        .agg(
            total_admissions=("admission_id", "count"),
            unique_patients=("patient_id", "nunique"),
            average_length_of_stay=("length_of_stay", "mean"),
            average_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    department_analysis = department_analysis.merge(
        departments[
            [
                "department_id",
                "department_name"
            ]
        ],
        on="department_id",
        how="left"
    )

    department_analysis["average_length_of_stay"] = (
        department_analysis["average_length_of_stay"]
        .round(2)
    )

    department_analysis["average_satisfaction"] = (
        department_analysis["average_satisfaction"]
        .round(2)
    )

    department_analysis = department_analysis[
        [
            "department_id",
            "department_name",
            "total_admissions",
            "unique_patients",
            "average_length_of_stay",
            "average_satisfaction"
        ]
    ]

    return department_analysis.sort_values(
        "total_admissions",
        ascending=False
    )


# ============================================================
# 2. ADMISSION TYPE COMPARISON
# ============================================================

def admission_type_comparison(data):

    admissions = data["admissions"].copy()

    admissions["length_of_stay"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

    admissions["patient_satisfaction"] = pd.to_numeric(
        admissions["patient_satisfaction"],
        errors="coerce"
    )

    comparison = (
        admissions
        .groupby("admission_type")
        .agg(
            total_admissions=("admission_id", "count"),
            unique_patients=("patient_id", "nunique"),
            average_length_of_stay=("length_of_stay", "mean"),
            average_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    comparison["average_length_of_stay"] = (
        comparison["average_length_of_stay"]
        .round(2)
    )

    comparison["average_satisfaction"] = (
        comparison["average_satisfaction"]
        .round(2)
    )

    return comparison.sort_values(
        "total_admissions",
        ascending=False
    )


# ============================================================
# 3. READMISSION ANALYSIS
# ============================================================

def readmission_analysis(data):

    admissions = data["admissions"].copy()

    admissions["length_of_stay"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

    admissions["patient_satisfaction"] = pd.to_numeric(
        admissions["patient_satisfaction"],
        errors="coerce"
    )

    comparison = (
        admissions
        .groupby("readmission_30_days")
        .agg(
            total_admissions=("admission_id", "count"),
            unique_patients=("patient_id", "nunique"),
            average_length_of_stay=("length_of_stay", "mean"),
            average_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    comparison["average_length_of_stay"] = (
        comparison["average_length_of_stay"]
        .round(2)
    )

    comparison["average_satisfaction"] = (
        comparison["average_satisfaction"]
        .round(2)
    )

    comparison["percentage_of_admissions"] = (
        comparison["total_admissions"]
        / comparison["total_admissions"].sum()
        * 100
    ).round(2)

    return comparison


# ============================================================
# 4. INSURANCE PERFORMANCE COMPARISON
# ============================================================

def insurance_performance_analysis(data):

    claims = data["insurance_claims"].copy()

    claims["billed_amount"] = pd.to_numeric(
        claims["billed_amount"],
        errors="coerce"
    )

    claims["approved_amount"] = pd.to_numeric(
        claims["approved_amount"],
        errors="coerce"
    )

    claims["denied_amount"] = pd.to_numeric(
        claims["denied_amount"],
        errors="coerce"
    )

    comparison = (
        claims
        .groupby("insurance_type")
        .agg(
            total_claims=("claim_id", "count"),
            total_billed=("billed_amount", "sum"),
            total_approved=("approved_amount", "sum"),
            total_denied=("denied_amount", "sum"),
            average_billed=("billed_amount", "mean"),
            average_approved=("approved_amount", "mean"),
            average_denied=("denied_amount", "mean")
        )
        .reset_index()
    )

    comparison["approval_rate"] = (
        comparison["total_approved"]
        / comparison["total_billed"]
        * 100
    ).round(2)

    comparison["denial_rate"] = (
        comparison["total_denied"]
        / comparison["total_billed"]
        * 100
    ).round(2)

    comparison = comparison.round(2)

    return comparison.sort_values(
        "total_billed",
        ascending=False
    )


# ============================================================
# 5. PAYMENT METHOD COMPARISON
# ============================================================

def payment_method_comparison(data):

    payments = data["payments"].copy()

    payments["payment_amount"] = pd.to_numeric(
        payments["payment_amount"],
        errors="coerce"
    )

    comparison = (
        payments
        .groupby("payment_method")
        .agg(
            total_payments=("payment_id", "count"),
            total_payment_amount=("payment_amount", "sum"),
            average_payment=("payment_amount", "mean"),
            median_payment=("payment_amount", "median")
        )
        .reset_index()
    )

    comparison["percentage_of_payments"] = (
        comparison["total_payments"]
        / comparison["total_payments"].sum()
        * 100
    ).round(2)

    comparison = comparison.round(2)

    return comparison.sort_values(
        "total_payment_amount",
        ascending=False
    )


# ============================================================
# 6. PROCEDURE PERFORMANCE ANALYSIS
# ============================================================

def procedure_performance_analysis(data):

    records = data["procedure_records"].copy()
    procedures = data["procedures"].copy()

    records["actual_cost"] = pd.to_numeric(
        records["actual_cost"],
        errors="coerce"
    )

    performance = (
        records
        .groupby("procedure_id")
        .agg(
            total_procedures=("procedure_record_id", "count"),
            total_cost=("actual_cost", "sum"),
            average_cost=("actual_cost", "mean")
        )
        .reset_index()
    )

    performance["successful_count"] = (
        records[
            records["outcome"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("successful")
        ]
        .groupby("procedure_id")
        .size()
    ).reindex(
        performance["procedure_id"],
        fill_value=0
    ).values

    performance["complication_count"] = (
        records[
            records["outcome"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("complication")
        ]
        .groupby("procedure_id")
        .size()
    ).reindex(
        performance["procedure_id"],
        fill_value=0
    ).values

    performance["success_rate"] = (
        performance["successful_count"]
        / performance["total_procedures"]
        * 100
    ).round(2)

    performance["complication_rate"] = (
        performance["complication_count"]
        / performance["total_procedures"]
        * 100
    ).round(2)

    performance = performance.merge(
        procedures[
            [
                "procedure_id",
                "procedure_name",
                "procedure_category",
                "base_cost"
            ]
        ],
        on="procedure_id",
        how="left"
    )

    performance["total_cost"] = performance["total_cost"].round(2)
    performance["average_cost"] = performance["average_cost"].round(2)

    return performance.sort_values(
        "total_procedures",
        ascending=False
    )


# ============================================================
# 7. PROCEDURE CATEGORY COMPARISON
# ============================================================

def procedure_category_comparison(data):

    records = data["procedure_records"].copy()
    procedures = data["procedures"].copy()

    records["actual_cost"] = pd.to_numeric(
        records["actual_cost"],
        errors="coerce"
    )

    merged = records.merge(
        procedures[
            [
                "procedure_id",
                "procedure_category"
            ]
        ],
        on="procedure_id",
        how="left"
    )

    comparison = (
        merged
        .groupby("procedure_category")
        .agg(
            total_procedures=("procedure_record_id", "count"),
            total_cost=("actual_cost", "sum"),
            average_cost=("actual_cost", "mean")
        )
        .reset_index()
    )

    comparison["percentage_of_procedures"] = (
        comparison["total_procedures"]
        / comparison["total_procedures"].sum()
        * 100
    ).round(2)

    comparison["total_cost"] = comparison["total_cost"].round(2)
    comparison["average_cost"] = comparison["average_cost"].round(2)

    return comparison.sort_values(
        "total_cost",
        ascending=False
    )


# ============================================================
# 8. DOCTOR WORKLOAD COMPARISON
# ============================================================

def doctor_workload_comparison(data):

    admissions = data["admissions"].copy()
    doctors = data["doctors"].copy()

    workload = (
        admissions
        .groupby("doctor_id")
        .agg(
            total_admissions=("admission_id", "count"),
            unique_patients=("patient_id", "nunique"),
            average_length_of_stay=("length_of_stay", "mean"),
            average_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    workload = workload.merge(
        doctors[
            [
                "doctor_id",
                "specialty",
                "department_id"
            ]
        ],
        on="doctor_id",
        how="left"
    )

    workload["average_length_of_stay"] = (
        workload["average_length_of_stay"]
        .round(2)
    )

    workload["average_satisfaction"] = (
        workload["average_satisfaction"]
        .round(2)
    )

    return workload.sort_values(
        "total_admissions",
        ascending=False
    )


# ============================================================
# 9. DIAGNOSIS ANALYSIS BY ADMISSION
# ============================================================

def diagnosis_business_analysis(data):

    records = data["diagnosis_records"].copy()
    diagnoses = data["diagnoses"].copy()

    analysis = (
        records
        .groupby("diagnosis_id")
        .agg(
            total_records=("diagnosis_id", "count"),
            unique_admissions=("admission_id", "nunique")
        )
        .reset_index()
    )

    analysis = analysis.merge(
        diagnoses[
            [
                "diagnosis_id",
                "diagnosis_name",
                "diagnosis_category"
            ]
        ],
        on="diagnosis_id",
        how="left"
    )

    analysis["percentage_of_diagnoses"] = (
        analysis["total_records"]
        / analysis["total_records"].sum()
        * 100
    ).round(2)

    return analysis.sort_values(
        "total_records",
        ascending=False
    )


# ============================================================
# 10. LABORATORY TEST COMPARISON
# ============================================================

def laboratory_business_analysis(data):

    results = data["lab_results"].copy()
    tests = data["lab_tests"].copy()

    abnormal_flag = (
        results["abnormal_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    results["is_abnormal"] = abnormal_flag.isin(
        [
            "yes",
            "true",
            "abnormal",
            "1"
        ]
    )

    comparison = (
        results
        .groupby("lab_test_id")
        .agg(
            total_tests=("lab_result_id", "count"),
            abnormal_results=("is_abnormal", "sum")
        )
        .reset_index()
    )

    comparison["abnormal_rate"] = (
        comparison["abnormal_results"]
        / comparison["total_tests"]
        * 100
    ).round(2)

    comparison = comparison.merge(
        tests[
            [
                "lab_test_id",
                "lab_test_name",
                "category"
            ]
        ],
        on="lab_test_id",
        how="left"
    )

    return comparison.sort_values(
        "abnormal_rate",
        ascending=False
    )


# ============================================================
# 11. PATIENT INSURANCE COMPARISON
# ============================================================

def patient_insurance_comparison(data):

    patients = data["patients"].copy()

    comparison = (
        patients
        .groupby("insurance_type")
        .agg(
            total_patients=("patient_id", "nunique")
        )
        .reset_index()
    )

    comparison["percentage_of_patients"] = (
        comparison["total_patients"]
        / comparison["total_patients"].sum()
        * 100
    ).round(2)

    return comparison.sort_values(
        "total_patients",
        ascending=False
    )


# ============================================================
# 12. GENDER BUSINESS COMPARISON
# ============================================================

def gender_business_comparison(data):

    patients = data["patients"].copy()
    admissions = data["admissions"].copy()

    admissions["length_of_stay"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

    admissions["patient_satisfaction"] = pd.to_numeric(
        admissions["patient_satisfaction"],
        errors="coerce"
    )

    patient_gender = patients[
        [
            "patient_id",
            "gender"
        ]
    ].drop_duplicates()

    merged = admissions.merge(
        patient_gender,
        on="patient_id",
        how="left"
    )

    comparison = (
        merged
        .groupby("gender")
        .agg(
            total_admissions=("admission_id", "count"),
            unique_patients=("patient_id", "nunique"),
            average_length_of_stay=("length_of_stay", "mean"),
            average_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    comparison["average_length_of_stay"] = (
        comparison["average_length_of_stay"]
        .round(2)
    )

    comparison["average_satisfaction"] = (
        comparison["average_satisfaction"]
        .round(2)
    )

    return comparison


# ============================================================
# 13. BUSINESS SUMMARY
# ============================================================

def business_summary(data):

    admissions = data["admissions"].copy()
    claims = data["insurance_claims"].copy()
    payments = data["payments"].copy()
    procedures = data["procedure_records"].copy()

    admissions["length_of_stay"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

    claims["billed_amount"] = pd.to_numeric(
        claims["billed_amount"],
        errors="coerce"
    )

    claims["approved_amount"] = pd.to_numeric(
        claims["approved_amount"],
        errors="coerce"
    )

    payments["payment_amount"] = pd.to_numeric(
        payments["payment_amount"],
        errors="coerce"
    )

    procedures["actual_cost"] = pd.to_numeric(
        procedures["actual_cost"],
        errors="coerce"
    )

    return {
        "total_patients": data["patients"]["patient_id"].nunique(),

        "total_admissions": len(admissions),

        "average_length_of_stay": round(
            admissions["length_of_stay"].mean(),
            2
        ),

        "total_claims": len(claims),

        "total_billed_amount": round(
            claims["billed_amount"].sum(),
            2
        ),

        "total_approved_amount": round(
            claims["approved_amount"].sum(),
            2
        ),

        "total_payments": len(payments),

        "total_payment_amount": round(
            payments["payment_amount"].sum(),
            2
        ),

        "total_procedure_records": len(procedures),

        "total_procedure_cost": round(
            procedures["actual_cost"].sum(),
            2
        )
    }


# ============================================================
# 14. RUN ALL BUSINESS ANALYSIS
# ============================================================

def run_business_analysis(data):

    results = {

        "business_summary":
            business_summary(data),

        "department_performance":
            department_performance_analysis(data),

        "admission_type_comparison":
            admission_type_comparison(data),

        "readmission_analysis":
            readmission_analysis(data),

        "insurance_performance":
            insurance_performance_analysis(data),

        "payment_method_comparison":
            payment_method_comparison(data),

        "procedure_performance":
            procedure_performance_analysis(data),

        "procedure_category_comparison":
            procedure_category_comparison(data),

        "doctor_workload":
            doctor_workload_comparison(data),

        "diagnosis_business_analysis":
            diagnosis_business_analysis(data),

        "laboratory_business_analysis":
            laboratory_business_analysis(data),

        "patient_insurance_comparison":
            patient_insurance_comparison(data),

        "gender_business_comparison":
            gender_business_comparison(data)
    }

    return results