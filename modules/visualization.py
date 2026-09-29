# ============================================================
# HEALTHCARE END-TO-END ANALYTICS
# VISUALIZATION MODULE
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_numeric(series):
    """
    Convert a pandas Series to numeric values.
    Invalid values are converted to NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def calculate_length_of_stay(admissions):
    """
    Calculate length of stay in days from admission and
    discharge dates.
    """

    admissions = admissions.copy()

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"],
        errors="coerce"
    )

    admissions["discharge_date"] = pd.to_datetime(
        admissions["discharge_date"],
        errors="coerce"
    )

    admissions["length_of_stay"] = (
        admissions["discharge_date"]
        - admissions["admission_date"]
    ).dt.total_seconds() / 86400

    # Remove impossible negative values
    admissions.loc[
        admissions["length_of_stay"] < 0,
        "length_of_stay"
    ] = pd.NA

    return admissions


# ============================================================
# 1. PATIENT ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 1.1 Patient Gender Distribution
# ------------------------------------------------------------

def plot_gender_distribution(data, ax=None):

    patients = data["patients"].copy()

    gender_counts = (
        patients["gender"]
        .fillna("Unknown")
        .value_counts()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=gender_counts.index.astype(str),
        y=gender_counts.values,
        ax=ax
    )

    ax.set_title("Patient Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Patients")

    return ax


# ------------------------------------------------------------
# 1.2 Patient Age Distribution
# ------------------------------------------------------------

def plot_age_distribution(data, ax=None):

    patients = data["patients"].copy()

    patients["date_of_birth"] = pd.to_datetime(
        patients["date_of_birth"],
        errors="coerce"
    )

    patients = patients.dropna(
        subset=["date_of_birth"]
    )

    today = pd.Timestamp.today()

    patients["age"] = (
        today.year
        - patients["date_of_birth"].dt.year
    )

    birthday_not_reached = (
        (patients["date_of_birth"].dt.month > today.month)
        |
        (
            (patients["date_of_birth"].dt.month == today.month)
            &
            (patients["date_of_birth"].dt.day > today.day)
        )
    )

    patients.loc[
        birthday_not_reached,
        "age"
    ] -= 1

    # Keep realistic ages
    patients = patients[
        patients["age"].between(0, 120)
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        patients["age"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Patient Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Patients")

    return ax


# ------------------------------------------------------------
# 1.3 Patient Insurance Distribution
# ------------------------------------------------------------

def plot_insurance_distribution(data, ax=None):

    patients = data["patients"].copy()

    insurance_counts = (
        patients["insurance_type"]
        .fillna("Unknown")
        .value_counts()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=insurance_counts.index.astype(str),
        y=insurance_counts.values,
        ax=ax
    )

    ax.set_title("Patients by Insurance Type")
    ax.set_xlabel("Insurance Type")
    ax.set_ylabel("Number of Patients")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    return ax


# ------------------------------------------------------------
# 1.4 Smoking Status
# ------------------------------------------------------------

def plot_smoking_status(data, ax=None):

    patients = data["patients"].copy()

    smoking_counts = (
        patients["smoking_status"]
        .fillna("Unknown")
        .value_counts()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=smoking_counts.index.astype(str),
        y=smoking_counts.values,
        ax=ax
    )

    ax.set_title("Patient Smoking Status")
    ax.set_xlabel("Smoking Status")
    ax.set_ylabel("Number of Patients")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    return ax


# ------------------------------------------------------------
# 1.5 Multiple Admissions
# ------------------------------------------------------------

def plot_multiple_admissions(data, ax=None):

    admissions = data["admissions"].copy()

    admission_counts = (
        admissions
        .groupby("patient_id")
        .size()
    )

    multiple_admission_counts = pd.Series({
        "One Admission": (
            admission_counts == 1
        ).sum(),

        "Multiple Admissions": (
            admission_counts >= 2
        ).sum()
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=multiple_admission_counts.index,
        y=multiple_admission_counts.values,
        ax=ax
    )

    ax.set_title(
        "Patients by Admission Frequency"
    )

    ax.set_xlabel(
        "Admission Category"
    )

    ax.set_ylabel(
        "Number of Patients"
    )

    return ax


# ============================================================
# 2. HOSPITAL OPERATIONS
# ============================================================


# ------------------------------------------------------------
# 2.1 Admissions by Department
# ------------------------------------------------------------

def plot_admissions_by_department(data, ax=None):

    admissions = data["admissions"].copy()

    department_counts = (
        admissions["department_id"]
        .value_counts()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=department_counts.index.astype(str),
        y=department_counts.values,
        ax=ax
    )

    ax.set_title(
        "Admissions by Department"
    )

    ax.set_xlabel(
        "Department ID"
    )

    ax.set_ylabel(
        "Number of Admissions"
    )

    return ax


# ------------------------------------------------------------
# 2.2 Monthly Admissions
# ------------------------------------------------------------

def plot_monthly_admissions(data, ax=None):

    admissions = data["admissions"].copy()

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"],
        errors="coerce"
    )

    valid_admissions = admissions.dropna(
        subset=["admission_date"]
    )

    monthly_admissions = (
        valid_admissions
        .groupby(
            valid_admissions[
                "admission_date"
            ].dt.to_period("M")
        )
        .size()
    )

    monthly_admissions.index = (
        monthly_admissions.index.astype(str)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

    sns.lineplot(
        x=monthly_admissions.index,
        y=monthly_admissions.values,
        marker="o",
        ax=ax
    )

    ax.set_title(
        "Monthly Admissions Trend"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Number of Admissions"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    return ax


# ------------------------------------------------------------
# 2.3 Average Length of Stay by Department
# ------------------------------------------------------------

def plot_average_los_by_department(data, ax=None):

    admissions = calculate_length_of_stay(
        data["admissions"]
    )

    los_by_department = (
        admissions
        .dropna(
            subset=["length_of_stay"]
        )
        .groupby("department_id")[
            "length_of_stay"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        x=los_by_department.index.astype(str),
        y=los_by_department.values,
        ax=ax
    )

    ax.set_title(
        "Average Length of Stay by Department"
    )

    ax.set_xlabel(
        "Department ID"
    )

    ax.set_ylabel(
        "Average Length of Stay (Days)"
    )

    return ax


# ------------------------------------------------------------
# 2.4 Doctor Workload
# ------------------------------------------------------------

def plot_doctor_workload(data, ax=None):

    admissions = data["admissions"].copy()

    doctor_workload = (
        admissions
        .groupby("doctor_id")[
            "admission_id"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        x=doctor_workload.index.astype(str),
        y=doctor_workload.values,
        ax=ax
    )

    ax.set_title(
        "Top 10 Doctors by Admission Workload"
    )

    ax.set_xlabel(
        "Doctor ID"
    )

    ax.set_ylabel(
        "Number of Admissions"
    )

    return ax


# ============================================================
# 3. CLINICAL SERVICES
# ============================================================


# ------------------------------------------------------------
# 3.1 Top 10 Diagnoses
# ------------------------------------------------------------

def plot_top_diagnoses(data, ax=None):

    diagnosis_records = (
        data["diagnosis_records"].copy()
    )

    diagnoses = data["diagnoses"].copy()

    diagnosis_counts = (
        diagnosis_records
        .groupby("diagnosis_id")
        .size()
        .reset_index(
            name="count"
        )
    )

    if "diagnosis_name" in diagnoses.columns:

        diagnosis_counts = (
            diagnosis_counts.merge(
                diagnoses[
                    [
                        "diagnosis_id",
                        "diagnosis_name"
                    ]
                ],
                on="diagnosis_id",
                how="left"
            )
        )

        diagnosis_counts["label"] = (
            diagnosis_counts[
                "diagnosis_name"
            ]
            .fillna(
                diagnosis_counts[
                    "diagnosis_id"
                ].astype(str)
            )
        )

    else:

        diagnosis_counts["label"] = (
            diagnosis_counts[
                "diagnosis_id"
            ].astype(str)
        )

    top_diagnoses = (
        diagnosis_counts
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
        .sort_values("count")
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        data=top_diagnoses,
        x="count",
        y="label",
        ax=ax
    )

    ax.set_title(
        "Top 10 Diagnoses"
    )

    ax.set_xlabel(
        "Number of Diagnosis Records"
    )

    ax.set_ylabel(
        "Diagnosis"
    )

    return ax


# ------------------------------------------------------------
# 3.2 Top 10 Laboratory Tests
# ------------------------------------------------------------

def plot_top_lab_tests(data, ax=None):

    lab_results = data["lab_results"].copy()
    lab_tests = data["lab_tests"].copy()

    lab_counts = (
        lab_results
        .groupby("lab_test_id")
        .size()
        .reset_index(
            name="tests"
        )
    )

    if "test_name" in lab_tests.columns:

        lab_counts = (
            lab_counts.merge(
                lab_tests[
                    [
                        "lab_test_id",
                        "test_name"
                    ]
                ],
                on="lab_test_id",
                how="left"
            )
        )

        lab_counts["label"] = (
            lab_counts[
                "test_name"
            ]
            .fillna(
                lab_counts[
                    "lab_test_id"
                ].astype(str)
            )
        )

    else:

        lab_counts["label"] = (
            lab_counts[
                "lab_test_id"
            ].astype(str)
        )

    top_tests = (
        lab_counts
        .sort_values(
            "tests",
            ascending=False
        )
        .head(10)
        .sort_values("tests")
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        data=top_tests,
        x="tests",
        y="label",
        ax=ax
    )

    ax.set_title(
        "Top 10 Laboratory Tests"
    )

    ax.set_xlabel(
        "Number of Tests"
    )

    ax.set_ylabel(
        "Laboratory Test"
    )

    return ax


# ------------------------------------------------------------
# 3.3 Abnormal Laboratory Results
# ------------------------------------------------------------

def plot_abnormal_lab_results(data, ax=None):

    lab_results = data["lab_results"].copy()

    if "abnormal_flag" not in lab_results.columns:

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

        ax.text(
            0.5,
            0.5,
            "Column 'abnormal_flag' was not found.",
            ha="center",
            va="center"
        )

        ax.set_axis_off()

        return ax

    abnormal_counts = (
        lab_results[
            "abnormal_flag"
        ]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

    sns.barplot(
        x=abnormal_counts.index,
        y=abnormal_counts.values,
        ax=ax
    )

    ax.set_title(
        "Laboratory Results by Abnormal Status"
    )

    ax.set_xlabel(
        "Abnormal Flag"
    )

    ax.set_ylabel(
        "Number of Results"
    )

    return ax


# ------------------------------------------------------------
# 3.4 Top 10 Medications
# ------------------------------------------------------------

def plot_top_medications(data, ax=None):

    medication_records = (
        data["medication_records"].copy()
    )

    medications = (
        data["medications"].copy()
    )

    medication_counts = (
        medication_records
        .groupby("medication_id")
        .size()
        .reset_index(
            name="records"
        )
    )

    if "medication_name" in medications.columns:

        medication_counts = (
            medication_counts.merge(
                medications[
                    [
                        "medication_id",
                        "medication_name"
                    ]
                ],
                on="medication_id",
                how="left"
            )
        )

        medication_counts["label"] = (
            medication_counts[
                "medication_name"
            ]
            .fillna(
                medication_counts[
                    "medication_id"
                ].astype(str)
            )
        )

    else:

        medication_counts["label"] = (
            medication_counts[
                "medication_id"
            ].astype(str)
        )

    top_medications = (
        medication_counts
        .sort_values(
            "records",
            ascending=False
        )
        .head(10)
        .sort_values("records")
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        data=top_medications,
        x="records",
        y="label",
        ax=ax
    )

    ax.set_title(
        "Top 10 Medications"
    )

    ax.set_xlabel(
        "Number of Medication Records"
    )

    ax.set_ylabel(
        "Medication"
    )

    return ax


# ------------------------------------------------------------
# 3.5 Top 10 Procedures
# ------------------------------------------------------------

def plot_top_procedures(data, ax=None):

    procedure_records = (
        data["procedure_records"].copy()
    )

    procedures = data["procedures"].copy()

    procedure_counts = (
        procedure_records
        .groupby("procedure_id")
        .size()
        .reset_index(
            name="procedures"
        )
    )

    if "procedure_name" in procedures.columns:

        procedure_counts = (
            procedure_counts.merge(
                procedures[
                    [
                        "procedure_id",
                        "procedure_name"
                    ]
                ],
                on="procedure_id",
                how="left"
            )
        )

        procedure_counts["label"] = (
            procedure_counts[
                "procedure_name"
            ]
            .fillna(
                procedure_counts[
                    "procedure_id"
                ].astype(str)
            )
        )

    else:

        procedure_counts["label"] = (
            procedure_counts[
                "procedure_id"
            ].astype(str)
        )

    top_procedures = (
        procedure_counts
        .sort_values(
            "procedures",
            ascending=False
        )
        .head(10)
        .sort_values("procedures")
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        data=top_procedures,
        x="procedures",
        y="label",
        ax=ax
    )

    ax.set_title(
        "Top 10 Procedures"
    )

    ax.set_xlabel(
        "Number of Procedures"
    )

    ax.set_ylabel(
        "Procedure"
    )

    return ax


# ============================================================
# 4. INSURANCE AND FINANCIAL ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 4.1 Insurance Claim Status
# ------------------------------------------------------------

def plot_claim_status(data, ax=None):

    claims = data["insurance_claims"].copy()

    claim_counts = (
        claims["claim_status"]
        .fillna("Unknown")
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

    sns.barplot(
        x=claim_counts.index.astype(str),
        y=claim_counts.values,
        ax=ax
    )

    ax.set_title(
        "Insurance Claims by Status"
    )

    ax.set_xlabel(
        "Claim Status"
    )

    ax.set_ylabel(
        "Number of Claims"
    )

    return ax


# ------------------------------------------------------------
# 4.2 Billed vs Approved Amount
# ------------------------------------------------------------

def plot_billed_vs_approved(data, ax=None):

    claims = data["insurance_claims"].copy()

    claims["billed_amount"] = (
        prepare_numeric(
            claims["billed_amount"]
        )
    )

    claims["approved_amount"] = (
        prepare_numeric(
            claims["approved_amount"]
        )
    )

    financial_summary = pd.Series({
        "Billed":
            claims["billed_amount"].sum(),

        "Approved":
            claims["approved_amount"].sum()
    })

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

    sns.barplot(
        x=financial_summary.index,
        y=financial_summary.values,
        ax=ax
    )

    ax.set_title(
        "Total Billed vs Approved Amount"
    )

    ax.set_xlabel(
        "Financial Measure"
    )

    ax.set_ylabel(
        "Amount"
    )

    return ax


# ------------------------------------------------------------
# 4.3 Payment Amount Distribution
# ------------------------------------------------------------

def plot_payment_distribution(data, ax=None):

    payments = data["payments"].copy()

    payments["payment_amount"] = (
        prepare_numeric(
            payments["payment_amount"]
        )
    )

    payment_values = (
        payments[
            "payment_amount"
        ].dropna()
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.histplot(
        payment_values,
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Payment Amount Distribution"
    )

    ax.set_xlabel(
        "Payment Amount"
    )

    ax.set_ylabel(
        "Number of Payments"
    )

    return ax


# ------------------------------------------------------------
# 4.4 Payment by Method
# ------------------------------------------------------------

def plot_payment_by_method(data, ax=None):

    payments = data["payments"].copy()

    payments["payment_amount"] = (
        prepare_numeric(
            payments["payment_amount"]
        )
    )

    payment_by_method = (
        payments
        .dropna(
            subset=["payment_method"]
        )
        .groupby(
            "payment_method"
        )["payment_amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        x=payment_by_method.index.astype(str),
        y=payment_by_method.values,
        ax=ax
    )

    ax.set_title(
        "Total Payment Amount by Payment Method"
    )

    ax.set_xlabel(
        "Payment Method"
    )

    ax.set_ylabel(
        "Total Payment Amount"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    return ax


# ------------------------------------------------------------
# 4.5 Financial Activity by Department
# ------------------------------------------------------------

def plot_financial_activity_by_department(
    data,
    ax=None
):

    claims = data["insurance_claims"].copy()
    admissions = data["admissions"].copy()

    claims["billed_amount"] = (
        prepare_numeric(
            claims["billed_amount"]
        )
    )

    department_finance = (
        claims
        .merge(
            admissions[
                [
                    "admission_id",
                    "department_id"
                ]
            ],
            on="admission_id",
            how="left"
        )
        .groupby(
            "department_id"
        )["billed_amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.barplot(
        x=department_finance.index.astype(str),
        y=department_finance.values,
        ax=ax
    )

    ax.set_title(
        "Billed Amount by Department"
    )

    ax.set_xlabel(
        "Department ID"
    )

    ax.set_ylabel(
        "Total Billed Amount"
    )

    return ax


# ============================================================
# 5. INTEGRATED HEALTHCARE-FINANCIAL ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 5.1 Length of Stay vs Paid Amount
# ------------------------------------------------------------

def plot_los_vs_paid_amount(data, ax=None):

    admissions = calculate_length_of_stay(
        data["admissions"]
    )

    claims = data["insurance_claims"].copy()
    payments = data["payments"].copy()

    payments["payment_amount"] = (
        prepare_numeric(
            payments["payment_amount"]
        )
    )

    payment_by_claim = (
        payments
        .dropna(
            subset=["claim_id"]
        )
        .groupby(
            "claim_id"
        )["payment_amount"]
        .sum()
        .reset_index()
    )

    integrated = (
        claims
        .merge(
            payment_by_claim,
            on="claim_id",
            how="left"
        )
        .merge(
            admissions[
                [
                    "admission_id",
                    "length_of_stay"
                ]
            ],
            on="admission_id",
            how="left"
        )
    )

    integrated = integrated.dropna(
        subset=[
            "length_of_stay",
            "payment_amount"
        ]
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.scatterplot(
        data=integrated,
        x="length_of_stay",
        y="payment_amount",
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        "Length of Stay vs Paid Amount"
    )

    ax.set_xlabel(
        "Length of Stay (Days)"
    )

    ax.set_ylabel(
        "Paid Amount"
    )

    return ax


# ============================================================
# PROCEDURES VS BILLED AMOUNT
# ============================================================

def plot_procedures_vs_billed_amount(data, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------------
    # Get tables
    # --------------------------------------------------------

    procedure_records = data["procedure_records"].copy()
    admissions = data["admissions"].copy()
    claims = data["insurance_claims"].copy()

    # --------------------------------------------------------
    # Clean numeric billed amount
    # --------------------------------------------------------

    claims["billed_amount"] = pd.to_numeric(
        claims["billed_amount"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Create admission -> patient mapping
    # --------------------------------------------------------

    patient_admissions = (
        admissions[
            ["admission_id", "patient_id"]
        ]
        .drop_duplicates(subset=["admission_id"])
    )

    # --------------------------------------------------------
    # Count procedures by patient
    # --------------------------------------------------------

    procedure_count = (
        procedure_records
        .merge(
            patient_admissions,
            on="admission_id",
            how="left"
        )
        .dropna(subset=["patient_id"])
        .groupby("patient_id")
        .size()
        .reset_index(name="procedure_count")
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # insurance_claims ALREADY contains patient_id
    # Therefore, do NOT merge claims with admissions here.
    # --------------------------------------------------------

    patient_billed = (
        claims
        .dropna(subset=["patient_id"])
        .groupby("patient_id")["billed_amount"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # Merge procedure count with billed amount
    # --------------------------------------------------------

    integrated = (
        procedure_count
        .merge(
            patient_billed,
            on="patient_id",
            how="inner"
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    sns.scatterplot(
        data=integrated,
        x="procedure_count",
        y="billed_amount",
        ax=ax
    )

    ax.set_title(
        "Procedures vs Billed Amount"
    )

    ax.set_xlabel(
        "Number of Procedures"
    )

    ax.set_ylabel(
        "Total Billed Amount ($)"
    )

    return ax


# ============================================================
# 5.3 PATIENT UTILIZATION INDEX
# ============================================================

def create_patient_utilization(data):

    patients = data["patients"].copy()
    admissions = data["admissions"].copy()
    diagnosis_records = (
        data["diagnosis_records"].copy()
    )
    lab_results = data["lab_results"].copy()
    medication_records = (
        data["medication_records"].copy()
    )
    procedure_records = (
        data["procedure_records"].copy()
    )

    patient_utilization = (
        patients[
            ["patient_id"]
        ]
        .drop_duplicates()
        .set_index("patient_id")
    )

    # --------------------------------------------------------
    # Admissions
    # --------------------------------------------------------

    admission_count = (
        admissions
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )
        .size()
        .rename("admissions")
    )

    # --------------------------------------------------------
    # Diagnoses
    # --------------------------------------------------------

    diagnosis_patient = (
        diagnosis_records
        .merge(
            admissions[
                [
                    "admission_id",
                    "patient_id"
                ]
            ],
            on="admission_id",
            how="left"
        )
    )

    diagnosis_count = (
        diagnosis_patient
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )
        .size()
        .rename("diagnoses")
    )

    # --------------------------------------------------------
    # Laboratory Results
    # --------------------------------------------------------

    lab_patient = (
        lab_results
        .merge(
            admissions[
                [
                    "admission_id",
                    "patient_id"
                ]
            ],
            on="admission_id",
            how="left"
        )
    )

    lab_count = (
        lab_patient
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )
        .size()
        .rename("lab_results")
    )

    # --------------------------------------------------------
    # Medications
    # --------------------------------------------------------

    medication_patient = (
        medication_records
        .merge(
            admissions[
                [
                    "admission_id",
                    "patient_id"
                ]
            ],
            on="admission_id",
            how="left"
        )
    )

    medication_count = (
        medication_patient
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )
        .size()
        .rename("medications")
    )

    # --------------------------------------------------------
    # Procedures
    # --------------------------------------------------------

    procedure_patient = (
        procedure_records
        .merge(
            admissions[
                [
                    "admission_id",
                    "patient_id"
                ]
            ],
            on="admission_id",
            how="left"
        )
    )

    procedure_count = (
        procedure_patient
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )
        .size()
        .rename("procedures")
    )

    # --------------------------------------------------------
    # Combine all utilization measures
    # --------------------------------------------------------

    patient_utilization = (
        patient_utilization
        .join(
            [
                admission_count,
                diagnosis_count,
                lab_count,
                medication_count,
                procedure_count
            ]
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Calculate utilization index
    # --------------------------------------------------------

    patient_utilization[
        "utilization_index"
    ] = (
        patient_utilization["admissions"]
        + patient_utilization["diagnoses"]
        + patient_utilization["lab_results"]
        + patient_utilization["medications"]
        + patient_utilization["procedures"]
    )

    return patient_utilization.reset_index()


# ------------------------------------------------------------
# 5.4 Utilization Index vs Length of Stay
# ------------------------------------------------------------

def plot_utilization_vs_los(data, ax=None):

    utilization = (
        create_patient_utilization(data)
    )

    admissions = calculate_length_of_stay(
        data["admissions"]
    )

    patient_los = (
        admissions
        .dropna(
            subset=["length_of_stay"]
        )
        .groupby(
            "patient_id"
        )["length_of_stay"]
        .mean()
        .reset_index()
    )

    integrated = (
        utilization
        .merge(
            patient_los,
            on="patient_id",
            how="inner"
        )
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.scatterplot(
        data=integrated,
        x="utilization_index",
        y="length_of_stay",
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        "Patient Utilization Index vs Average Length of Stay"
    )

    ax.set_xlabel(
        "Patient Utilization Index"
    )

    ax.set_ylabel(
        "Average Length of Stay (Days)"
    )

    return ax


# ------------------------------------------------------------
# 5.5 Utilization Index vs Paid Amount
# ------------------------------------------------------------

def plot_utilization_vs_paid_amount(
    data,
    ax=None
):

    utilization = (
        create_patient_utilization(data)
    )

    payments = data["payments"].copy()

    payments["payment_amount"] = (
        prepare_numeric(
            payments["payment_amount"]
        )
    )

    patient_paid = (
        payments
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )["payment_amount"]
        .sum()
        .reset_index()
    )

    integrated = (
        utilization
        .merge(
            patient_paid,
            on="patient_id",
            how="inner"
        )
    )

    integrated = integrated.dropna(
        subset=[
            "utilization_index",
            "payment_amount"
        ]
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.scatterplot(
        data=integrated,
        x="utilization_index",
        y="payment_amount",
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        "Patient Utilization Index vs Paid Amount"
    )

    ax.set_xlabel(
        "Patient Utilization Index"
    )

    ax.set_ylabel(
        "Total Paid Amount"
    )

    return ax


# ------------------------------------------------------------
# 5.6 Utilization Index vs Billed Amount
# ------------------------------------------------------------

def plot_utilization_vs_billed_amount(
    data,
    ax=None
):

    utilization = (
        create_patient_utilization(data)
    )

    claims = data["insurance_claims"].copy()

    claims["billed_amount"] = (
        prepare_numeric(
            claims["billed_amount"]
        )
    )

    # insurance_claims already contains patient_id
    patient_billed = (
        claims
        .dropna(
            subset=["patient_id"]
        )
        .groupby(
            "patient_id"
        )["billed_amount"]
        .sum()
        .reset_index()
    )

    integrated = (
        utilization
        .merge(
            patient_billed,
            on="patient_id",
            how="inner"
        )
    )

    integrated = integrated.dropna(
        subset=[
            "utilization_index",
            "billed_amount"
        ]
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    sns.scatterplot(
        data=integrated,
        x="utilization_index",
        y="billed_amount",
        alpha=0.6,
        ax=ax
    )

    ax.set_title(
        "Patient Utilization Index vs Billed Amount"
    )

    ax.set_xlabel(
        "Patient Utilization Index"
    )

    ax.set_ylabel(
        "Total Billed Amount"
    )

    return ax

# ============================================================
# 5.7 CORRELATION ANALYSIS
# ============================================================

def create_correlation_dataset(data):

    # --------------------------------------------------------
    # Admissions
    # --------------------------------------------------------

    admissions = calculate_length_of_stay(
        data["admissions"]
    )

    admissions["patient_satisfaction"] = prepare_numeric(
        admissions["patient_satisfaction"]
    )

    admissions["length_of_stay"] = prepare_numeric(
        admissions["length_of_stay"]
    )

    admission_summary = (
        admissions
        .groupby("patient_id")
        .agg(
            admissions=("admission_id", "nunique"),
            avg_length_of_stay=("length_of_stay", "mean"),
            avg_satisfaction=("patient_satisfaction", "mean")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Procedures
    # --------------------------------------------------------

    procedure_patient = (
        data["procedure_records"]
        .merge(
            data["admissions"][
                ["admission_id", "patient_id"]
            ],
            on="admission_id",
            how="left"
        )
    )

    procedure_summary = (
        procedure_patient
        .groupby("patient_id")
        .size()
        .reset_index(name="procedure_count")
    )

    # --------------------------------------------------------
    # Diagnoses
    # --------------------------------------------------------

    diagnosis_patient = (
        data["diagnosis_records"]
        .merge(
            data["admissions"][
                ["admission_id", "patient_id"]
            ],
            on="admission_id",
            how="left"
        )
    )

    diagnosis_summary = (
        diagnosis_patient
        .groupby("patient_id")
        .size()
        .reset_index(name="diagnosis_count")
    )

    # --------------------------------------------------------
    # Laboratory Tests
    # --------------------------------------------------------

    lab_patient = (
        data["lab_results"]
        .merge(
            data["admissions"][
                ["admission_id", "patient_id"]
            ],
            on="admission_id",
            how="left"
        )
    )

    lab_summary = (
        lab_patient
        .groupby("patient_id")
        .size()
        .reset_index(name="lab_test_count")
    )

    # --------------------------------------------------------
    # Medications
    # --------------------------------------------------------

    medication_patient = (
        data["medication_records"]
        .merge(
            data["admissions"][
                ["admission_id", "patient_id"]
            ],
            on="admission_id",
            how="left"
        )
    )

    medication_summary = (
        medication_patient
        .groupby("patient_id")
        .size()
        .reset_index(name="medication_count")
    )

    # --------------------------------------------------------
    # Financial Data
    # --------------------------------------------------------

    claims = data["insurance_claims"].copy()

    claims["billed_amount"] = prepare_numeric(
        claims["billed_amount"]
    )

    claims["approved_amount"] = prepare_numeric(
        claims["approved_amount"]
    )

    claims["denied_amount"] = prepare_numeric(
        claims["denied_amount"]
    )

    financial_summary = (
        claims
        .groupby("patient_id")
        .agg(
            total_billed=("billed_amount", "sum"),
            total_approved=("approved_amount", "sum"),
            total_denied=("denied_amount", "sum")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Payments
    # --------------------------------------------------------

    payments = data["payments"].copy()

    payments["payment_amount"] = prepare_numeric(
        payments["payment_amount"]
    )

    payment_summary = (
        payments
        .groupby("patient_id")
        .agg(
            total_paid=("payment_amount", "sum")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    correlation_data = (
        admission_summary
        .merge(
            procedure_summary,
            on="patient_id",
            how="left"
        )
        .merge(
            diagnosis_summary,
            on="patient_id",
            how="left"
        )
        .merge(
            lab_summary,
            on="patient_id",
            how="left"
        )
        .merge(
            medication_summary,
            on="patient_id",
            how="left"
        )
        .merge(
            financial_summary,
            on="patient_id",
            how="left"
        )
        .merge(
            payment_summary,
            on="patient_id",
            how="left"
        )
    )

    correlation_data = correlation_data.fillna(0)

    return correlation_data

def plot_correlation_heatmap(data, ax=None):

    correlation_data = create_correlation_dataset(data)

    numeric_data = correlation_data.select_dtypes(
        include="number"
    )

    correlation_matrix = numeric_data.corr()

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(14, 10)
        )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Healthcare Variables Correlation Matrix"
    )

    return ax

def get_strongest_correlations(data, min_correlation=0.30):

    correlation_data = create_correlation_dataset(data)

    numeric_data = correlation_data.select_dtypes(
        include="number"
    )

    correlation_matrix = numeric_data.corr()

    pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            variable_1 = columns[i]
            variable_2 = columns[j]

            correlation = correlation_matrix.loc[
                variable_1,
                variable_2
            ]

            if abs(correlation) >= min_correlation:

                pairs.append({
                    "Variable 1": variable_1,
                    "Variable 2": variable_2,
                    "Correlation": correlation
                })

    strongest = pd.DataFrame(pairs)

    if not strongest.empty:

        strongest["Absolute Correlation"] = (
            strongest["Correlation"].abs()
        )

        strongest = (
            strongest
            .sort_values(
                "Absolute Correlation",
                ascending=False
            )
            .drop(
                columns=["Absolute Correlation"]
            )
            .reset_index(drop=True)
        )

    return strongest


# ============================================================
# CORRELATION / REGRESSION PLOT
# ============================================================

def plot_correlation_with_regression(
    data,
    x_col,
    y_col,
    title=None,
    xlabel=None,
    ylabel=None
):
    """
    Create a scatter plot with a linear regression line
    and display the Pearson correlation coefficient.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a copy
    df = data.copy()

    # Keep only the two required columns
    df = df[[x_col, y_col]].copy()

    # Convert to numeric
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # Remove missing values
    df = df.dropna()

    # Check data
    if len(df) < 2:
        print("Not enough data to create the correlation plot.")
        return

    # Calculate correlation
    correlation = df[x_col].corr(df[y_col])

    # Create plot
    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.6},
        line_kws={"linewidth": 2}
    )

    # Title
    if title is None:
        title = f"{x_col} vs {y_col}"

    plt.title(
        f"{title}\nPearson Correlation (r) = {correlation:.2f}",
        fontsize=14
    )

    # Axis labels
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print correlation
    print(f"Pearson correlation (r): {correlation:.4f}")
# ============================================================
# 6. SHOW ALL VISUALIZATIONS
# ============================================================

def show_all_visualizations(data):

    # --------------------------------------------------------
    # Patient Analysis
    # --------------------------------------------------------

    plot_gender_distribution(data)

    plot_age_distribution(data)

    plot_insurance_distribution(data)

    plot_smoking_status(data)

    plot_multiple_admissions(data)

    # --------------------------------------------------------
    # Hospital Operations
    # --------------------------------------------------------

    plot_monthly_admissions(data)

    plot_admissions_by_department(data)

    plot_average_los_by_department(data)

    plot_doctor_workload(data)

    # --------------------------------------------------------
    # Clinical Services
    # --------------------------------------------------------

    plot_top_diagnoses(data)

    plot_top_lab_tests(data)

    plot_abnormal_lab_results(data)

    plot_top_medications(data)

    plot_top_procedures(data)

    # --------------------------------------------------------
    # Insurance and Finance
    # --------------------------------------------------------

    plot_claim_status(data)

    plot_billed_vs_approved(data)

    plot_payment_distribution(data)

    plot_payment_by_method(data)

    plot_financial_activity_by_department(data)

    # --------------------------------------------------------
    # Integrated Analysis
    # --------------------------------------------------------

    plot_los_vs_paid_amount(data)

    plot_procedures_vs_billed_amount(data)

    plot_utilization_vs_los(data)

    plot_utilization_vs_paid_amount(data)

    plot_utilization_vs_billed_amount(data)

    plt.show()