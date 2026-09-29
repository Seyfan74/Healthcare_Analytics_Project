import pandas as pd


# =========================================================
# DATA CLEANING FUNCTION
# =========================================================

def clean_all_data(data):
    """
    Clean all healthcare datasets.
    """

    cleaned_data = {}

    # Loop through every healthcare table
    for table_name, df in data.items():

        # Make a copy of the DataFrame
        df = df.copy()

        # -------------------------------------------------
        # 1. Remove duplicate rows
        # -------------------------------------------------

        df = df.drop_duplicates()

        # -------------------------------------------------
        # 2. Standardize column names
        # -------------------------------------------------

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        # -------------------------------------------------
        # 3. Remove extra spaces from text columns
        # -------------------------------------------------

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in text_columns:
            df[column] = df[column].str.strip()

        # -------------------------------------------------
        # 4. Replace empty strings with missing values
        # -------------------------------------------------

        df = df.replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

        # -------------------------------------------------
        # 5. Save cleaned DataFrame
        # -------------------------------------------------

        cleaned_data[table_name] = df

    # Return all cleaned datasets
    return cleaned_data