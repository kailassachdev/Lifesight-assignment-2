import pandas as pd

file_path = r"c:\Users\KAILAS\Desktop\assignment 1\Train\Assessment 2 - MMM Weekly.csv"

try:
    df = pd.read_csv(file_path)

    print("--- First 5 rows ---")
    print(df.head())
    print("\n--- Columns ---")
    print(df.columns)
    print("\n--- Data Types and Non-Null Counts ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

