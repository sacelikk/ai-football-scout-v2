import pandas as pd

df1 = pd.read_csv("players_data-2024_2025.csv")
df2 = pd.read_csv("players_data-2025_2026.csv")

print("24-25 columns:", len(df1.columns))
print("25-26 columns:", len(df2.columns))

common = set(df1.columns).intersection(set(df2.columns))
print("\nCommon columns:", len(common))
print(sorted(list(common)))
