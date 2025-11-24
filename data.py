import pandas as pd

# Load CSV file
df = pd.read_csv("IMFs_ydb.csv")

# Drop last 9 columns
# Use iloc to select all columns except the last 9
df_reduced = df.iloc[:, :-9]

# Save result
df_reduced.to_csv("IFs_ydb.csv", index=False)

print("Finished removing the last 9 columns.")
