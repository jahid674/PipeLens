import pandas as pd

# ===== Path to your dataset =====
file_path = "BugDoc-master/examples/0.1_filtered_adult_sp.csv"   # change if different extension

# ---- Load dataset ----
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# ---- Keep first 13 columns only ----
df = df.iloc[:, :14]

print("New shape:", df.shape)

# ---- Save BACK to the same path (overwrite) ----
df.to_csv(file_path, index=False)

print("Done. Dataset updated with only first 13 columns.")
