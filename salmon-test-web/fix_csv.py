"""
Quick script to fix CSV file with inconsistent trailing commas
"""
import pandas as pd

# Read the CSV
df = pd.read_csv('selected_samples_info.csv')

# Show original state
print("Original columns:", df.columns.tolist())
print("Original shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Clean up
# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

# Remove empty columns
df = df.dropna(axis=1, how='all')

# Clean column names
df.columns = df.columns.str.strip()

# Remove empty rows
df = df.dropna(how='all')

print("\n" + "="*50)
print("After cleaning:")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Save cleaned version
output_file = 'selected_samples_info_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Cleaned CSV saved to: {output_file}")

# Verify Sex column
print("\nSex column info:")
print(f"  Total rows: {len(df)}")
print(f"  Non-null: {df['Sex'].notna().sum()}")
print(f"  Null: {df['Sex'].isna().sum()}")
print(f"  Values: {df['Sex'].value_counts(dropna=False).to_dict()}")
