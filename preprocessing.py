import pandas as pd

# Load dataset
file_path = "//content//CFS_in_the_NHS 12_month_follow_up DATA_anonymised(1).csv"
df = pd.read_csv(file_path)

# Display dataset information
print(df.info())
print(df.head())
# Fill missing values:
# - Numerical features → median
# - Categorical features → mode

for col in df.columns:
    if df[col].dtype == 'object':  # Categorical
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Numerical
        df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Missing values handled.")
# Identify categorical features
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\n✅ Categorical features encoded.")
# prompt: from sklearn.preprocessing import StandardScaler
# # Standardize numerical features
# scaler = StandardScaler()
# df[df.columns] = scaler.fit_transform(df[df.columns])
# print("\n✅ Features standardized.")

from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print("\n✅ Features standardized.")
processed_file_path = "preprocessed_CFS_dataset.csv"  # Save in the current directory
df.to_csv(processed_file_path, index=False)

print(f"\n✅ Preprocessed dataset saved as: {processed_file_path}")
