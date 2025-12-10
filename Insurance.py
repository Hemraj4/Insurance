
# Insurance Data Analysis Project

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")

print(df)
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning


df_cleaned = df.copy()

# Remove duplicates

df_cleaned.drop_duplicates(inplace=True)

print(df_cleaned.shape)
print(df_cleaned.isnull().sum())
print(df_cleaned.dtypes)

# Encoding categorical columns


df_cleaned["sex"] = df_cleaned["sex"].map({"male": 0, "female": 1})
df_cleaned["smoker"] = df_cleaned["smoker"].map({"yes": 1, "no": 0})
df_cleaned["region"] = df_cleaned["region"].map({
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
})

print(df_cleaned.head())

# Exploratory Data Analysis (EDA)


# Age Distribution
plt.figure()
sns.histplot(df_cleaned["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# BMI Distribution
plt.figure()
sns.histplot(df_cleaned["bmi"], bins=20, kde=True)
plt.title("BMI Distribution")
plt.show()

# Charges Distribution
plt.figure()
sns.histplot(df_cleaned["charges"], bins=30, kde=True)
plt.title("Medical Charges Distribution")
plt.show()

# Sex vs Charges
plt.figure()
sns.boxplot(x="sex", y="charges", data=df_cleaned)
plt.title("Sex vs Charges")
plt.show()

# Smoker vs Charges
plt.figure()
sns.boxplot(x="smoker", y="charges", data=df_cleaned)
plt.title("Smoker vs Charges")
plt.show()

# Region vs Charges
plt.figure()
sns.boxplot(x="region", y="charges", data=df_cleaned)
plt.title("Region vs Charges")
plt.show()

# Age vs Charges
plt.figure()
sns.scatterplot(x="age", y="charges", data=df_cleaned)
plt.title("Age vs Charges")
plt.show()

# BMI vs Charges
plt.figure()
sns.scatterplot(x="bmi", y="charges", data=df_cleaned)
plt.title("BMI vs Charges")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Final Dataset

print("Final Cleaned Dataset:")
print(df_cleaned)
