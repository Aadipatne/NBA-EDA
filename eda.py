import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load the dataset
# -----------------------------

data = pd.read_csv(
    "2021-2022 NBA Player Stats - Regular.csv",
    sep=";",
    encoding="latin1"
)

# -----------------------------
# Basic EDA
# -----------------------------

print("First 5 Rows:")
print(data.head())

print("\nRows and Columns:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSummary Statistics:")
print(data.describe())

print("\nPositions:")
print(data["Pos"].value_counts())

print("\nTeams:")
print(data["Tm"].value_counts())

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove duplicate rows
data = data.drop_duplicates()

# Remove extra spaces
data["Player"] = data["Player"].str.strip()
data["Pos"] = data["Pos"].str.strip()
data["Tm"] = data["Tm"].str.strip()

# Convert columns to numbers
columns = [
    "Age", "G", "GS", "MP", "PTS",
    "TRB", "AST", "STL", "BLK",
    "FG%", "3P%", "FT%"
]

for col in columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Fill missing shooting percentages
data["FG%"] = data["FG%"].fillna(0)
data["3P%"] = data["3P%"].fillna(0)
data["FT%"] = data["FT%"].fillna(0)

print("\nCleaned Data:")
print(data.head())

# -----------------------------
# Feature Engineering
# -----------------------------

data["PointsPerMinute"] = data["PTS"] / data["MP"]
data["AssistsPerGame"] = data["AST"] / data["G"]
data["ReboundsPerGame"] = data["TRB"] / data["G"]

print("\nNew Features:")
print(
    data[
        [
            "Player",
            "PointsPerMinute",
            "AssistsPerGame",
            "ReboundsPerGame"
        ]
    ].head()
)

# -----------------------------
# Histogram - Points
# -----------------------------

plt.figure(figsize=(8,5))
plt.hist(data["PTS"], bins=20)
plt.title("Distribution of Points")
plt.xlabel("Points")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Histogram - Age
# -----------------------------

plt.figure(figsize=(8,5))
plt.hist(data["Age"], bins=15)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Scatter Plot
# -----------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x="AST", y="PTS")
plt.title("Assists vs Points")
plt.show()

# -----------------------------
# Scatter Plot
# -----------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x="MP", y="PTS")
plt.title("Minutes Played vs Points")
plt.show()

# -----------------------------
# Players by Position
# -----------------------------

plt.figure(figsize=(8,5))
data["Pos"].value_counts().plot(kind="bar")
plt.title("Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.show()

# -----------------------------
# Top 10 Scorers
# -----------------------------

top10 = data.sort_values(by="PTS", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top10, x="PTS", y="Player")
plt.title("Top 10 Players by Points")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Conclusions
# -----------------------------

print("\n----- Conclusions -----")

print(
    "Most Common Position:",
    data["Pos"].value_counts().idxmax()
)

print(
    "Team with Most Players:",
    data["Tm"].value_counts().idxmax()
)

print(
    "Highest Scorer:",
    data.loc[data["PTS"].idxmax(), "Player"]
)

print(
    "Highest Points Per Minute:",
    data.loc[data["PointsPerMinute"].idxmax(), "Player"]
)

print(
    "The dataset was cleaned by removing duplicate rows, trimming text columns, converting numeric columns, and filling missing shooting percentages."
)