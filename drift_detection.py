
import pandas as pd

from sklearn.datasets import load_iris

from evidently import Report
from evidently.presets import DataDriftPreset

# Load dataset
iris = load_iris(as_frame=True)

df = iris.frame

# Remove target column
df = df.drop(columns=["target"])

# Reference data
reference_data = df.iloc[:100].copy()

# Current data
current_data = df.iloc[100:].copy()

# Simulate data drift
current_data["sepal length (cm)"] += 1.5

# Create report
report = Report([
    DataDriftPreset()
])

# Run drift analysis
result = report.run(
    current_data,
    reference_data
)

# Save report
result.save_html("drift_report.html")

print("Data drift analysis completed!")
print("Report saved successfully!")
