import pandas as pd

# Load data
df = pd.read_csv("../experiment_results.csv")

# Compute average grouped by Dataset and Solver
avg_df = df.groupby(['Dataset', 'Solver'], as_index=False).mean(numeric_only=True)

# Create pivot table with multiple values
pivot = avg_df.pivot(index='Dataset', columns='Solver', values=['Time Taken', 'Initial Reach', 'Final Reach'])

# Save to CSV
pivot.to_csv("pivot_table.csv")

# Optional: print to verify
print("Pivot table saved as 'pivot_table.csv':")
print(pivot.head())
