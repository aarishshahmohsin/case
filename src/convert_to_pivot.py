import pandas as pd

df = pd.read_csv("/Users/aarish/case/experiment_results.csv")
avg_df = df.groupby(["Dataset", "Solver"], as_index=False).mean(numeric_only=True)
pivot = avg_df.pivot(
    index="Dataset",
    columns="Solver",
    values=["Time Taken", "Initial Reach", "Final Reach"],
)

pivot.to_csv("pivot_table.csv")
print("Pivot table saved as 'pivot_table.csv':")
print(pivot.head())
