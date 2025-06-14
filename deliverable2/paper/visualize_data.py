import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

font = { 'weight' : 'bold',
        'size'   : 16}

import matplotlib
matplotlib.rc('font', **font)

with open('spmv_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

melted = df.melt(id_vars=["Dataset", "Type"], 
                 value_vars=["TPR Base (ms)", "TPR Opt. (ms)", "WPR Base (ms)", "WPR Shared (ms)"],
                 var_name="Method", value_name="Time (ms)")

plt.figure(figsize=(14, 6))
sns.barplot(data=melted, x="Dataset", y="Time (ms)", hue="Method")
plt.xticks(rotation=90)
plt.title("SpMV Execution Times per Dataset")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="NNZ", y="WPR Shared (ms)", label="WPR Shared")
sns.scatterplot(data=df, x="NNZ", y="WPR Base (ms)", label="WPR Base", marker="x")
plt.xscale("log")
plt.yscale("log")
plt.title("Execution Time vs. NNZ (log-log scale)")
plt.xlabel("Number of Non-Zeros (NNZ)")
plt.ylabel("Execution Time (ms)")
plt.legend()
plt.tight_layout()
plt.show()

df["Speedup WPR Shared"] = df["WPR Base (ms)"] / df["WPR Shared (ms)"]

# Assign green or red based on speedup
colors = ["green" if s > 1 else "red" for s in df["Speedup WPR Shared"]]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Dataset", y="Speedup WPR Shared", palette=colors)

# Baseline at 1
plt.axhline(1.0, color="black", linewidth=1, linestyle="--")

plt.xticks(rotation=90)
plt.title("Speedup of WPR Shared over WPR Base")
plt.ylabel("Speedup Factor")
plt.xlabel("Dataset")
plt.tight_layout()
plt.show()


# Assuming df is already loaded
df["Speedup TPR"] = df["TPR Base (ms)"] / df["TPR Opt. (ms)"]

# Define bar colors: green if speedup > 1, else red
colors = ["green" if s > 1 else "red" for s in df["Speedup TPR"]]

# Plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x="Dataset", y="Speedup TPR", palette=colors)

# Add a horizontal line at 1 (no speedup)
plt.axhline(1.0, color="black", linewidth=1, linestyle="--")

plt.xticks(rotation=90)
plt.title("Speedup of TPR Read-Only Cache + Loop Unrolling over TPR Base")
plt.ylabel("Speedup Factor")
plt.xlabel("Dataset")
plt.tight_layout()
plt.show()
