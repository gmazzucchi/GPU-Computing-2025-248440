import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json5
import re

data = json5.load(open("results.json", "r"))

matrices_to_plot = {
    "dataset/pkustk14/pkustk14.mtx",
    # "dataset/wiki-Talk/wiki-Talk.mtx",
    "dataset/cage14/cage14.mtx",
    "dataset/rajat31/rajat31.mtx",
    "dataset/helm2d03/helm2d03.mtx",
    "dataset/ins2/ins2.mtx",
    "dataset/bcsstk13/bcsstk13.mtx",
    "dataset/trans5/trans5.mtx",
    "dataset/af23560/af23560.mtx",
    "dataset/cont11_l/cont11_l.mtx",
    "dataset/sme3Db/sme3Db.mtx",
    # "dataset/TSOPF_RS_b2383/TSOPF_RS_b2383.mtx",
    "dataset/great-britain_osm/great-britain_osm.mtx",
    "dataset/webbase-1M/webbase-1M.mtx",
    "dataset/neos3/neos3.mtx",
    "dataset/torso1/torso1.mtx"
}


datasets = []
metrics = {
    'Execution Time (ms)': [
        'thread_per_row_baseline_executiontime',
        'thread_per_row_optimized_executiontime',
        'warp_per_row_baseline_executiontime',
        'warp_per_row_shared_memory_executiontime',
        'cusparse_executiontime',
    ],
    'Throughput (GOp/s)': [
        'thread_per_row_baseline_throughput',
        'thread_per_row_optimized_throughput',
        'warp_per_row_baseline_throughput',
        'warp_per_row_shared_memory_throughput',
        'cusparse_throughput',
    ],
    'Performance (GFLOPS)': [
        'thread_per_row_baseline_performance',
        'thread_per_row_optimized_performance',
        'warp_per_row_baseline_performance',
        'warp_per_row_shared_memory_performance',
        'cusparse_performance',
    ]
}

labels = [
    "Thread-per-row (baseline)",
    "Thread-per-row (opt)",
    "Warp-per-row (baseline)",
    "Warp-per-row (shared memory)",
    "cuSPARSE"
]

plot_data = {metric: [] for metric in metrics}
for key, entry in data.items():
    if key not in matrices_to_plot:
        continue
    dataset = key.split("/")[-1]
    datasets.append(dataset)
    for category, keys in metrics.items():
        plot_data[category].append([entry.get(k, 0.0) for k in keys])

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

def plot_grouped_bar(metric_name, filename, logscale=True):
    data_matrix = np.array(plot_data[metric_name])
    num_datasets, num_methods = data_matrix.shape

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.15
    x = np.arange(num_datasets)

    for i in range(num_methods):
        ax.bar(x + i * width, data_matrix[:, i], width, label=labels[i])

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} on torso1")
    ax.set_xticks(x + width * (num_methods - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right")

    if logscale:
        ax.set_yscale("log")

    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

plot_grouped_bar("Execution Time (ms)", "paper/execution_time.png", logscale=True)
plot_grouped_bar("Throughput (GOp/s)", "paper/throughput.png")
plot_grouped_bar("Performance (GFLOPS)", "paper/performance.png")
plt.show()
