import json5
import matplotlib.pyplot as plt
import numpy as np

# --- Load JSON ---
with open("results.json", "r") as f:
    data = json5.load(f)
    
FONTSIZE=16

# --- Extract Dataset Names ---
datasets = []

# --- Initialize Metrics ---
cpu_naive_time = []
cpu_openmp_time = []
thread_time = []
warp_time = []

cpu_naive_throughput = []
cpu_openmp_throughput = []
thread_throughput = []
warp_throughput = []

cpu_naive_gflops = []
cpu_openmp_gflops = []
thread_gflops = []
warp_gflops = []

matrices_to_plot = set(
    ["dataset/pkustk14/pkustk14.mtx",
    "dataset/wiki-Talk/wiki-Talk.mtx",
    "dataset/cage14/cage14.mtx",
    "dataset/rajat31/rajat31.mtx",
    # "dataset/bcsstk13/bcsstk13.mtx",
    # "dataset/trans5/trans5.mtx",
    # "dataset/af23560/af23560.mtx",
    # "dataset/cont11_l/cont11_l.mtx",
    # "dataset/sme3Db/sme3Db.mtx",
    # "dataset/TSOPF_RS_b2383/TSOPF_RS_b2383.mtx",
    "dataset/helm2d03/helm2d03.mtx",
    # "dataset/great-britain_osm/great-britain_osm.mtx",
    # "dataset/webbase-1M/webbase-1M.mtx",
    # "dataset/neos3/neos3.mtx",
    "dataset/ins2/ins2.mtx",
    # "dataset/torso1/torso1.mtx",
    ]
)

# --- Fill Metrics ---
for key in data:
    entry = data[key]
    if key not in matrices_to_plot:
        continue

    datasets.append(key.split("/")[-1])
    
    cpu_naive_time.append(entry["cpu_naive_time"])
    cpu_openmp_time.append(entry["cpu_openmp_time"])
    thread_time.append(entry["thread_per_row_executiontime"])
    warp_time.append(entry["warp_per_row_executiontime"])

    cpu_naive_throughput.append(entry["cpu_naive_throughput"])
    cpu_openmp_throughput.append(entry["cpu_openmp_throughput"])
    thread_throughput.append(entry["thread_per_row_throughput"])
    warp_throughput.append(entry["warp_per_row_throughput"])

    cpu_naive_gflops.append(entry["cpu_naive_gflops"])
    cpu_openmp_gflops.append(entry["cpu_openmp_gflops"])
    thread_gflops.append(entry["thread_per_row_performance"])
    warp_gflops.append(entry["warp_per_row_performance"])

# --- Plotting Function ---
def plot_metric(filename, metric_lists, labels, ylabel, title):
    x = np.arange(len(datasets))
    width = 0.2

    plt.figure(figsize=(12, 6))
    for i, (values, label) in enumerate(zip(metric_lists, labels)):
        plt.bar(x + i * width, values, width, label=label)

    plt.xticks(x + width * 1.5, datasets, rotation=30, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(filename)


font = { 'weight' : 'bold',
        'size'   : 22}

import matplotlib
matplotlib.rc('font', **font)

# --- Plot Time ---
plot_metric(
    "execution_time.png",
    [cpu_naive_time, cpu_openmp_time, thread_time, warp_time],
    ["CPU Naive", "CPU OpenMP", "Thread/Row", "Warp/Row"],
    "Time (ms)",
    "Execution Time per Method"
)

# --- Plot Throughput ---
plot_metric(
    "throughput.png",
    [cpu_naive_throughput, cpu_openmp_throughput, thread_throughput, warp_throughput],
    ["CPU Naive", "CPU OpenMP", "Thread/Row", "Warp/Row"],
    "Throughput (GOp/s)",
    "Throughput per Method"
)

# --- Plot GFLOPS ---
plot_metric(
    "GFLOPS.png",
    [cpu_naive_gflops, cpu_openmp_gflops, thread_gflops, warp_gflops],
    ["CPU Naive", "CPU OpenMP", "Thread/Row", "Warp/Row"],
    "GFLOPS",
    "Performance (GFLOPS) per Method"
)

plt.show()

