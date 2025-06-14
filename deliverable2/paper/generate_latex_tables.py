import pandas as pd
import json

# JSON data (manually loaded as Python dict for this example)
json_data = {
    "dataset/pkustk14/pkustk14.mtx": {"csr_conversion_time": 82.7195, "baseline_executiontime": 4.74224, "baseline_throughput": 38.4401, "baseline_performance": 6.32124, "cusparse_executiontime": 5.06525, "cusparse_throughput": 35.9888, "cusparse_performance": 5.91814, "optimized_executiontime": 0.679008, "optimized_throughput": 268.468, "optimized_performance": 44.148},
    "dataset/wiki-Talk/wiki-Talk.mtx": {"csr_conversion_time": 39.7196, "baseline_executiontime": 4.50435, "baseline_throughput": 21.8826, "baseline_performance": 2.22958, "cusparse_executiontime": 3.92131, "cusparse_throughput": 25.1362, "cusparse_performance": 2.56109, "optimized_executiontime": 0.738432, "optimized_throughput": 133.482, "optimized_performance": 13.6002},
    "dataset/cage14/cage14.mtx": {"csr_conversion_time": 197.404, "baseline_executiontime": 8.17795, "baseline_throughput": 42.756, "baseline_performance": 6.635, "cusparse_executiontime": 5.57011, "cusparse_throughput": 62.7737, "cusparse_performance": 9.7414, "optimized_executiontime": 0.627744, "optimized_throughput": 557.005, "optimized_performance": 86.4376},
    "dataset/rajat31/rajat31.mtx": {"csr_conversion_time": 188.754, "baseline_executiontime": 4.54435, "baseline_throughput": 70.1607, "baseline_performance": 8.94132, "cusparse_executiontime": 6.54128, "cusparse_throughput": 48.742, "cusparse_performance": 6.21171, "optimized_executiontime": 1.79978, "optimized_throughput": 177.153, "optimized_performance": 22.5764},
    "dataset/bcsstk13/bcsstk13.mtx": {"csr_conversion_time": 0.77802, "baseline_executiontime": 10.1448, "baseline_throughput": 0.104752, "baseline_performance": 0.0169321, "cusparse_executiontime": 4.7889, "cusparse_throughput": 0.221905, "cusparse_performance": 0.0358688, "optimized_executiontime": 0.026048, "optimized_throughput": 40.797, "optimized_performance": 6.59444},
    "dataset/af23560/af23560.mtx": {"csr_conversion_time": 4.27698, "baseline_executiontime": 2.9969, "baseline_throughput": 2.06481, "baseline_performance": 0.323172, "cusparse_executiontime": 4.64845, "cusparse_throughput": 1.3312, "cusparse_performance": 0.208352, "optimized_executiontime": 0.033568, "optimized_throughput": 184.343, "optimized_performance": 28.8522},
    "dataset/neos3/neos3.mtx": {"csr_conversion_time": 23.5805, "baseline_executiontime": 3.33178, "baseline_throughput": 9.8772, "baseline_performance": 1.23359, "cusparse_executiontime": 5.36742, "cusparse_throughput": 6.13118, "cusparse_performance": 0.765739, "optimized_executiontime": 0.2184, "optimized_throughput": 150.68, "optimized_performance": 18.8189},
    "dataset/ins2/ins2.mtx": {"csr_conversion_time": 27.5924, "baseline_executiontime": 4.11216, "baseline_throughput": 10.1361, "baseline_performance": 1.4887, "cusparse_executiontime": 4.92906, "cusparse_throughput": 8.45625, "cusparse_performance": 1.24198, "optimized_executiontime": 0.699968, "optimized_throughput": 59.5475, "optimized_performance": 8.74582},
    "dataset/helm2d03/helm2d03.mtx": {"csr_conversion_time": 33.4673, "baseline_executiontime": 5.86486, "baseline_throughput": 7.48294, "baseline_performance": 1.0688, "cusparse_executiontime": 5.36595, "cusparse_throughput": 8.17868, "cusparse_performance": 1.16818, "optimized_executiontime": 0.691744, "optimized_throughput": 63.4431, "optimized_performance": 9.06171},
    "dataset/great-britain_osm/great-britain_osm.mtx": {"csr_conversion_time": 163.996, "baseline_executiontime": 5.67962, "baseline_throughput": 56.2534, "baseline_performance": 5.74441, "cusparse_executiontime": 6.69242, "cusparse_throughput": 47.7402, "cusparse_performance": 4.87508, "optimized_executiontime": 2.92486, "optimized_throughput": 109.235, "optimized_performance": 11.1547}
}

# Base table structure
columns = ["name", "rows", "cols", "nnz", "sym", "type", "csr_time", "base_perf", "cusparse_perf", "opt_perf"]
latex_rows = []

# Manual metadata table
base_table = [
    ("pkustk14", 151926, 151926, 7494215, r"\cmark", "structural"),
    ("wiki-Talk", 2394385, 2394385, 5021410, r"\xmark", "graph"),
    ("cage14", 1505785, 1505785, 27130349, r"\xmark", "graph"),
    ("rajat31", 4690002, 4690002, 20316253, r"\xmark", "circuit"),
    ("bcsstk13", 2003, 2003, 42943, r"\cmark", "fluid"),
    ("af23560", 23560, 23560, 484256, r"\xmark", "fluid"),
    ("neos3", 512209, 518832, 2055024, r"\xmark", "optimization"),
    ("ins2", 309412, 309412, 1530448, r"\cmark", "optimization"),
    ("helm2d03", 392257, 392257, 1567096, r"\cmark", "2D/3D PDE"),
    ("great-britain_osm", 7733822, 7733822, 8156517, r"\cmark", "graph")
]

# Assemble LaTeX rows
for name, rows, cols, nnz, sym, mtype in base_table:
    key = f"dataset/{name}/{name}.mtx"
    entry = json_data[key]
    latex_rows.append(
        f"{name} & {rows} & {cols} & {nnz} & {sym} & {mtype} & {entry['csr_conversion_time']:.2f} & {entry['baseline_executiontime']:.2f} & {entry['cusparse_executiontime']:.2f} & {entry['optimized_executiontime']:.2f} \\\\"
    )


print('\n'.join(latex_rows))