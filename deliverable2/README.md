# Instructions

Download the tarballs of the sparse matrix and unzip them into the `dataset` folder, so you have a file structure like this:

```dataset/matrixname/matrixname.mtx```

Then `make cluster` for building, and `sbatch COMPLETE_PROFILING_ON_CLUSTER.sh` for running the program for all matrices.
This saves the results into `results.json`. With the script `better_plots.py` the plots are generated.

In `profiling_results` there are all the nsys profiling files.

