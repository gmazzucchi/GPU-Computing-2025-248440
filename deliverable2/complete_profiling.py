#!/usr/bin/env python3
import os
import subprocess

REBUILD_FLAG = False

def run_program_on_dataset(executable, dataset_dir):
    dataset_elems = os.listdir(dataset_dir)
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    for entry in dataset_elems:
        if os.path.isdir(os.path.join(dataset_dir, entry)):
            sample_path = os.path.join(dataset_dir, entry, entry + ".mtx")
            with open(os.path.join(log_folder, entry + "_cpu.log"), "w") as cpu_log_file, \
                 open(os.path.join(log_folder, entry + "_gpu.log"), "w") as gpu_log_file:
                if not os.path.isfile(sample_path):
                    continue
                exec_result = subprocess.run([executable, sample_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,    
                                             text=True)
                for line in exec_result.stdout:
                    print(line, end='', flush=True)


if __name__ == "__main__":
    print("{", flush=True)
    executable = os.path.join("build", "SpVM")
    dataset_dir = "dataset"
    if REBUILD_FLAG:
        subprocess.run(["make", "clean"], check=True)    
        subprocess.run(["make", "cluster"], check=True)

    if not os.path.isdir(dataset_dir):
        exit(1)
    run_program_on_dataset(executable, dataset_dir)
    print("}", flush=True)

