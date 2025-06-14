#!/usr/bin/env python3
import os
import subprocess

REBUILD_FLAG = False

def run_program_on_dataset(executable, dataset_dir):
    dataset_elems = os.listdir(dataset_dir)
    print("{", flush=True)
    
    os.makedirs("profiling_results", exist_ok=True)

    for entry in dataset_elems:
        if os.path.isdir(os.path.join(dataset_dir, entry)):
            sample_path = os.path.join(dataset_dir, entry, entry + ".mtx")
            if not os.path.isfile(sample_path):
                continue
            
            exec_result = subprocess.run([
                    # "nsys", "profile",
                    # "--show-output", "true",
                    # "--trace", "cuda,nvtx",
                    # "-o", f"profiling_results/{entry}",
                    executable,
                    sample_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True)

            if exec_result.returncode != 0:
                print(f"// Error processing {entry}: exited with code {exec_result.returncode}", flush=True)
                continue

            for line in exec_result.stdout.splitlines():
                print(line, flush=True)

    print("}", flush=True)


if __name__ == "__main__":
    executable = os.path.join("build", "SpVM")
    dataset_dir = "dataset"
    if REBUILD_FLAG:
        subprocess.run(["make", "clean"], check=True)    
        subprocess.run(["make", "cluster"], check=True)

    if not os.path.isdir(dataset_dir):
        exit(1)
    run_program_on_dataset(executable, dataset_dir)

