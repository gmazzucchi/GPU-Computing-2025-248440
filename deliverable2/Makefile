.PHONY: all cluster pc profiling example clean

all: clean cluster profiling

cluster:
	mkdir -p build
	nvcc -Xcompiler -fopenmp -I/usr/include/eigen3 -Iinclude src/main.cu src/mmio.cpp -o build/SpVM

pc:
	mkdir -p build
	nvcc -arch=sm_35 -Xcompiler -fopenmp -I/usr/include/eigen3 -Iinclude src/main.cu src/mmio.cpp -o build/SpVM

profiling:
	python3 complete_profiling.py > results.json
	python3 plot.py

example:
	./build/SpVM ./dataset_small/ash219/ash219.mtx

clean:
	rm -r build
	mkdir build

