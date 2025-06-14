.PHONY: all cluster pc profiling example clean

all: clean cluster profiling

cluster:
	mkdir -p build
	nvcc --gpu-architecture=sm_70 -lcusparse -Xcompiler -fopenmp -I/usr/include/eigen3 -Iinclude src/main.cu src/mmio.cpp -o build/SpVM

pc:
	mkdir -p build
	nvcc -lcusparse -G -arch=sm_35 -Xcompiler -fopenmp -I/usr/include/eigen3 -Iinclude src/main.cu src/mmio.cpp -o build/SpVM

profiling:
	python3 complete_profiling.py > results.json
	python3 plot.py

example:
	build/SpVM dataset/bcsstk13/bcsstk13.mtx
#build/SpVM dataset/cage14/cage14.mtx
# nsys profile --quiet --trace='cuda,nvtx' -o example_profiling build/SpVM dataset/cage14/cage14.mtx

clean:
	rm -r build
	mkdir build

