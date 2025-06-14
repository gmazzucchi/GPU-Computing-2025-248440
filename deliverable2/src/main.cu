#include "cxxopts.hpp"
#include "mmio.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>

#define EIGEN_CHECK_ENABLED
// #define LOG_RESULTS_ENABLED

#define CPU_COMPUTATION_ENABLED
#define OPENMP_ENABLED
#define GPU_THREAD_PER_ROW_ENABLED
#define GPU_WARP_PER_ROW_ENABLED

#if defined(EIGEN_CHECK_ENABLED)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

struct COOEntry {
    int row;
    int col;
    double val;
};

#define CHECK_EIGEN(result_vector, test_name)                                                          \
    if (!check_with_eigen_result(result_vector, NZ_ROWS_COUNT)) {                                      \
        std::cout << test_name ": Eigen check failed at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1);                                                                                       \
    }

#if defined(GPU_THREAD_PER_ROW_ENABLED)
__global__ void spmv_thread_per_row(
    const int *row_ptr,
    const int *col_idx,
    const double *values,
    const double *x,
    double *y,
    int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double dot = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            dot += values[j] * x[col_idx[j]];
        }
        y[row] = dot;
    }
}
#endif  // defined(GPU_THREAD_PER_ROW_ENABLED)

#if defined(GPU_WARP_PER_ROW_ENABLED)
__global__ void spmv_warp_per_row(const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, int num_rows) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id < num_rows) {
        int row_start = row_ptr[warp_id];
        int row_end   = row_ptr[warp_id + 1];
        double sum    = 0.0;
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += values[j] * x[col_idx[j]];
        }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0)
            y[warp_id] = sum;
    }
}
#endif  // defined(GPU_WARP_PER_ROW_ENABLED)

#if defined(EIGEN_CHECK_ENABLED)
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
Eigen::VectorXd y;

void compute_eigen_result(int *csr_rows_ptr, int *csr_cols_ptr, double *csr_values_ptr, int M, int N, double *random_vector, const int nz) {
    std::vector<T> rows_cols_and_nz;
    rows_cols_and_nz.reserve(nz);
    for (int row = 0; row < M; ++row) {
        for (int idx = csr_rows_ptr[row]; idx < csr_rows_ptr[row + 1]; ++idx) {
            rows_cols_and_nz.emplace_back(row, csr_cols_ptr[idx], csr_values_ptr[idx]);
        }
    }
    SpMat mat(M, N);
    mat.setFromTriplets(rows_cols_and_nz.begin(), rows_cols_and_nz.end());
    Eigen::VectorXd x(N);
    for (int i = 0; i < N; ++i) {
        x(i) = random_vector[i];
    }
    y = mat * x;
}

bool check_with_eigen_result(double *result, const int M) {
    for (int i = 0; i < M; ++i) {
        if (std::abs(result[i] - y(i)) > 1e-2) {
            std::cout << "Mismatch at row " << i << ": our result = " << result[i] << ", Eigen result = " << y(i) << std::endl;
            return false;
        }
    }
    return true;
}
#endif  // defined(EIGEN_CHECK_ENABLED)

#if defined(CPU_COMPUTATION_ENABLED)
void spvm_cpu(int *csr_rows_ptr, int *csr_cols_ptr, double *csr_values_ptr, const int nrows, double *random_vector, double *result) {
    for (int row = 0; row < nrows; ++row) {
        double sum = 0.0;
        for (int idx = csr_rows_ptr[row]; idx < csr_rows_ptr[row + 1]; ++idx) {
            sum += csr_values_ptr[idx] * random_vector[csr_cols_ptr[idx]];
        }
        result[row] = sum;
    }
}
#endif

#if defined(OPENMP_ENABLED)
#include <omp.h>

void spvm_cpu_openMP(int *csr_rows_ptr, int *csr_cols_ptr, double *csr_values_ptr, const int nrows, double *random_vector, double *result) {
#pragma omp parallel for
    for (int row = 0; row < nrows; ++row) {
        double sum    = 0.0;
        int row_start = csr_rows_ptr[row];
        int row_end   = csr_rows_ptr[row + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            sum += csr_values_ptr[idx] * random_vector[csr_cols_ptr[idx]];
        }

        result[row] = sum;
    }
}
#endif  // defined(OPENMP_ENABLED)

void profile_kernel(
    const std::string &name,
    const size_t nzeros,
    const size_t num_rows,
    const size_t num_cols,
    const std::function<void()> &kernel_launcher) {
    cudaEvent_t start, stop;
    float ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_launcher();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "\"" << name << "_executiontime\": " << ms << "," << std::endl;
    size_t bytes     = nzeros * (sizeof(int) + sizeof(double)) + num_cols * sizeof(double) + num_rows * sizeof(double);
    float throughput = (float)bytes / (ms * 1e6f);
    std::cout << "\"" << name << "_throughput\": " << throughput << "," << std::endl;

    float time_s = ms / 1000.0f;
    float gflops = (2.0f * nzeros) / (time_s * 1e9f);
    std::cout << "\"" << name << "_performance\": " << gflops << "," << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("SpVM", "Sparse Matrix Vector Multiplication");
    options.add_options()("p,path", "Matrix market input file path", cxxopts::value<std::string>());
    options.parse_positional({"path"});
    auto args_result = options.parse(argc, argv);

    if (args_result.count("path") == 0) {
        std::cout << "Please provide the path to the matrix market file." << std::endl;
        std::cout << "Usage: " << argv[0] << " --path <path>" << std::endl;
        return 1;
    }
    std::string path = args_result["path"].as<std::string>();
    std::cout << "\"" << path << "\"" << ": {" << std::endl;

    MM_typecode matcode;
    FILE *f;
    int NZ_ROWS_COUNT, NZ_COLS_COUNT, nz_elements_in_MM_file, ret_code;

    if ((f = fopen(path.c_str(), "r")) == NULL) {
        std::cout << "Could not open Matrix Market file [" << path << "]" << std::endl;
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
        std::cout << "Could not process Matrix Market banner." << std::endl;
        exit(1);
    }
    if (mm_is_complex(matcode)) {
        std::cout << "Complex matrices are not supported." << std::endl;
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &NZ_ROWS_COUNT, &NZ_COLS_COUNT, &nz_elements_in_MM_file)) != 0) {
        exit(1);
    }

    size_t NZ_COUNT = nz_elements_in_MM_file;
    if (mm_is_symmetric(matcode)) {
        NZ_COUNT = nz_elements_in_MM_file * 2;
        if (NZ_ROWS_COUNT != NZ_COLS_COUNT) {
            std::cout << "Matrix is symmetric but not square." << std::endl;
            exit(1);
        }
    }

    int *coo_nz_rows_ptr           = (int *)malloc(NZ_COUNT * sizeof(int));
    int *coo_nz_cols_ptr           = (int *)malloc(NZ_COUNT * sizeof(int));
    double *coo_nz_values_ptr      = (double *)malloc(NZ_COUNT * sizeof(double));
    double *unused_imag_values_ptr = (double *)malloc(NZ_COUNT * sizeof(double));

    memset(coo_nz_rows_ptr, 0, NZ_COUNT * sizeof(int));
    memset(coo_nz_cols_ptr, 0, NZ_COUNT * sizeof(int));
    memset(coo_nz_values_ptr, 0, NZ_COUNT * sizeof(double));
    memset(unused_imag_values_ptr, 0, NZ_COUNT * sizeof(double));

    for (size_t inz = 0; inz < nz_elements_in_MM_file; inz++) {
#warning Save the imag value in a tmp unused variable
        ret_code = mm_read_mtx_crd_entry(
            f, &coo_nz_rows_ptr[inz], &coo_nz_cols_ptr[inz], &coo_nz_values_ptr[inz], &unused_imag_values_ptr[inz], matcode);
        coo_nz_rows_ptr[inz]--;
        coo_nz_cols_ptr[inz]--;
        if (ret_code != 0) {
            std::cout << "Could not read matrix data at line " << inz << std::endl;
            exit(1);
        }
        if (mm_is_symmetric(matcode) && coo_nz_rows_ptr[inz] != coo_nz_cols_ptr[inz]) {
            coo_nz_rows_ptr[inz + nz_elements_in_MM_file]        = coo_nz_cols_ptr[inz];
            coo_nz_cols_ptr[inz + nz_elements_in_MM_file]        = coo_nz_rows_ptr[inz];
            coo_nz_values_ptr[inz + nz_elements_in_MM_file]      = coo_nz_values_ptr[inz];
            unused_imag_values_ptr[inz + nz_elements_in_MM_file] = unused_imag_values_ptr[inz];
        }
    }
    fclose(f);

    if (mm_is_pattern(matcode)) {
        for (size_t inz = 0; inz < NZ_COUNT; inz++) {
            coo_nz_values_ptr[inz] = 1.0;
        }
    }

    auto start_sorting_time = std::chrono::high_resolution_clock::now();

    std::vector<COOEntry> coo_entries(NZ_COUNT);
    for (size_t i = 0; i < NZ_COUNT; ++i) {
        coo_entries[i].row = coo_nz_rows_ptr[i];
        coo_entries[i].col = coo_nz_cols_ptr[i];
        coo_entries[i].val = coo_nz_values_ptr[i];
    }

    std::sort(coo_entries.begin(), coo_entries.end(), [](const COOEntry &a, const COOEntry &b) {
        return (a.row < b.row) || (a.row == b.row && a.col < b.col);
    });

    for (size_t i = 0; i < NZ_COUNT; ++i) {
        coo_nz_rows_ptr[i]   = coo_entries[i].row;
        coo_nz_cols_ptr[i]   = coo_entries[i].col;
        coo_nz_values_ptr[i] = coo_entries[i].val;
    }
    auto stop_sorting_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> sorting_time = stop_sorting_time - start_sorting_time;
    // std::cout << "Sorting time: " << sorting_time.count() << " ms" << std::endl;

    // std::cout << "Matrix type: " << mm_typecode_to_str(matcode) << std::endl;
    // std::cout << "Matrix size: " << NZ_ROWS_COUNT << " x " << NZ_COLS_COUNT << std::endl;
    // std::cout << "Number of non-zeros: " << NZ_COUNT << std::endl;

    // CSR Conversion: time paid by all implementations
    auto CSR_time_start         = std::chrono::high_resolution_clock::now();
    const size_t CSR_ROWS_COUNT = NZ_ROWS_COUNT + 1;

    int *csr_rows_ptr      = (int *)malloc((CSR_ROWS_COUNT) * sizeof(int));
    int *csr_cols_ptr      = (int *)malloc(NZ_COUNT * sizeof(int));
    double *csr_values_ptr = (double *)malloc(NZ_COUNT * sizeof(double));

    for (int i = 0; i < CSR_ROWS_COUNT; ++i) {
        csr_rows_ptr[i] = 0;
    }

    for (int i = 0; i < CSR_ROWS_COUNT; ++i) {
        csr_rows_ptr[i] = 0;
    }
    for (size_t i = 0; i < NZ_COUNT; ++i) {
        csr_rows_ptr[coo_nz_rows_ptr[i]]++;
    }
    for (int i = 0, cumsum = 0; i <= NZ_ROWS_COUNT; ++i) {
        int temp        = csr_rows_ptr[i];
        csr_rows_ptr[i] = cumsum;
        cumsum += temp;
    }
    int *row_positions = (int *)malloc((CSR_ROWS_COUNT) * sizeof(int));
    memcpy(row_positions, csr_rows_ptr, (CSR_ROWS_COUNT) * sizeof(int));

    for (size_t i = 0; i < NZ_COUNT; ++i) {
        int row              = coo_nz_rows_ptr[i];
        int dest             = row_positions[row];
        csr_cols_ptr[dest]   = coo_nz_cols_ptr[i];
        csr_values_ptr[dest] = coo_nz_values_ptr[i];
        row_positions[row]++;
    }

    free(row_positions);

    auto CSR_time_stop                                            = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> csr_conversion_time = CSR_time_stop - CSR_time_start;
    std::cout << "\"" << "csr_conversion_time\": " << csr_conversion_time.count() << "," << std::endl;

    double *random_vector = (double *)malloc(NZ_COLS_COUNT * sizeof(double));
    double *cpu_result    = (double *)malloc(NZ_ROWS_COUNT * sizeof(double));
    double *gpu_result    = (double *)malloc(NZ_ROWS_COUNT * sizeof(double));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < NZ_COLS_COUNT; ++i) {
        random_vector[i] = dis(gen);
    }
#if defined(EIGEN_CHECK_ENABLED)
    compute_eigen_result(csr_rows_ptr, csr_cols_ptr, csr_values_ptr, NZ_ROWS_COUNT, NZ_COLS_COUNT, random_vector, NZ_COUNT);
#endif

#if defined(CPU_COMPUTATION_ENABLED)
    auto t0 = std::chrono::high_resolution_clock::now();
    spvm_cpu(csr_rows_ptr, csr_cols_ptr, csr_values_ptr, NZ_ROWS_COUNT, random_vector, cpu_result);
    auto t1                                                = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = t1 - t0;
    std::cout << "\"" << "cpu_naive_time\": " << cpu_duration.count() << "," << std::endl;
    {
        float ms         = cpu_duration.count();
        size_t bytes     = NZ_COUNT * (sizeof(int) + sizeof(double)) + NZ_COLS_COUNT * sizeof(double) + NZ_ROWS_COUNT * sizeof(double);
        float throughput = (float)bytes / (ms * 1e6f);  // GB/s
        std::cout << "\"" << "cpu_naive_throughput\": " << throughput << "," << std::endl;

        float time_s = ms / 1000.0f;
        float gflops = (2.0f * NZ_COUNT) / (time_s * 1e9f);
        std::cout << "\"" << "cpu_naive_gflops\": " << gflops << "," << std::endl;
    }
#if defined(EIGEN_CHECK_ENABLED)
    CHECK_EIGEN(cpu_result, "CPU COMPUTATION");
#endif  // defined(EIGEN_CHECK_ENABLED)
#if defined(LOG_RESULTS_ENABLED)
    std::cout << "CPU result:\n";
    for (int i = 0; i < NZ_ROWS_COUNT; ++i) {
        std::cout << cpu_result[i] << " ";
    }
    std::cout << std::endl;
#endif  // defined(LOG_RESULTS_ENABLED)
#endif  // defined(CPU_COMPUTATION_ENABLED)

#if defined(OPENMP_ENABLED)
    auto openmpstart = std::chrono::high_resolution_clock::now();
    spvm_cpu_openMP(csr_rows_ptr, csr_cols_ptr, csr_values_ptr, NZ_ROWS_COUNT, random_vector, cpu_result);
    auto openmpstop                                          = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> openmpduration = openmpstop - openmpstart;
    std::cout << "\"" << "cpu_openmp_time\": " << openmpduration.count() << "," << std::endl;
    {
        float ms         = openmpduration.count();
        size_t bytes     = NZ_COUNT * (sizeof(int) + sizeof(double)) + NZ_COLS_COUNT * sizeof(double) + NZ_ROWS_COUNT * sizeof(double);
        float throughput = (float)bytes / (ms * 1e6f);  // GB/s
        std::cout << "\"" << "cpu_openmp_throughput\": " << throughput << "," << std::endl;

        float time_s = ms / 1000.0f;
        float gflops = (2.0f * NZ_COUNT) / (time_s * 1e9f);
        std::cout << "\"" << "cpu_openmp_gflops\": " << gflops << "," << std::endl;
    }
#if defined(EIGEN_CHECK_ENABLED)
    CHECK_EIGEN(cpu_result, "CPU COMPUTATION");
#endif  // defined(EIGEN_CHECK_ENABLED)
#if defined(LOG_RESULTS_ENABLED)
    std::cout << "CPU result:\n";
    for (int i = 0; i < NZ_ROWS_COUNT; ++i) {
        std::cout << cpu_result[i] << " ";
    }
    std::cout << std::endl;
#endif  // defined(LOG_RESULTS_ENABLED)
#endif  // defined(CPU_COMPUTATION_ENABLED)

    int *coo_rows_ptr_on_gpu, *csr_rows_ptr_on_gpu, *csr_cols_ptr_on_gpu;
    double *nz_values_ptr_on_gpu, *random_vector_ptr_on_gpu, *result_vector_ptr_on_gpu;

    cudaMalloc(&coo_rows_ptr_on_gpu, sizeof(int) * NZ_COUNT);
    cudaMalloc(&csr_rows_ptr_on_gpu, sizeof(int) * CSR_ROWS_COUNT);
    cudaMalloc(&csr_cols_ptr_on_gpu, sizeof(int) * NZ_COUNT);
    cudaMalloc(&nz_values_ptr_on_gpu, sizeof(double) * NZ_COUNT);
    cudaMalloc(&random_vector_ptr_on_gpu, sizeof(double) * NZ_COUNT);
    cudaMalloc(&result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT);
    cudaMemcpy(coo_rows_ptr_on_gpu, coo_nz_rows_ptr, sizeof(int) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(csr_rows_ptr_on_gpu, csr_rows_ptr, sizeof(int) * CSR_ROWS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(csr_cols_ptr_on_gpu, csr_cols_ptr, sizeof(int) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(nz_values_ptr_on_gpu, csr_values_ptr, sizeof(double) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(random_vector_ptr_on_gpu, random_vector, sizeof(double) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);

    int blockSize = 256;
#if defined(GPU_THREAD_PER_ROW_ENABLED)
    blockSize    = 128;
    int gridSize = (NZ_ROWS_COUNT + blockSize - 1) / blockSize;
    profile_kernel("thread_per_row", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
        spmv_thread_per_row<<<gridSize, blockSize>>>(
            csr_rows_ptr_on_gpu,
            csr_cols_ptr_on_gpu,
            nz_values_ptr_on_gpu,
            random_vector_ptr_on_gpu,
            result_vector_ptr_on_gpu,
            NZ_ROWS_COUNT);
    });
    cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);
#if defined(LOG_RESULTS_ENABLED)
    std::cout << "\nThread-per-row result:\n";
    for (int i = 0; i < NZ_ROWS_COUNT; ++i)
        std::cout << gpu_result[i] << " ";
    std::cout << std::endl;
#endif  // defined(LOG_RESULTS_ENABLED)
#if defined(EIGEN_CHECK_ENABLED)
    CHECK_EIGEN(gpu_result, "thread_per_row");
#endif  // defined(EIGEN_CHECK_ENABLED)
#endif  //

#if defined(GPU_WARP_PER_ROW_ENABLED)
    cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);
    blockSize         = 128;
    int warpsPerBlock = blockSize / 32;
    gridSize          = (NZ_ROWS_COUNT + warpsPerBlock - 1) / warpsPerBlock;

    profile_kernel("warp_per_row", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
        spmv_warp_per_row<<<gridSize, blockSize>>>(
            csr_rows_ptr_on_gpu,
            csr_cols_ptr_on_gpu,
            nz_values_ptr_on_gpu,
            random_vector_ptr_on_gpu,
            result_vector_ptr_on_gpu,
            NZ_ROWS_COUNT);
    });
    cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);
#if defined(LOG_RESULTS_ENABLED)
    std::cout << "\nWarp-per-row result:\n";
    for (int i = 0; i < NZ_ROWS_COUNT; ++i)
        std::cout << gpu_result[i] << " ";
    std::cout << std::endl;
#endif  // defined(LOG_RESULTS_ENABLED)
#if defined(EIGEN_CHECK_ENABLED)
    CHECK_EIGEN(gpu_result, "warp_per_row");
#endif  // defined(EIGEN_CHECK_ENABLED)
#endif  // defined(GPU_WARP_PER_ROW_ENABLED)
    std::cout << "}," << std::endl;

    cudaFree(csr_rows_ptr_on_gpu);
    cudaFree(csr_cols_ptr_on_gpu);
    cudaFree(nz_values_ptr_on_gpu);
    cudaFree(random_vector_ptr_on_gpu);
    cudaFree(result_vector_ptr_on_gpu);

    free(csr_rows_ptr);
    free(csr_cols_ptr);
    free(csr_values_ptr);

    free(coo_nz_rows_ptr);
    free(coo_nz_cols_ptr);
    free(coo_nz_values_ptr);
    free(unused_imag_values_ptr);

    return 0;
}
