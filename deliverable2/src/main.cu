#include "cxxopts.hpp"
#include "mmio.h"

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cuda_runtime.h>
#include <cusparse.h>
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

__global__ void spmv_thread_per_row_opt(
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_idx,
    const double *__restrict__ values,
    const double *__restrict__ x,
    double *__restrict__ y,
    const int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows)
        return;

    int start  = row_ptr[row];
    int end    = row_ptr[row + 1];
    double sum = 0.0;

    int j = start;
    for (; j + 3 < end; j += 4) {
        int col0 = col_idx[j];
        int col1 = col_idx[j + 1];
        int col2 = col_idx[j + 2];
        int col3 = col_idx[j + 3];

        double val0 = values[j];
        double val1 = values[j + 1];
        double val2 = values[j + 2];
        double val3 = values[j + 3];

        sum += val0 * __ldg(&x[col0]);
        sum += val1 * __ldg(&x[col1]);
        sum += val2 * __ldg(&x[col2]);
        sum += val3 * __ldg(&x[col3]);
    }

    for (; j < end; j++) {
        int col    = col_idx[j];
        double val = values[j];
        sum += val * __ldg(&x[col]);
    }

    y[row] = sum;
}

__global__ void spmv_warp_per_row_shared(
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_idx,
    const double *__restrict__ values,
    const double *__restrict__ x,
    double *__restrict__ y,
    int num_rows,
    int x_len) {

    extern __shared__ double x_shared[];

    int thread_id = threadIdx.x;
    int warp_id_in_block = thread_id / warpSize;
    int lane = thread_id % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;

    if (global_warp_id >= num_rows)
        return;

    int row_start = row_ptr[global_warp_id];
    int row_end   = row_ptr[global_warp_id + 1];

    int min_col = INT_MAX, max_col = INT_MIN;
    for (int j = row_start + lane; j < row_end; j += warpSize) {
        int col = col_idx[j];
        if (col >= 0 && col < x_len) {
            min_col = min(min_col, col);
            max_col = max(max_col, col);
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        min_col = min(min_col, __shfl_down_sync(0xffffffff, min_col, offset));
        max_col = max(max_col, __shfl_down_sync(0xffffffff, max_col, offset));
    }

    min_col = __shfl_sync(0xffffffff, min_col, 0);
    max_col = __shfl_sync(0xffffffff, max_col, 0);

    int tile_offset = min_col;
    int tile_len = max_col - min_col + 1;
    const int MAX_TILE = 1024; // keeps shared memory usage per block under 8 KB

    double sum = 0.0;

    // check if we have enough shared memory
    if (tile_len > 0 && tile_len <= MAX_TILE && (tile_offset + tile_len <= x_len)) {
        for (int i = lane; i < tile_len; i += warpSize) {
            x_shared[i] = x[tile_offset + i];
        }

        __syncwarp();

        for (int j = row_start + lane; j < row_end; j += warpSize) {
            int col = col_idx[j];
            double val = values[j];
            if (col >= tile_offset && col < tile_offset + tile_len) {
                sum += val * x_shared[col - tile_offset];
            } else if (col >= 0 && col < x_len) {
                sum += val * x[col];
            }
        }
    } else {
        // if the tile is too large and does not fit in shared memory
        // I use directly global memory
        for (int j = row_start + lane; j < row_end; j += warpSize) {
            int col = col_idx[j];
            if (col >= 0 && col < x_len) {
                sum += values[j] * x[col];
            }
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        y[global_warp_id] = sum;
    }
}

/***
 * EIGEN RESULT CHECKERS
 */
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

/***
 * KERNEL PROFILERS
 */
#define N_RUN_GEOM_MEAN (10U)

void profile_kernel(
    const std::string &name,
    const size_t nzeros,
    const size_t num_rows,
    const size_t num_cols,
    const std::function<void()> &kernel_launcher) {
    std::vector<float> times_ms(N_RUN_GEOM_MEAN);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (unsigned i = 0; i < N_RUN_GEOM_MEAN; ++i) {
        cudaEventRecord(start);
        kernel_launcher();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_ms[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double log_sum = 0.0;
    for (float t : times_ms)
        log_sum += std::log(t);
    float geom_time_ms = std::exp(log_sum / N_RUN_GEOM_MEAN);

    size_t bytes     = nzeros * (sizeof(int) + sizeof(double)) + num_cols * sizeof(double) + num_rows * sizeof(double);
    float throughput = static_cast<float>(bytes) / (geom_time_ms * 1e6f);

    float time_s = geom_time_ms / 1000.0f;
    float gflops = (2.0f * static_cast<float>(nzeros)) / (time_s * 1e9f);

    std::cout << "\"" << name << "_executiontime\": " << geom_time_ms << "," << std::endl;
    std::cout << "\"" << name << "_throughput\": " << throughput << "," << std::endl;
    std::cout << "\"" << name << "_performance\": " << gflops << "," << std::endl;
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
    compute_eigen_result(csr_rows_ptr, csr_cols_ptr, csr_values_ptr, NZ_ROWS_COUNT, NZ_COLS_COUNT, random_vector, NZ_COUNT);

    int *coo_rows_ptr_on_gpu, *csr_rows_ptr_on_gpu, *csr_cols_ptr_on_gpu;
    double *nz_values_ptr_on_gpu, *random_vector_ptr_on_gpu, *result_vector_ptr_on_gpu;

    cudaMalloc(&coo_rows_ptr_on_gpu, sizeof(int) * NZ_COUNT);
    cudaMalloc(&csr_rows_ptr_on_gpu, sizeof(int) * CSR_ROWS_COUNT);
    cudaMalloc(&csr_cols_ptr_on_gpu, sizeof(int) * NZ_COUNT);
    cudaMalloc(&nz_values_ptr_on_gpu, sizeof(double) * NZ_COUNT);
    cudaMalloc(&random_vector_ptr_on_gpu, sizeof(double) * NZ_COLS_COUNT);
    cudaMalloc(&result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT);
    cudaMemcpy(coo_rows_ptr_on_gpu, coo_nz_rows_ptr, sizeof(int) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(csr_rows_ptr_on_gpu, csr_rows_ptr, sizeof(int) * CSR_ROWS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(csr_cols_ptr_on_gpu, csr_cols_ptr, sizeof(int) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(nz_values_ptr_on_gpu, csr_values_ptr, sizeof(double) * NZ_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(random_vector_ptr_on_gpu, random_vector, sizeof(double) * NZ_COLS_COUNT, cudaMemcpyHostToDevice);
    cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);

    {
        int blockSize = 128;
        int gridSize  = (NZ_ROWS_COUNT + blockSize - 1) / blockSize;
        // Warmup
        spmv_thread_per_row<<<gridSize, blockSize>>>(
            csr_rows_ptr_on_gpu,
            csr_cols_ptr_on_gpu,
            nz_values_ptr_on_gpu,
            random_vector_ptr_on_gpu,
            result_vector_ptr_on_gpu,
            NZ_ROWS_COUNT);

        spmv_thread_per_row<<<gridSize, blockSize>>>(
            csr_rows_ptr_on_gpu,
            csr_cols_ptr_on_gpu,
            nz_values_ptr_on_gpu,
            random_vector_ptr_on_gpu,
            result_vector_ptr_on_gpu,
            NZ_ROWS_COUNT);
    }

    {
        int blockSize = 128;
        int gridSize  = (NZ_ROWS_COUNT + blockSize - 1) / blockSize;

        profile_kernel("thread_per_row_baseline", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
            spmv_thread_per_row<<<gridSize, blockSize>>>(
                csr_rows_ptr_on_gpu,
                csr_cols_ptr_on_gpu,
                nz_values_ptr_on_gpu,
                random_vector_ptr_on_gpu,
                result_vector_ptr_on_gpu,
                NZ_ROWS_COUNT);
        });
        cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);
        CHECK_EIGEN(gpu_result, "thread_per_row_baseline");
    }
    {
        cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);
        int blockSize     = 128;
        int warpsPerBlock = blockSize / 32;
        int gridSize      = (NZ_ROWS_COUNT + warpsPerBlock - 1) / warpsPerBlock;

        profile_kernel("warp_per_row_baseline", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
            spmv_warp_per_row<<<gridSize, blockSize>>>(
                csr_rows_ptr_on_gpu,
                csr_cols_ptr_on_gpu,
                nz_values_ptr_on_gpu,
                random_vector_ptr_on_gpu,
                result_vector_ptr_on_gpu,
                NZ_ROWS_COUNT);
        });
        cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);
        CHECK_EIGEN(gpu_result, "baseline");
    }
    {
        cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);

        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        int nnz = 0;
        cudaMemcpy(&nnz, csr_rows_ptr_on_gpu + NZ_ROWS_COUNT, sizeof(int), cudaMemcpyDeviceToHost);

        cusparseCreateCsr(
            &matA,
            NZ_ROWS_COUNT,
            NZ_ROWS_COUNT,
            nnz,
            const_cast<int *>(csr_rows_ptr_on_gpu),
            const_cast<int *>(csr_cols_ptr_on_gpu),
            const_cast<double *>(nz_values_ptr_on_gpu),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F);

        cusparseCreateDnVec(&vecX, NZ_ROWS_COUNT, const_cast<double *>(random_vector_ptr_on_gpu), CUDA_R_64F);
        cusparseCreateDnVec(&vecY, NZ_ROWS_COUNT, result_vector_ptr_on_gpu, CUDA_R_64F);

        const double alpha = 1.0;
        const double beta  = 0.0;

        size_t bufferSize = 0;
        void *dBuffer     = nullptr;

        cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &bufferSize);

        cudaMalloc(&dBuffer, bufferSize);

        profile_kernel("cusparse", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
            cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, dBuffer);
        });

        cudaFree(dBuffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
        cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);

        CHECK_EIGEN(gpu_result, "cusparse");
    }

    /***
     * OPTIMIZED VERSIONS
     */
    {
        cudaMemset(result_vector_ptr_on_gpu, 0, sizeof(double) * NZ_ROWS_COUNT);

        int blockSize = 128;
        int gridSize  = (NZ_ROWS_COUNT + blockSize - 1) / blockSize;
        size_t shm    = 1024 * sizeof(double);

        profile_kernel("thread_per_row_optimized", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
            spmv_thread_per_row_opt<<<gridSize, blockSize, shm>>>(
                csr_rows_ptr_on_gpu,
                csr_cols_ptr_on_gpu,
                nz_values_ptr_on_gpu,
                random_vector_ptr_on_gpu,
                result_vector_ptr_on_gpu,
                NZ_ROWS_COUNT);
        });

        cudaMemcpy(gpu_result, result_vector_ptr_on_gpu, sizeof(double) * NZ_ROWS_COUNT, cudaMemcpyDeviceToHost);
        CHECK_EIGEN(gpu_result, "thread_per_row_optimized");
    }

    {
        int warps_per_block = 4;
        int threads_per_block = warps_per_block * 32;
        int total_warps = (NZ_ROWS_COUNT + 31) / 32;
        int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
        int shared_memory_bytes = 1024 * sizeof(double);  // max tile size

        profile_kernel("warp_per_row_shared_memory", NZ_COUNT, NZ_ROWS_COUNT, NZ_COLS_COUNT, [&]() {
            spmv_warp_per_row_shared<<<num_blocks, threads_per_block, shared_memory_bytes>>>(
            csr_rows_ptr_on_gpu, 
            csr_cols_ptr_on_gpu, 
            nz_values_ptr_on_gpu, 
            random_vector_ptr_on_gpu, 
            result_vector_ptr_on_gpu, 
            NZ_ROWS_COUNT, 
            NZ_COLS_COUNT);
        });
    }

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
