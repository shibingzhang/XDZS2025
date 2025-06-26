#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <math.h>
#include <fstream>
#include <hipsparse.h>
#include <rocsparse/rocsparse.h>

#define HIP_CHECK(cmd) do {                                  \
  hipError_t e = (cmd);                                      \
  if (e != hipSuccess) {                                     \
    std::cerr << "HIP Error: " << hipGetErrorString(e)       \
              << " at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                  \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

using namespace std;

class CSRMatrix{
public:
    float* values;
    int* col_index;
    int* row_position;
    int rows, cols;
    int nnz;
};

CSRMatrix readMTXFile_CSR(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file.");
    }

    std::string line;
    int rows, cols, nnz;

    while(getline(file, line)){
        if(line[0] == '%' || line.empty()) continue;
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        break;
    }

    float* values = new float[nnz];
    int* col_indices = new int[nnz];
    int* row_indices = new int[rows + 1];
    int row, col;
    float val;
    int sum = 0;
    row_indices[0] = 0;


    while(getline(file, line)){
        if(line[0] == '%' || line.empty()) continue;
        std::istringstream iss(line);
        iss >> row >> col >> val;

        row_indices[row]++;

        col_indices[sum] = col - 1;

        values[sum] = val;
        sum++;

    }
    row_indices[rows] = nnz;

    //处理row_indices数组，如何该处的值为零，让其等于前一个值，否则等于前面的累加值加上自己（即当前行前面的非零元素个数）
    sum = 0;
    for(int i = 1; i < rows; i++){
        if(row_indices[i] != 0){
            sum += row_indices[i];
            row_indices[i] = sum;
        }
        else row_indices[i] = row_indices[i - 1];
    }

    CSRMatrix csr_matrix;
    csr_matrix.rows = rows;
    csr_matrix.cols = cols;
    csr_matrix.nnz = nnz;
    csr_matrix.values = values;
    csr_matrix.col_index = col_indices;
    csr_matrix.row_position = row_indices;

    return csr_matrix;

}


bool checkResult(float* C, float* Cd, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            //cout << i << " " << j << " " << C[i * n + j] << " " << Cd[i * n + j] << endl;
            if(fabs(C[i * n + j] - Cd[i * n + j]) > 1e-3){
                //cout << 1e-7 * 10000000 << endl;
                //cout << 1e-6 * 1000000 << endl;
                cout << "difference: " << fabs(C[i * n + j] - Cd[i * n + j]) << endl;
                cout << "Error in " << i << " " << j << " " << C[i * n + j] << " " << Cd[i * n + j] << endl;
                return false;
            }
        }
    }
    cout << "Correct" << endl;
    return true;
}

bool checkResult1(float* C, float* dC, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(fabs(C[i * n + j] - dC[i + j * m]) > 1e-3){
            //if(C[i * n + j] != dC[i + j * m]){
                //cout << 1e-7 * 10000000 << endl;  
                cout << fabs(C[i * n + j] - dC[i + j * m]) << endl;
                cout << "Error in " << i << " " << j << " " << C[i * n + j] << " " << dC[i + j * m] << endl;
                return false;
            }
        }
    }
    cout << "Correct" << endl;
    return true;
}

// 使用CPU计算
void CSRonHost(CSRMatrix& A, float* B, float* C, int n){
    for(int i = 0; i < A.rows; i++){
        for(int j = A.row_position[i]; j < A.row_position[i + 1]; j++){
            for(int k = 0; k < n; k++){
                C[i * n + k] += A.values[j] * B[A.col_index[j] * n + k];
            }
        }
    }
}


void SpMM_sparse(const CSRMatrix &csr_matrix, int B_cols,
                   const int* row_position, const int* col_index, const float* values,
                   const float* Bd, float* Cd) {

    hipsparseHandle_t handle;
    hipsparseCreate(&handle);
    hipsparseMatDescr_t descr;
    hipsparseStatus_t hipsp_stat;
    hipsparseCreateMatDescr(&descr);
    hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
    float one=1, zero=0;
    hipsp_stat = hipsparseScsrmm2(handle,
                                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                HIPSPARSE_OPERATION_TRANSPOSE,
                                csr_matrix.rows,
                                B_cols,
                                csr_matrix.cols,
                                csr_matrix.nnz,
                                &one,
                                descr,
                                values,
                                row_position,
                                col_index,
                                Bd,
                                B_cols,
                                &zero,
                                Cd,
                                csr_matrix.rows);
    if (hipsp_stat != HIPSPARSE_STATUS_SUCCESS) {
        printf("hipSparse Error: \n");
    }
    
}


__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0.0;
}

__global__ void topoCacheCoarsenSPMMKernelWarp64(
  int m, int k, const int* A_indptr, const int* A_indices, const float* A_value,
  const float* B, float* C
) {
  extern __shared__ int sh[];  // 用于共享内存缓存索引和值

  // warp size 改为 64
  int sm_offset = threadIdx.y * 64;
  int thread_idx = sm_offset + threadIdx.x;
  int value_off = 64 * blockDim.y;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 7) + threadIdx.x;  // tile_k * 128，每线程算两个列
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;

    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();

    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 64) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncthreads();
        ptr += 64;

        for (int kk = 0; kk < 64 && jj + kk < hb; kk++) {
          offset = sh[sm_offset + kk] + cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[sm_offset + kk + value_off] * B[offset]);
          acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[sm_offset + kk + value_off] * B[offset + 64]);
        }
        __syncthreads();
      }
      offset = rid * k + cid;
      C[offset] = acc1;
      C[offset + 64] = acc2;
    }
    else {
      int nout = (k - cid + 63) / 64;
      for (int jj = lb; jj < hb; jj += 64) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncthreads();
        ptr += 64;

        for (int kk = 0; kk < 64 && jj + kk < hb; kk++) {
          offset = sh[sm_offset + kk] + cid;
          if (nout > 0)
            acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[sm_offset + kk + value_off] * B[offset]);
          if (nout > 1)
            acc2 = sum_reduce(acc2, reinterpret_cast<float*>(sh)[sm_offset + kk + value_off] * B[offset + 64]);
        }
        __syncthreads();
      }
      offset = rid * k + cid;
      if (nout > 0)
        C[offset] = acc1;
      if (nout > 1)
        C[offset + 64] = acc2;
    }
  }
}

__global__ void topoCacheSPMMKernelWarp64(
  int m, int k, const int* A_indptr, const int* A_indices,
  const float* A_value, const float* B, float* C 
) {
  extern __shared__ int sh[];  // 前半为 int，后半 reinterpret 为 float

  // 关键变化：线程组织以 64 为 warp size
  const int sm_offset = threadIdx.y * 64;  // 一行64个线程
  const int thread_idx = sm_offset + threadIdx.x;
  const int value_off = blockDim.y * 64;  // 对应原 value 偏移

  int cid = (blockIdx.y << 6) + threadIdx.x;      // col id: 64个线程做64列
  int rid = blockDim.y * blockIdx.x + threadIdx.y;  // 每个block处理4行

  if (rid < m && cid < k) {
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    float acc1 = sum_init();

    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 64) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncthreads();
        ptr += 64;
        for (int kk = 0; kk < 64 && jj + kk < hb; kk++) {
          int offset = sh[sm_offset + kk] + cid;
          float val = reinterpret_cast<float*>(sh)[sm_offset + kk + value_off];
          acc1 = sum_reduce(acc1, val * B[offset]);
        }
        __syncthreads();
      }
      C[rid * k + cid] = acc1;
    } else {
      // boundary warp
      int nout = (k - cid + 63) / 64;
      for (int jj = lb; jj < hb; jj += 64) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncthreads();
        ptr += 64;
        for (int kk = 0; kk < 64 && jj + kk < hb; kk++) {
          int offset = sh[sm_offset + kk] + cid;
          float val = reinterpret_cast<float*>(sh)[sm_offset + kk + value_off];
          if (nout > 0) acc1 = sum_reduce(acc1, val * B[offset]);
        }
        __syncthreads();
      }
      if (nout > 0) C[rid * k + cid] = acc1;
    }
  }
}
__global__ void topoCacheSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  int value_off = blockDim.y * blockDim.x;  

  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncthreads();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncthreads();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          reinterpret_cast<float*>(sh)[thread_idx + value_off] = A_value[ptr];
        }
        __syncthreads();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, reinterpret_cast<float*>(sh)[(sm_offset+kk+value_off)]*B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncthreads();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
    }
  }
}
__global__ void topoSimpleSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* A_value, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]*A_value[ptr]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}

/**
 * topoSimpleSPMMKernel：普通实现
 * topoCacheSPMMKernel：一个warp为32，一个线程计算一个结果
 * topoCacheSPMMKernelWarp64：一个warp为64，一个线程计算一个结果
 * topoCacheCoarsenSPMMKernelWarp64：一个warp为64，一个线程计算两个结果
*/
void SpMM(const CSRMatrix &csr_matrix, int B_cols,
                      const int* row_position, const int* col_index, const float* values,
                      const float* Bd, float* Cd) {
    if (B_cols <= 16) {
        const int row_per_block = 128 / B_cols;
        const int n_block = (csr_matrix.rows + row_per_block - 1) / row_per_block;
        const dim3 block(B_cols, row_per_block, 1);
        const dim3 grid(n_block, 1, 1);
        topoSimpleSPMMKernel<<<grid, block>>>(
            csr_matrix.rows, B_cols, row_position, col_index, values, Bd, Cd
        );
    } else if (B_cols <= 32) {
        const int tile_k = (B_cols + 31) / 32;
        const int n_block = (csr_matrix.rows + 3) / 4;
        const dim3 block(32, 4, 1);
        const dim3 grid(n_block, tile_k, 1);
        size_t shared = 256 * sizeof(int);
        topoCacheSPMMKernel<<<grid, block, shared>>>(
            csr_matrix.rows, B_cols, row_position, col_index, values, Bd, Cd
        );
    }else if (B_cols <= 64) {
        const int tile_k = (B_cols + 63) / 64;  // 每个warp处理64列
        const int n_block = (csr_matrix.rows + 3) / 4;
        const dim3 block(64, 4, 1);  // 每个block是64×4线程
        const dim3 grid(n_block, tile_k, 1);
        size_t shared = 2 * 64 * 4 * sizeof(int);  // 每个线程1个int，1个float，共256 threads
        topoCacheSPMMKernelWarp64<<<grid, block, shared>>>(
            csr_matrix.rows, B_cols, row_position, col_index, values, Bd, Cd
        );
    } else {
        const int tile_k = (B_cols + 127) / 128;  // 每个线程块处理128列输出
        const int n_block = (csr_matrix.rows + 3) / 4;  // 每个block负责4行
        const dim3 block(64, 4, 1);  // 每个线程块64x4=256线程
        const dim3 grid(n_block, tile_k, 1);     // 网格维度
        size_t shared = 2 * 64 * 4 * sizeof(int);  // 共享内存大小（indices + values）
        topoCacheCoarsenSPMMKernelWarp64<<<grid, block, shared>>>(
            csr_matrix.rows, B_cols, row_position, col_index, values, Bd, Cd
        );
    }
}


int main(int argc, char* argv[]){

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];

    CSRMatrix csr_matrix = readMTXFile_CSR(filename);

    int B_cols = 16;
    float* B = new float[B_cols * csr_matrix.cols]; // 存储稠密矩阵B
    float* C = new float[csr_matrix.rows * B_cols]; // 存储CPU计算结果
    float* C_sparse = new float[csr_matrix.rows * B_cols]; //存储稀疏库计算结果
    float* C_kernel = new float[csr_matrix.rows * B_cols]; //存储kernel计算结果

    struct timeval start, end;

    for(int i = 0; i < csr_matrix.rows * B_cols; i++){
        B[i] = float((rand() % 101) - 50) / 100.0f;
        C[i] = 0.0;
    }

    gettimeofday(&start, NULL);
    CSRonHost(csr_matrix, B, C, B_cols);//CPU计算
    gettimeofday(&end, NULL);
    double time_cpu = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "SpMM on CPU time: " << time_cpu << "s" << endl;


    float* values;//矩阵的非零值
    int* col_index, *row_position;//是 CSR 格式稀疏矩阵的数据部分，也会被复制到 GPU 上，分别存储列索引和行指针。
    float* Bd, * Cd_base, * Cd_op;//分配GPU上的数组。对应稠密矩阵B和结果矩阵C

    HIP_CHECK(hipMalloc((void**)&Bd, B_cols * csr_matrix.cols * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&Cd_base, csr_matrix.rows * B_cols * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&Cd_op, csr_matrix.rows * B_cols * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&values, csr_matrix.nnz * sizeof(float))); // 非零值
    HIP_CHECK(hipMalloc((void**)&col_index, csr_matrix.nnz * sizeof(int))); // 列数组
    HIP_CHECK(hipMalloc((void**)&row_position, (csr_matrix.rows + 1) * sizeof(int))); // 行数组

    HIP_CHECK(hipMemcpy(Bd, B, B_cols * csr_matrix.cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(values, csr_matrix.values, csr_matrix.nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(col_index, csr_matrix.col_index, csr_matrix.nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(row_position, csr_matrix.row_position, (csr_matrix.rows + 1) * sizeof(int), hipMemcpyHostToDevice));

    gettimeofday(&start, NULL);
    for(int i = 0;i < 100; i++){
        SpMM_sparse(csr_matrix, B_cols, row_position, col_index, values, Bd, Cd_base);
    }
    HIP_CHECK(hipDeviceSynchronize());
    gettimeofday(&end, NULL);
    HIP_CHECK(hipMemcpy(C_sparse, Cd_base, csr_matrix.rows * B_cols * sizeof(float), hipMemcpyDeviceToHost));
    if(checkResult1(C, C_sparse, csr_matrix.rows, B_cols)){
        printf("pass \n");
    }
    double time_dcu_sparse = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "SpMM on DCU sparse time: " << time_dcu_sparse << "s" << endl;

    // SpMM-DCU-warp
    gettimeofday(&start, NULL);
    for(int i = 0;i < 100; i++){
        SpMM(csr_matrix, B_cols, row_position, col_index, values, Bd, Cd_op);
    }
    HIP_CHECK(hipDeviceSynchronize());
    gettimeofday(&end, NULL);
    HIP_CHECK(hipMemcpy(C_kernel, Cd_op, csr_matrix.rows * B_cols * sizeof(float), hipMemcpyDeviceToHost));
    if(checkResult(C_kernel, C, csr_matrix.rows, B_cols)){
        printf("pass \n");
    }
    double time_dcu_op = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "SpMM on DCU op time: " << time_dcu_op << "s" << endl;
    double speed_dcu = time_dcu_sparse / time_dcu_op;
    cout << "speed up with sparse: " << speed_dcu << endl;
    
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd_base));
    HIP_CHECK(hipFree(Cd_op));
    HIP_CHECK(hipFree(values));
    HIP_CHECK(hipFree(col_index));
    HIP_CHECK(hipFree(row_position));

    delete []B;
    delete []C;
    delete []C_sparse;
    delete []C_kernel;

    return 0;

}
