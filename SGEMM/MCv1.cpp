#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipblas.h>
#include <time.h>

#define WIDTH 1024

#define SMEM_POFFSET (BK+2)

const int M = WIDTH;
const int N = WIDTH;
const int K = WIDTH;

const int BM = 128;
const int BN = 128;
const int BK = 16;

const int tileBM = 64;
const int tileBN = 32;
const int tileBK = 8;

constexpr int WARPNUM = (BM/tileBM) * (BN/tileBN);
constexpr int LDSOFFSET = BM*SMEM_POFFSET;
constexpr int DBUFFEROFFSET = BM*SMEM_POFFSET + BN*BK;

const int WARPSIZE = 64;

using floatx4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using floatx2 = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;

#define FLOAT4(element) (reinterpret_cast<float4*>(&(element))[0])
#define FLOAT2(element) (reinterpret_cast<float2*>(&(element))[0])
#define FLOATX2(element) (reinterpret_cast<floatx2*>(&(element))[0])
#define FLOATX4(element) (reinterpret_cast<floatx4*>(&(element))[0])

__global__ 
__launch_bounds__(WARPSIZE*WARPNUM, 1) 
void Sgemm_MC(float* __restrict__ A, 
              float* __restrict__ B,
              float* __restrict__ C){

    extern __shared__ float SMEM[];
    // SMEM_A[BM*SMEM_POFFSET], SMEM_B[BK*BN], SMEM_C[BM*BN]
    float ra[2][4][2];
    float rb[2][2][2];
    float rc[2][4][4] = {0.0f};
    
    int warpIdx = threadIdx.z * blockDim.y + threadIdx.y;

    int warpOffsetA;
    int warpOffsetB;
    int warpOffsetC;

    int flagA = 0;
    int flagB = 1;
    int temp;

    warpOffsetA = (blockIdx.y*BM + warpIdx*(BM/WARPNUM))*K + 0*BK + (threadIdx.x>>4)*K + (threadIdx.x&15);

    #pragma unroll
    for(int i = 0; i < BM/WARPNUM/4; i++){
        SMEM[(warpIdx*(BM/WARPNUM)+(i*4)+(threadIdx.x>>4))*SMEM_POFFSET + (threadIdx.x&15)] = A[warpOffsetA];
        warpOffsetA += 4*K;
    }

    warpOffsetB = blockIdx.x*BN + (0*BK + warpIdx*(BK/WARPNUM)) * N + (threadIdx.x<<1);

    #pragma unroll
    for(int i = 0; i < BK/WARPNUM; i++){
        FLOAT2(SMEM[LDSOFFSET + (warpIdx*(BK/WARPNUM)+i)*BN + (threadIdx.x<<1)]) = FLOAT2(B[warpOffsetB]);
        warpOffsetB += N;
    }

    __syncthreads();

    for(int iter = 1; iter < K/BK; iter++){
        
        warpOffsetA = (blockIdx.y*BM + warpIdx*(BM/WARPNUM))*K + iter*BK + (threadIdx.x>>4)*K + (threadIdx.x&15);

        #pragma unroll
        for(int i = 0; i < BM/WARPNUM/4; i++){
            SMEM[flagB*DBUFFEROFFSET + (warpIdx*(BM/WARPNUM)+(i*4)+(threadIdx.x>>4))*SMEM_POFFSET + (threadIdx.x&15)] = A[warpOffsetA];
            warpOffsetA += (4*K);
        }

        warpOffsetB = blockIdx.x*BN + (iter*BK + warpIdx*(BK/WARPNUM)) * N + (threadIdx.x<<1);

        #pragma unroll
        for(int i = 0; i < BK/WARPNUM; i++){
            FLOAT2(SMEM[flagB*DBUFFEROFFSET + LDSOFFSET + (warpIdx*(BK/WARPNUM)+i)*BN + (threadIdx.x<<1)]) = FLOAT2(B[warpOffsetB]);
            warpOffsetB += N;
        }

        // __syncthreads();
        
        warpOffsetA = flagA*DBUFFEROFFSET + (threadIdx.y*tileBM + (threadIdx.x&15))*SMEM_POFFSET + (threadIdx.x>>4);
        warpOffsetB = flagA*DBUFFEROFFSET + LDSOFFSET + (threadIdx.x>>4)*BN + threadIdx.z*tileBN + (threadIdx.x&15);

        ra[0][0][0] = SMEM[warpOffsetA + 0*tileBK];
        ra[0][0][1] = SMEM[warpOffsetA + 0*tileBK + 4];
        ra[0][1][0] = SMEM[warpOffsetA + 0*tileBK + 16*SMEM_POFFSET];
        ra[0][1][1] = SMEM[warpOffsetA + 0*tileBK + 16*SMEM_POFFSET + 4];
        ra[0][2][0] = SMEM[warpOffsetA + 0*tileBK + 32*SMEM_POFFSET];
        ra[0][2][1] = SMEM[warpOffsetA + 0*tileBK + 32*SMEM_POFFSET + 4];
        ra[0][3][0] = SMEM[warpOffsetA + 0*tileBK + 48*SMEM_POFFSET];
        ra[0][3][1] = SMEM[warpOffsetA + 0*tileBK + 48*SMEM_POFFSET + 4];
        
        ra[1][0][0] = SMEM[warpOffsetA + 1*tileBK];
        ra[1][0][1] = SMEM[warpOffsetA + 1*tileBK + 4];
        ra[1][1][0] = SMEM[warpOffsetA + 1*tileBK + 16*SMEM_POFFSET];
        ra[1][1][1] = SMEM[warpOffsetA + 1*tileBK + 16*SMEM_POFFSET + 4];
        ra[1][2][0] = SMEM[warpOffsetA + 1*tileBK + 32*SMEM_POFFSET];
        ra[1][2][1] = SMEM[warpOffsetA + 1*tileBK + 32*SMEM_POFFSET + 4];
        ra[1][3][0] = SMEM[warpOffsetA + 1*tileBK + 48*SMEM_POFFSET];
        ra[1][3][1] = SMEM[warpOffsetA + 1*tileBK + 48*SMEM_POFFSET + 4];

        rb[0][0][0] = SMEM[warpOffsetB];
        rb[0][0][1] = SMEM[warpOffsetB + BN*4];
        rb[0][1][0] = SMEM[warpOffsetB + 16];
        rb[0][1][1] = SMEM[warpOffsetB + BN*4 + 16];

        FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[0][0][0]));  
        FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[0][1][0]));      
        FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[0][2][0]));  
        FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[0][3][0]));      
        
        FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[1][0][0]));      
        FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[1][1][0]));   
        FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[1][2][0]));      
        FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[1][3][0])); 

        rb[1][0][0] = SMEM[warpOffsetB + (tileBK*BN)];
        rb[1][0][1] = SMEM[warpOffsetB + (tileBK*BN) + BN*4];
        rb[1][1][0] = SMEM[warpOffsetB + (tileBK*BN) + 16];
        rb[1][1][1] = SMEM[warpOffsetB + (tileBK*BN) + BN*4 + 16];

        FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[0][0][0]));  
        FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[0][1][0]));      
        FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[0][2][0]));  
        FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[0][3][0]));      
        
        FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[1][0][0]));      
        FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[1][1][0]));   
        FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[1][2][0]));      
        FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[1][3][0]));   

        __syncthreads();

        temp = flagA;
        flagA = flagB;
        flagB = temp;
    }

    warpOffsetA = flagA*DBUFFEROFFSET + (threadIdx.y*tileBM+(threadIdx.x&15))*SMEM_POFFSET + (threadIdx.x>>4);
    warpOffsetB = flagA*DBUFFEROFFSET + LDSOFFSET + (threadIdx.x>>4)*BN + threadIdx.z*tileBN + (threadIdx.x&15);

    ra[0][0][0] = SMEM[warpOffsetA + 0*tileBK];
    ra[0][0][1] = SMEM[warpOffsetA + 0*tileBK + 4];
    ra[0][1][0] = SMEM[warpOffsetA + 0*tileBK + 16*SMEM_POFFSET];
    ra[0][1][1] = SMEM[warpOffsetA + 0*tileBK + 16*SMEM_POFFSET + 4];
    ra[0][2][0] = SMEM[warpOffsetA + 0*tileBK + 32*SMEM_POFFSET];
    ra[0][2][1] = SMEM[warpOffsetA + 0*tileBK + 32*SMEM_POFFSET + 4];
    ra[0][3][0] = SMEM[warpOffsetA + 0*tileBK + 48*SMEM_POFFSET];
    ra[0][3][1] = SMEM[warpOffsetA + 0*tileBK + 48*SMEM_POFFSET + 4];

    rb[0][0][0] = SMEM[warpOffsetB];
    rb[0][0][1] = SMEM[warpOffsetB + BN*4];
    rb[0][1][0] = SMEM[warpOffsetB + 16];
    rb[0][1][1] = SMEM[warpOffsetB + BN*4 + 16];
    ra[1][0][0] = SMEM[warpOffsetA + 1*tileBK];
    ra[1][0][1] = SMEM[warpOffsetA + 1*tileBK + 4];

    FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[0][0][0]));  
    FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[0][1][0]));      
    FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[0][2][0]));  
    FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[0][3][0]));      
    
    FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[1][0][0]));      
    FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[1][1][0]));   
    FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[1][2][0]));      
    FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[1][3][0])); 

    ra[1][1][0] = SMEM[warpOffsetA + 1*tileBK + 16*SMEM_POFFSET];
    ra[1][1][1] = SMEM[warpOffsetA + 1*tileBK + 16*SMEM_POFFSET + 4];
    ra[1][2][0] = SMEM[warpOffsetA + 1*tileBK + 32*SMEM_POFFSET];
    ra[1][2][1] = SMEM[warpOffsetA + 1*tileBK + 32*SMEM_POFFSET + 4];
    ra[1][3][0] = SMEM[warpOffsetA + 1*tileBK + 48*SMEM_POFFSET];
    ra[1][3][1] = SMEM[warpOffsetA + 1*tileBK + 48*SMEM_POFFSET + 4];

    rb[1][0][0] = SMEM[warpOffsetB + (tileBK*BN)];
    rb[1][0][1] = SMEM[warpOffsetB + (tileBK*BN) + BN*4];
    rb[1][1][0] = SMEM[warpOffsetB + (tileBK*BN) + 16];
    rb[1][1][1] = SMEM[warpOffsetB + (tileBK*BN) + BN*4 + 16];

    FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[0][0][0]));  
    FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[0][1][0]));      
    FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[0][2][0]));  
    FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[0][3][0]));      
    
    FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[1][0][0]));      
    FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[1][1][0]));   
    FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[1][2][0]));      
    FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[1][3][0]));   

    warpOffsetC = (blockIdx.y*BM + threadIdx.y*tileBM + (threadIdx.x>>4))*N + blockIdx.x*BN + threadIdx.z*tileBN + (threadIdx.x&15);
    
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 2; j++){
            C[warpOffsetC + (j*16) + (0*N)  ] = rc[j][i][0];
            C[warpOffsetC + (j*16) + (4*N)  ] = rc[j][i][1];
            C[warpOffsetC + (j*16) + (8*N)  ] = rc[j][i][2];
            C[warpOffsetC + (j*16) + (12*N) ] = rc[j][i][3];
        }
        warpOffsetC += (N*16);
    }
}

int main(int argc, char **argv)
{
    // 初始化hipRAND随机数发生器
    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(gen, 0);

    // 计时器
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float dt;

    // 分配矩阵内存空间
    float *C = (float *) malloc(M * N * sizeof(float));
    float *blas_C = (float *) malloc(M * N * sizeof(float));

    float *dA, *dB, *dC;
    hipMalloc(&dA, M * K * sizeof(float));
    hipMalloc(&dB, K * N * sizeof(float));
    hipMalloc(&dC, M * N * sizeof(float));

    // 随机生成矩阵A和B
    hipEventRecord(start);
    hiprandGenerateUniform(gen, dA, M * K);
    hiprandGenerateUniform(gen, dB, K * N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    printf("The M, N, K in this case is %d, %d, %d individually. \n", M, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP kernel launch before error: %s\n", hipGetErrorString(err));
    }

    dim3 blockPerGrid((N+(BN-1))/BN, (M+(BM-1))/BM, 1);
    dim3 threadPerBlock(WARPSIZE, BM/tileBM, BN/tileBN);

    int SMEM_BTYES = 2 * (BM*SMEM_POFFSET + BN*BK) * sizeof(float);
    // int SMEM_BTYES = (BM*BN) * sizeof(float);

    for(int i = 0; i < 3; i++){
        hipLaunchKernelGGL(Sgemm_MC, blockPerGrid, threadPerBlock, SMEM_BTYES, 0, dA, dB, dC);
    }
    hipDeviceSynchronize();

    hipEventRecord(start);
    for(int i = 0; i < 100; i++){
        hipLaunchKernelGGL(Sgemm_MC, blockPerGrid, threadPerBlock, SMEM_BTYES, 0, dA, dB, dC);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    dt /= 100.0f;  // 取平均
    printf("Average self-sgemm kernel time over servel runs: %8.3f ms.\n", dt);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP kernel launch after error: %s\n", hipGetErrorString(err));
    }

    hipMemcpy(C, dC, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // 创建hipBLAS句柄
    hipblasHandle_t handle;
    hipblasCreate(&handle);

      // 矩阵乘法（SGEMM）
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 预热3次
    for (int i = 0; i < 3; ++i) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    // hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, width, width, width,
    //              &alpha_single, d_B, CUDA_R_16F, width, d_A, CUDA_R_16F, width,
    //              &beta_single, d_C_cublas, CUDA_R_32F, width, CUDA_R_32F,
    //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    hipDeviceSynchronize();

    // 正式计算：执行10次并取平均时间
    hipEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    dt /= 100.0f;  // 取平均
    printf("Average matrix multiplication time over servel runs: %8.3f ms.\n", dt);

    hipMemcpy(blas_C, dC, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // Step 3: 验证两个结果是否一致（最大误差 <= 1e-3）
    float max_diff = 0.0f;
    bool flag = true;
    for (int i = 0; i < M * N; ++i) {
        int idx = i;
        float diff = fabs(C[idx] - blas_C[idx]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1e-3f) {
            printf("Mismatch at (%d, %d): custom=%f, hipBLAS=%f, diff=%f\n",
                    idx/N, idx%N, C[idx], blas_C[idx], diff);
            fprintf(stderr, "Verification failed: max difference exceeds 1e-3.\n");
            flag = false;
            break;
        }
    }
    if(flag){
        printf("Verification passed! Max difference = %.6f\n", max_diff);
    }

    // 清理资源
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hiprandDestroyGenerator(gen);
    hipblasDestroy(handle);
    free(C);
    free(blas_C);

    return 0;
}
