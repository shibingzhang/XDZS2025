#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hipblas.h>
#include <rocblas.h>
#include <time.h>

#define WIDTH 8192
// better in WIDTH >= 2048 for (2048/256) * (2048/128) = 8 * 16 = 128 or at least one >= 2048 and the other >= 1024

#define DATA_TYPE float
#define RA_FOR_TRANS_TYPE float
#define THREAD_TILE_X 8
#define THREAD_TILE_Y 8

#define SMEM_POFFSET (BK+2)

constexpr int M = WIDTH;
constexpr int N = WIDTH;
constexpr int K = WIDTH;

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int tileBM = 64;
constexpr int tileBN = 64;
constexpr int tileBK = 8;

constexpr int WARPNUM = (BM/tileBM) * (BN/tileBN);
constexpr int LDSOFFSET = BM*SMEM_POFFSET;

constexpr int WARPSIZE = 64;

using floatx4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using floatx2 = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;

#define FLOAT4(element) (reinterpret_cast<float4*>(&(element))[0])
#define FLOAT2(element) (reinterpret_cast<float2*>(&(element))[0])
#define FLOATX2(element) (reinterpret_cast<floatx2*>(&(element))[0])
#define FLOATX4(element) (reinterpret_cast<floatx4*>(&(element))[0])

__host__ __device__ dim3 get_swizzled_data_block_idx(const int gridDim_x,
                                                     const int gridDim_y,
                                                     const int blockIdx_x,
                                                     const int blockIdx_y,
                                                     const int TILE_WIDTH) {
  const int blocks_per_tile = gridDim_y * TILE_WIDTH;
  const int num_tiles = gridDim_x / TILE_WIDTH;
  const int block_idx_flatterned = blockIdx_y * gridDim_x + blockIdx_x;
  const int tile_id = block_idx_flatterned / blocks_per_tile;
  int block_idx_in_tile = block_idx_flatterned % blocks_per_tile;
  int block_idx_x_in_tile = block_idx_in_tile % TILE_WIDTH;
  int block_idx_y_in_tile = block_idx_in_tile / TILE_WIDTH;
  if (blockIdx_x >= num_tiles * TILE_WIDTH) {
    const int last_tile_dim_x = gridDim_x - num_tiles * TILE_WIDTH;
    block_idx_x_in_tile = block_idx_in_tile % last_tile_dim_x;
    block_idx_y_in_tile = block_idx_in_tile / last_tile_dim_x;
  }
  const int swizzled_block_idx_flatterned =
      block_idx_y_in_tile * gridDim_x + block_idx_x_in_tile + tile_id * TILE_WIDTH;
  const int swizzled_block_idx_x = swizzled_block_idx_flatterned % gridDim_x;
  const int swizzled_block_idx_y = swizzled_block_idx_flatterned / gridDim_x;

  return dim3(swizzled_block_idx_x, swizzled_block_idx_y, 1);
}

__global__ 
__launch_bounds__(WARPSIZE*WARPNUM, 1) 
void Sgemm_MC(float* __restrict__ A, 
              float* __restrict__ B, 
              float* __restrict__ C){

    extern __shared__ float SMEM[];
    // SMEM_A[BM*SMEM_POFFSET], SMEM_B[BK*BN], SMEM_C[BM*BN]
    float ra[2][4][2];
    float rb[2][4][2];
    float rc[4][4][4] = {0.0f};
    
    int warpIdx = threadIdx.z * blockDim.y + threadIdx.y;

    int warpOffsetA;
    int warpOffsetB;
    int warpOffsetA_2;
    int warpOffsetB_2;
    int warpOffsetA_3;
    int warpOffsetB_3;
    int warpOffsetC;
    
    // const dim3 swizzled_block_idx = get_swizzled_data_block_idx(gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, 64);
    // int blockIdx_x = swizzled_block_idx.x;
    // int blockIdx_y = swizzled_block_idx.y;

    warpOffsetA_3 = (threadIdx.y*tileBM+(threadIdx.x&15))*SMEM_POFFSET + ((threadIdx.x>>4));
    warpOffsetB_3 = LDSOFFSET + (threadIdx.x>>4)*BN + threadIdx.z*tileBN + (threadIdx.x&15);

    for(int iter = 0; iter < K/BK; iter++){
        
        warpOffsetA = (blockIdx.y*BM + warpIdx*(BM/WARPNUM) + (threadIdx.x>>4))*K + iter*BK + (threadIdx.x&15);
        warpOffsetA_2 = (warpIdx*(BM/WARPNUM) + (threadIdx.x>>4))*SMEM_POFFSET + (threadIdx.x&15);

        #pragma unroll
        for(int i = 0; i < BM/WARPNUM/4; i++){
            SMEM[warpOffsetA_2] = A[warpOffsetA];
            warpOffsetA += (4*K);
            warpOffsetA_2 += (4*SMEM_POFFSET);
        }

        warpOffsetB = (iter*BK + warpIdx*(BK/WARPNUM))*N + blockIdx.x*BN + (threadIdx.x<<1);
        warpOffsetB_2 = LDSOFFSET + (warpIdx*(BK/WARPNUM))*BN + (threadIdx.x<<1);

        #pragma unroll
        for(int i = 0; i < BK/WARPNUM; i++){
            FLOAT2(SMEM[warpOffsetB_2]) = FLOAT2(B[warpOffsetB]);
            warpOffsetB += N;
            warpOffsetB_2 += BN;
        }

        __syncthreads();
        
        ra[0][0][0] = SMEM[warpOffsetA_3 + 0*tileBK];
        ra[0][0][1] = SMEM[warpOffsetA_3 + 0*tileBK + 4];
        ra[0][1][0] = SMEM[warpOffsetA_3 + 0*tileBK + 16*SMEM_POFFSET];
        ra[0][1][1] = SMEM[warpOffsetA_3 + 0*tileBK + 16*SMEM_POFFSET + 4];
        ra[0][2][0] = SMEM[warpOffsetA_3 + 0*tileBK + 32*SMEM_POFFSET];
        ra[0][2][1] = SMEM[warpOffsetA_3 + 0*tileBK + 32*SMEM_POFFSET + 4];
        ra[0][3][0] = SMEM[warpOffsetA_3 + 0*tileBK + 48*SMEM_POFFSET];
        ra[0][3][1] = SMEM[warpOffsetA_3 + 0*tileBK + 48*SMEM_POFFSET + 4];
        
        ra[1][0][0] = SMEM[warpOffsetA_3 + 1*tileBK];
        ra[1][0][1] = SMEM[warpOffsetA_3 + 1*tileBK + 4];
        ra[1][1][0] = SMEM[warpOffsetA_3 + 1*tileBK + 16*SMEM_POFFSET];
        ra[1][1][1] = SMEM[warpOffsetA_3 + 1*tileBK + 16*SMEM_POFFSET + 4];
        ra[1][2][0] = SMEM[warpOffsetA_3 + 1*tileBK + 32*SMEM_POFFSET];
        ra[1][2][1] = SMEM[warpOffsetA_3 + 1*tileBK + 32*SMEM_POFFSET + 4];
        ra[1][3][0] = SMEM[warpOffsetA_3 + 1*tileBK + 48*SMEM_POFFSET];
        ra[1][3][1] = SMEM[warpOffsetA_3 + 1*tileBK + 48*SMEM_POFFSET + 4];

        rb[0][0][0] = SMEM[warpOffsetB_3];
        rb[0][0][1] = SMEM[warpOffsetB_3 + BN*4];
        rb[0][1][0] = SMEM[warpOffsetB_3 + 16];
        rb[0][1][1] = SMEM[warpOffsetB_3 + BN*4 + 16];
        rb[0][2][0] = SMEM[warpOffsetB_3 + 32];
        rb[0][2][1] = SMEM[warpOffsetB_3 + BN*4 + 32];
        rb[0][3][0] = SMEM[warpOffsetB_3 + 48];
        rb[0][3][1] = SMEM[warpOffsetB_3 + BN*4 + 48];

        FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[0][0][0]));  
        FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[0][1][0]));      
        FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[0][2][0]));  
        FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][0][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[0][3][0]));  
        
        FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[1][0][0]));      
        FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[1][1][0]));   
        FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[1][2][0]));      
        FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][1][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[1][3][0])); 
        
        FLOATX4(rc[2][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][2][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[2][0][0]));  
        FLOATX4(rc[2][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][2][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[2][1][0]));      
        FLOATX4(rc[2][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][2][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[2][2][0]));  
        FLOATX4(rc[2][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][2][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[2][3][0]));  
        
        FLOATX4(rc[3][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][3][0]), FLOATX2(ra[0][0][0]), FLOATX4(rc[3][0][0]));      
        FLOATX4(rc[3][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][3][0]), FLOATX2(ra[0][1][0]), FLOATX4(rc[3][1][0]));  
        FLOATX4(rc[3][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][3][0]), FLOATX2(ra[0][2][0]), FLOATX4(rc[3][2][0]));      
        FLOATX4(rc[3][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[0][3][0]), FLOATX2(ra[0][3][0]), FLOATX4(rc[3][3][0])); 

        rb[1][0][0] = SMEM[warpOffsetB_3 + (tileBK*BN)];
        rb[1][0][1] = SMEM[warpOffsetB_3 + (tileBK*BN) + BN*4];
        rb[1][1][0] = SMEM[warpOffsetB_3 + (tileBK*BN) + 16];
        rb[1][1][1] = SMEM[warpOffsetB_3 + (tileBK*BN) + BN*4 + 16];
        rb[1][2][0] = SMEM[warpOffsetB_3 + (tileBK*BN) + 32];
        rb[1][2][1] = SMEM[warpOffsetB_3 + (tileBK*BN) + BN*4 + 32];
        rb[1][3][0] = SMEM[warpOffsetB_3 + (tileBK*BN) + 48];
        rb[1][3][1] = SMEM[warpOffsetB_3 + (tileBK*BN) + BN*4 + 48];
  
        FLOATX4(rc[0][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[0][0][0]));  
        FLOATX4(rc[0][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[0][1][0]));      
        FLOATX4(rc[0][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[0][2][0]));  
        FLOATX4(rc[0][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][0][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[0][3][0]));      
        
        FLOATX4(rc[1][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[1][0][0]));      
        FLOATX4(rc[1][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[1][1][0]));   
        FLOATX4(rc[1][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[1][2][0]));      
        FLOATX4(rc[1][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][1][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[1][3][0]));   
        
        FLOATX4(rc[2][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][2][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[2][0][0]));  
        FLOATX4(rc[2][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][2][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[2][1][0]));      
        FLOATX4(rc[2][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][2][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[2][2][0]));  
        FLOATX4(rc[2][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][2][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[2][3][0]));          
    
        FLOATX4(rc[3][0][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][3][0]), FLOATX2(ra[1][0][0]), FLOATX4(rc[3][0][0]));      
        FLOATX4(rc[3][1][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][3][0]), FLOATX2(ra[1][1][0]), FLOATX4(rc[3][1][0]));  
        FLOATX4(rc[3][2][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][3][0]), FLOATX2(ra[1][2][0]), FLOATX4(rc[3][2][0]));      
        FLOATX4(rc[3][3][0]) = __builtin_amdgcn_mmac_f32_16x16x8f32(FLOATX2(rb[1][3][0]), FLOATX2(ra[1][3][0]), FLOATX4(rc[3][3][0])); 
        __syncthreads();
    }

    warpOffsetC = (blockIdx.y*BM + threadIdx.y*tileBM + (threadIdx.x>>4))*N + blockIdx.x*BN + threadIdx.z*tileBN + (threadIdx.x&15);
    
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 4; j++){
            C[warpOffsetC + (j*16) + (0*N)  ] = rc[j][i][0];
            C[warpOffsetC + (j*16) + (4*N)  ] = rc[j][i][1];
            C[warpOffsetC + (j*16) + (8*N)  ] = rc[j][i][2];
            C[warpOffsetC + (j*16) + (12*N) ] = rc[j][i][3];
        }
        warpOffsetC += (N*16);
    }

}

__global__ __launch_bounds__(256) void assign_scalar(float* matrix, int M, int N, float scalar) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        int idx = row * N + col; 
        matrix[idx] = scalar;
    }
}

void verify_result(const float* C, const float* blas_C, int M, int N) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    bool flag = true;

    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(C[i] - blas_C[i]);
        sum_diff += diff;

        if (diff > max_diff) {
            max_diff = diff;
        }

        if (diff > 2e-1f) {
            printf("Mismatch at (%d, %d): custom=%f, blas=%f, diff=%f\n",
                   i / N, i % N, C[i], blas_C[i], diff);
            fprintf(stderr, "Verification failed: max difference exceeds 2e-1.\n");
            flag = false;
            break;
        }
    }

    float mean_diff = sum_diff / (M * N);

    if (flag) {
        printf("Verification passed!\n");
        printf("Max difference = %.6f\n", max_diff);
        printf("Mean absolute error = %.6f\n", mean_diff);
    }
}

__global__  void Sgemm_SIMT(DATA_TYPE* __restrict__ Ad,
                            DATA_TYPE* __restrict__ Bd,
                            DATA_TYPE* __restrict__ Cd,
                            int width) {

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bdim_x = blockDim.x;
    int bdim_y = blockDim.y;

    int index_j = (bid_y * bdim_y + tid_y) << 3;
    int index_i = (bid_x * bdim_x + tid_x) << 3;
    int index_Cd = index_j * N + index_i;  //计算结果矩阵Cd 的索引

    int tid_in_blk = tid_y * bdim_x + tid_x;
    int offset_inner = tid_in_blk << 2;   //线程在线程块的偏移量

    int ldsm_row = ((bid_y * bdim_y) << 3) + (offset_inner >> 3);
    int ldsm_col = (offset_inner & 7);
    // 下面要改
    int ldsn_row = offset_inner >> 7;   //B矩阵在行上的偏移量
    int ldsn_col = ((bid_x * bdim_x) << 3) + (offset_inner & 127);
    DATA_TYPE rA[8], rB[8], rC[64] = {0};
    // DATA_TYPE rA[16], rB[16], rC[128] = {0};
    RA_FOR_TRANS_TYPE rA_for_trans[4];
    RA_FOR_TRANS_TYPE rB_for_none[4];
    // __shared__ DATA_TYPE ldsa[1024];
    // __shared__ DATA_TYPE ldsb[1024];
    __shared__ DATA_TYPE ldsa[2][1024];     // blockDim.x * blockDim.y * 4(vector dim)
    __shared__ DATA_TYPE ldsb[2][1024];

    // int diverse_id = (tid_in_blk / 32) * 128 + ((tid_in_blk % 2)) * 64 + (tid_in_blk % 32)/2;
    int diverse_id = ((tid_in_blk >> 5) << 7) + ((tid_in_blk & 1) << 6) + ((tid_in_blk & 31) >> 1);
    // int diverse_id = 0;

    FLOAT4(rA_for_trans) = FLOAT4(Ad[ldsm_row * K + ldsm_col]);
    int txp = ((tid_in_blk & 1) << 9) + (tid_in_blk >> 1);
    // 写共享内存A时bank冲突不严重，2路bank冲突
    ldsa[0][txp] = rA_for_trans[0];
    ldsa[0][txp + 128] = rA_for_trans[1];
    ldsa[0][txp + 256] = rA_for_trans[2];
    ldsa[0][txp + 384] = rA_for_trans[3];
    // FLOAT4(ldsb[0][offset_inner]) = FLOAT4(Bd[(ldsn_row) * width + ldsn_col]);
    // 产生4路bank冲突
    FLOAT4(rB_for_none) = FLOAT4(Bd[(ldsn_row) * N + ldsn_col]);
    // FLOAT4(ldsb[0][offset_inner]) = FLOAT4(rB_for_none);
    ldsb[0][diverse_id] = rB_for_none[0];
    ldsb[0][diverse_id + 16] = rB_for_none[1];
    ldsb[0][diverse_id + 32] = rB_for_none[2];
    ldsb[0][diverse_id + 48] = rB_for_none[3];
    __syncthreads();

    int db_idx;
    int db_nxt_idx;

    for (int j = 8; j < K; j += 8) {
        db_idx = ((j >> 3) & 1);
        db_nxt_idx = db_idx ^ 1;
        FLOAT4(rA_for_trans) = FLOAT4(Ad[ldsm_row * K + ldsm_col + j]);
        FLOAT4(rB_for_none) = FLOAT4(Bd[(ldsn_row + j) * N + ldsn_col]);

        ldsa[db_idx][txp] = rA_for_trans[0];
        ldsa[db_idx][txp + 128] = rA_for_trans[1];
        ldsa[db_idx][txp + 256] = rA_for_trans[2];
        ldsa[db_idx][txp + 384] = rA_for_trans[3];

        ldsb[db_idx][diverse_id] = rB_for_none[0];
        ldsb[db_idx][diverse_id + 16] = rB_for_none[1];
        ldsb[db_idx][diverse_id + 32] = rB_for_none[2];
        ldsb[db_idx][diverse_id + 48] = rB_for_none[3];

        for (int i = 0; i < 8; i++) {    
            FLOAT4(rA[0]) = FLOAT4(ldsa[db_nxt_idx][(i << 7) + (tid_y << 3)]);
            FLOAT4(rA[4]) = FLOAT4(ldsa[db_nxt_idx][(i << 7) + (tid_y << 3) + 4]); 

            rB[0] = ldsb[db_nxt_idx][(i << 7) + (tid_x)];
            rB[1] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (1 << 4)];
            rB[2] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (2 << 4)];
            rB[3] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (3 << 4)];
            rB[4] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (4 << 4)];
            rB[5] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (5 << 4)];
            rB[6] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (6 << 4)];
            rB[7] = ldsb[db_nxt_idx][(i << 7) + (tid_x) + (7 << 4)];

            #pragma unroll
            for(int inner_i = 0; inner_i < 8; inner_i++){
                #pragma unroll
                for(int inner_j = 0; inner_j < 8; inner_j++){
                    rC[(inner_i << 3) + inner_j] += rA[inner_i] * rB[inner_j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 8; i++) {
    FLOAT4(rA[0]) = FLOAT4(ldsa[db_idx][(i << 7) + (tid_y << 3)]);
    FLOAT4(rA[4]) = FLOAT4(ldsa[db_idx][(i << 7) + (tid_y << 3) + 4]); 

    rB[0] = ldsb[db_idx][(i << 7) + (tid_x)];
    rB[1] = ldsb[db_idx][(i << 7) + (tid_x) + (1 << 4)];
    rB[2] = ldsb[db_idx][(i << 7) + (tid_x) + (2 << 4)];
    rB[3] = ldsb[db_idx][(i << 7) + (tid_x) + (3 << 4)];
    rB[4] = ldsb[db_idx][(i << 7) + (tid_x) + (4 << 4)];
    rB[5] = ldsb[db_idx][(i << 7) + (tid_x) + (5 << 4)];
    rB[6] = ldsb[db_idx][(i << 7) + (tid_x) + (6 << 4)];
    rB[7] = ldsb[db_idx][(i << 7) + (tid_x) + (7 << 4)];

    #pragma unroll
    for(int inner_i = 0; inner_i < 8; inner_i++){
        #pragma unroll
        for(int inner_j = 0; inner_j < 8; inner_j++){
            rC[(inner_i << 3) + inner_j] += rA[inner_i] * rB[inner_j];
        }
    }
}
    #pragma unroll
    for(int inner_i = 0; inner_i < 8; inner_i++){
        int tmp = N * inner_i;
        #pragma unroll
        for(int inner_j = 0; inner_j < 8; inner_j += 4){
            FLOAT4(Cd[index_Cd + tmp + inner_j]) = FLOAT4(rC[(inner_i << 3) + inner_j]);
        }
    }
}

int main(int argc, char **argv)
{
    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(gen, 0);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float dt;

    float *C = (float *) malloc(M * N * sizeof(float));
    float *blas_C = (float *) malloc(M * N * sizeof(float));

    float *dA, *dB, *dC, *dC_blas;
    hipMalloc(&dA, M * K * sizeof(float));
    hipMalloc(&dB, K * N * sizeof(float));
    hipMalloc(&dC, M * N * sizeof(float));
    hipMalloc(&dC_blas, M * N * sizeof(float));


    dim3 blockDimMA(16, 16);
    dim3 gridDimMA((K + blockDimMA.x - 1) / blockDimMA.x,
                 (M + blockDimMA.y - 1) / blockDimMA.y);
    dim3 blockDimMB(16, 16);
    dim3 gridDimMB((K + blockDimMB.x - 1) / blockDimMB.x,
                 (M + blockDimMB.y - 1) / blockDimMB.y);

    hipEventRecord(start);
    hiprandGenerateUniform(gen, dA, M * K);
    hiprandGenerateUniform(gen, dB, K * N);
    // assign_scalar<<<gridDimMA, blockDimMA>>>(dA, M, K, 0.2f);
    // assign_scalar<<<gridDimMB, blockDimMB>>>(dB, K, N, 0.2f);
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

    int SMEM_BTYES = (BM*SMEM_POFFSET + BN*BK) * sizeof(float);
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
    dt /= 100.0f;  
    printf("Average MC-sgemm kernel time over servel runs: %8.3f ms.\n", dt);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP kernel launch after error: %s\n", hipGetErrorString(err));
    }

    hipMemcpy(C, dC, M * N * sizeof(float), hipMemcpyDeviceToHost);

    dim3 block(16, 16);
    dim3 grid(N / (block.x * THREAD_TILE_X), M / (block.y * THREAD_TILE_Y));

    hipEventRecord(start);
    for(int i = 0; i < 100; i++){
        Sgemm_SIMT<<< grid, block >>> (dA, dB, dC, WIDTH);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    dt /= 100.0f;  
    printf("Average SIMT-sgemm kernel time over servel runs: %8.3f ms.\n", dt);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // hipblasSetMathMode(handle, HIPBLAS_XF32_XDL_MATH);
    // the flag 'HIPBLAS_XF32_XDL_MATH' would be faster a little, there is a numerical degradation yet. 
    // btw, only 'HIPBLAS_XF32_XDL_MATH' is useful with hipblasSetMathMode API

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < 3; ++i) {
        // hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        //              N, M, K, &alpha, dB, N, dA, K, &beta, dC_blas, N);
        // hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, N, M, K,
        //             &alpha, dB, HIPBLAS_R_32F, N, dA, HIPBLAS_R_32F, K,
        //             &beta, dC_blas, HIPBLAS_R_32F, N, HIPBLAS_R_32F,
        //             HIPBLAS_GEMM_DEFAULT);

        // hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
        //             &alpha, dB, HIPBLAS_R_32F, N, dA, HIPBLAS_R_32F, K,
        //             &beta, dC_blas, HIPBLAS_R_32F, N, HIPBLAS_R_32F,
        //             HIPBLAS_GEMM_DEFAULT);

        hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K,
                    &alpha, dA, HIPBLAS_R_32F, K, dB, HIPBLAS_R_32F, N,
                    &beta, dC_blas, HIPBLAS_R_32F, M, HIPBLAS_R_32F,
                    HIPBLAS_GEMM_DEFAULT);

        // hipblasGemmExWithFlags(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, WIDTH, WIDTH, WIDTH,
        //             &alpha, dB, HIPBLAS_R_32F, WIDTH, dA, HIPBLAS_R_32F, WIDTH,
        //             &beta, dC_blas, HIPBLAS_R_32F, WIDTH, HIPBLAS_R_32F,
        //             HIPBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_FLAGS_CHECK_SOLUTION_INDEX);
        // rocblasGemmEx(handle, ROCBLAS_OP_N, ROCBLAS_OP_N, WIDTH, WIDTH, WIDTH,
        //             &alpha, dB, ROCBLAS_R_32F, WIDTH, dA, ROCBLAS_R_32F, WIDTH,
        //             &beta, dC_blas, ROCBLAS_R_32F, WIDTH, ROCBLAS_R_32F,
        //             ROCBLAS_GEMM_DEFAULT);
    }
    hipDeviceSynchronize();

    hipEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        // hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        //              N, M, K, &alpha, dB, N, dA, K, &beta, dC_blas, N);

        // hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
        //             &alpha, dB, HIPBLAS_R_32F, N, dA, HIPBLAS_R_32F, K,
        //             &beta, dC_blas, HIPBLAS_R_32F, N, HIPBLAS_R_32F,
        //             HIPBLAS_GEMM_DEFAULT);

        hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K,
                    &alpha, dA, HIPBLAS_R_32F, K, dB, HIPBLAS_R_32F, N,
                    &beta, dC_blas, HIPBLAS_R_32F, M, HIPBLAS_R_32F,
                    HIPBLAS_GEMM_DEFAULT);

        // hipblasGemmExWithFlags(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, WIDTH, WIDTH, WIDTH,
        //             &alpha, dB, HIPBLAS_R_32F, WIDTH, dA, HIPBLAS_R_32F, WIDTH,
        //             &beta, dC_blas, HIPBLAS_R_32F, WIDTH, HIPBLAS_R_32F,
        //             HIPBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_FLAGS_CHECK_SOLUTION_INDEX);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&dt, start, stop);
    dt /= 100.0f;  
    printf("Average matrix multiplication time over servel runs: %8.3f ms.\n", dt);

    hipMemcpy(blas_C, dC_blas, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // for(int i = 0; i < 16; i++){
    //     printf("%f -- %f\n", C[i], blas_C[i]);
    // }

    verify_result(C, blas_C, M, N);

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
