#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "conv2d.h"

/*选手自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    float*   pin;                            //输入数据地址
    float*   pweight;                        //权值数据地址
    float*   pout;                           //输出数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
}mykernelParamType;                          

/* 
    +op : 将一个进程需要使用的输入矩阵数据和权重矩阵数据放在连续的位置，减少计算时的循环判断次数
   */
extern "C" __global__ void myKernelConv2dGpu(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,256)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    int y = by * b_size_y + ty;
    if(x>=Ox_w)
    {
        return;
    }
   
    int z = bz;
    
    int z1 = z;
    int z2 = z+n/8;
    int z3 = z+n/4;
    int z4 = z+(n/8)*3;
    int z5 = z+n/2;
    int z6 = z+(n/8)*5;
    int z7 = z+(n/8)*6;
    int z8 = z+(n/8)*7;

    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    float sum6 = 0.0;
    float sum7 = 0.0;
    float sum8 = 0.0;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[512];
    __shared__ float buf_pw[64];
    
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset3 = z3*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset4 = z4*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset5 = z5*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset6 = z6*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset7 = z7*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset8 = z8*c*inChannelOffset + posh_ori*w + posw_ori;

    int pw_offset = y*c*weightChannelOffset; 
    int out=k*Ox_w;
    int out1=y*Ox_w + x;
    int channel;
    int pin_pos = 0;
    int buf_pos = 0;
    //寄存器
    float reg_pw1;
    float reg_pw2;
    float reg_pw3;
    float reg_pw4;

    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty +64;
    int buf_ty2=buf_pin_offset + ty +128;
    int buf_ty3=buf_pin_offset + ty +192;
    int buf_ty4=buf_pin_offset + ty +256;
    int buf_ty5=buf_pin_offset + ty +320;
    int buf_ty6=buf_pin_offset + ty +384;
    int buf_ty7=buf_pin_offset + ty +448;
    int buf_tx=buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
        if(ty<weightChannelOffset)
        {
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_ty] = param.pin[pin_pos];
            
            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_ty1] = param.pin[pin_pos];

            pin_pos = pin_offset3 + pin_ty;
            buf_pin[buf_ty2] = param.pin[pin_pos];

            pin_pos = pin_offset4 + pin_ty;
            buf_pin[buf_ty3] = param.pin[pin_pos];

            pin_pos = pin_offset5 + pin_ty;
            buf_pin[buf_ty4] = param.pin[pin_pos];

            pin_pos = pin_offset6 + pin_ty;
            buf_pin[buf_ty5] = param.pin[pin_pos];

            pin_pos = pin_offset7 + pin_ty;
            buf_pin[buf_ty6] = param.pin[pin_pos];

            pin_pos = pin_offset8 + pin_ty;
            buf_pin[buf_ty7] = param.pin[pin_pos];
        }
        if(tx<weightChannelOffset)
        {
            buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        }
        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pin_offset3 += inChannelOffset;
        pin_offset4 += inChannelOffset;
        pin_offset5 += inChannelOffset;
        pin_offset6 += inChannelOffset;
        pin_offset7 += inChannelOffset;
        pin_offset8 += inChannelOffset;

        pw_offset   += weightChannelOffset;
        //寄存器读数
        reg_pw1= buf_pw[buf_pw_offset + 0];
        reg_pw2= buf_pw[buf_pw_offset + 1];
        reg_pw3= buf_pw[buf_pw_offset + 2];
        reg_pw4= buf_pw[buf_pw_offset + 3];
        
        sum1 += buf_pin[buf_pin_offset + 0] * reg_pw1;
        sum1 += buf_pin[buf_pin_offset + 1] * reg_pw2;
        sum1 += buf_pin[buf_pin_offset + 2] * reg_pw3;
        sum1 += buf_pin[buf_pin_offset + 3] * reg_pw4;
    
        sum2 += buf_pin[buf_pin_offset + 64] * reg_pw1;
        sum2 += buf_pin[buf_pin_offset + 65] * reg_pw2;
        sum2 += buf_pin[buf_pin_offset + 66] * reg_pw3;
        sum2 += buf_pin[buf_pin_offset + 67] * reg_pw4;

        sum3 += buf_pin[buf_pin_offset + 128] * reg_pw1;
        sum3 += buf_pin[buf_pin_offset + 129] * reg_pw2;
        sum3 += buf_pin[buf_pin_offset + 130] * reg_pw3;
        sum3 += buf_pin[buf_pin_offset + 131] * reg_pw4;

        sum4 += buf_pin[buf_pin_offset + 192] * reg_pw1;
        sum4 += buf_pin[buf_pin_offset + 193] * reg_pw2;
        sum4 += buf_pin[buf_pin_offset + 194] * reg_pw3;
        sum4 += buf_pin[buf_pin_offset + 195] * reg_pw4;

        sum5 += buf_pin[buf_pin_offset + 256] * reg_pw1;
        sum5 += buf_pin[buf_pin_offset + 257] * reg_pw2;
        sum5 += buf_pin[buf_pin_offset + 258] * reg_pw3;
        sum5 += buf_pin[buf_pin_offset + 259] * reg_pw4;

        sum6 += buf_pin[buf_pin_offset + 320] * reg_pw1;
        sum6 += buf_pin[buf_pin_offset + 321] * reg_pw2;
        sum6 += buf_pin[buf_pin_offset + 322] * reg_pw3;
        sum6 += buf_pin[buf_pin_offset + 323] * reg_pw4;

        sum7 += buf_pin[buf_pin_offset + 384] * reg_pw1;
        sum7 += buf_pin[buf_pin_offset + 385] * reg_pw2;
        sum7 += buf_pin[buf_pin_offset + 386] * reg_pw3;
        sum7 += buf_pin[buf_pin_offset + 387] * reg_pw4;

        sum8 += buf_pin[buf_pin_offset + 448] * reg_pw1;
        sum8 += buf_pin[buf_pin_offset + 449] * reg_pw2;
        sum8 += buf_pin[buf_pin_offset + 450] * reg_pw3;
        sum8 += buf_pin[buf_pin_offset + 451] * reg_pw4;
        __syncthreads();
    } 
    //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
    int outOffset3 = z3*out+out1;
    param.pout[outOffset3] = sum3;
    int outOffset4 = z4*out+out1;
    param.pout[outOffset4] = sum4;
    int outOffset5 = z5*out+out1;
    param.pout[outOffset5] = sum5;
    int outOffset6 = z6*out+out1;
    param.pout[outOffset6] = sum6;
    int outOffset7 = z7*out+out1;
    param.pout[outOffset7] = sum7;
    int outOffset8 = z8*out+out1;
    param.pout[outOffset8] = sum8;
}
extern "C" __global__ void myKernelConv2dGpu_1(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,1024)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int y = by * b_size_y + ty;
    int z = bz;
    
    int z1 = z;
    int z2 = z+n/2;

    float sum1 = 0.0;
    float sum2 = 0.0;

    float reg_pw1;
    float reg_pw2;
    float reg_pw3;
    float reg_pw4;
    float reg_pw5;
    float reg_pw6;
    float reg_pw7;

    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[3136];
    __shared__ float buf_pw[1568];
    
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;

    int pw_offset = y*c*weightChannelOffset;
    int out=k*Ox_w;
    int out1=y*Ox_w + x;
     
    int channel,i,j;
    int pin_pos = 0;
    int buf_pos = 0;
    float reg_pw=0.0;

    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty + 1568;
    int buf_tx=buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
       
        pin_pos = pin_offset1 + pin_ty;
        buf_pin[buf_ty] = param.pin[pin_pos];

        pin_pos = pin_offset2 + pin_ty;
        buf_pin[buf_ty1] = param.pin[pin_pos];

        if(ty<=16)
        {
            int a=ty+32;
            int pin_ty=(a/s)*w + a%s;
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_pin_offset + a] = param.pin[pin_pos];

            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_pin_offset + a + 1568] = param.pin[pin_pos];
        }
        
        buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        if(tx<=16)
        {
            buf_pw[buf_pw_offset + tx+32] = param.pweight[pw_offset + tx+32];
        }
    
        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pw_offset   += weightChannelOffset;
        
        for(i = 0; i < weightChannelOffset; ++i)
        {
            reg_pw=buf_pw[buf_pw_offset + i];
            sum1 += buf_pin[buf_pin_offset + i] * reg_pw;
            sum2 += buf_pin[buf_pin_offset + i + 1568] * reg_pw;
        }
        __syncthreads();
    } 
    //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
}
extern "C" __global__ void myKernelConv2dGpu_2(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,1024)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int y = by * b_size_y + ty;
    int z = bz;
    
    int z1 = z;
    int z2 = z+n/8;
    int z3 = z+n/4;
    int z4 = z+(n/8)*3;
    int z5 = z+n/2;
    int z6 = z+(n/8)*5;
    int z7 = z+(n/8)*6;
    int z8 = z+(n/8)*7;

    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    float sum6 = 0.0;
    float sum7 = 0.0;
    float sum8 = 0.0;


    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[2304];
    __shared__ float buf_pw[288];
    
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset3 = z3*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset4 = z4*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset5 = z5*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset6 = z6*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset7 = z7*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset8 = z8*c*inChannelOffset + posh_ori*w + posw_ori;
    
    int pw_offset = y*c*weightChannelOffset;
    int out=k*Ox_w;
    int out1=y*Ox_w + x; 
     
    int channel;
    int pin_pos = 0;
    

    //寄存器
    float reg_pw1;
    float reg_pw2;
    float reg_pw3;
    float reg_pw4;
    float reg_pw5;
    float reg_pw6;
    float reg_pw7;
    float reg_pw8;
    float reg_pw9;

    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty+ 288;
    int buf_ty2=buf_pin_offset + ty+ 576;
    int buf_ty3=buf_pin_offset + ty+ 864;
    int buf_ty4=buf_pin_offset + ty+ 1152;
    int buf_ty5=buf_pin_offset + ty+ 1440;
    int buf_ty6=buf_pin_offset + ty+ 1728;
    int buf_ty7=buf_pin_offset + ty+ 2016;
    int buf_tx=buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
        if(ty<weightChannelOffset)
        {
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_ty] = param.pin[pin_pos];
            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_ty1] = param.pin[pin_pos];
            pin_pos = pin_offset3 + pin_ty;
            buf_pin[buf_ty2] = param.pin[pin_pos];
            pin_pos = pin_offset4 + pin_ty;
            buf_pin[buf_ty3] = param.pin[pin_pos];
            pin_pos = pin_offset5 + pin_ty;
            buf_pin[buf_ty4] = param.pin[pin_pos];
            pin_pos = pin_offset6 + pin_ty;
            buf_pin[buf_ty5] = param.pin[pin_pos];
            pin_pos = pin_offset7 + pin_ty;
            buf_pin[buf_ty6] = param.pin[pin_pos];
            pin_pos = pin_offset8 + pin_ty;
            buf_pin[buf_ty7] = param.pin[pin_pos];
        }
        if(tx<weightChannelOffset)
        {
            buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        }
        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pin_offset3 += inChannelOffset;
        pin_offset4 += inChannelOffset;
        pin_offset5 += inChannelOffset;
        pin_offset6 += inChannelOffset;
        pin_offset7 += inChannelOffset;
        pin_offset8 += inChannelOffset;

        pw_offset   += weightChannelOffset;
        
        reg_pw1= buf_pw[buf_pw_offset + 0];
        reg_pw2= buf_pw[buf_pw_offset + 1];
        reg_pw3= buf_pw[buf_pw_offset + 2];
        reg_pw4= buf_pw[buf_pw_offset + 3];
        reg_pw5= buf_pw[buf_pw_offset + 4];
        reg_pw6= buf_pw[buf_pw_offset + 5];
        reg_pw7= buf_pw[buf_pw_offset + 6];
        reg_pw8= buf_pw[buf_pw_offset + 7];
        reg_pw9= buf_pw[buf_pw_offset + 8];

        sum1 += buf_pin[buf_pin_offset + 0] * reg_pw1;
        sum1 += buf_pin[buf_pin_offset + 1] * reg_pw2;
        sum1 += buf_pin[buf_pin_offset + 2] * reg_pw3;
        sum1 += buf_pin[buf_pin_offset + 3] * reg_pw4;
        sum1 += buf_pin[buf_pin_offset + 4] * reg_pw5;
        sum1 += buf_pin[buf_pin_offset + 5] * reg_pw6;
        sum1 += buf_pin[buf_pin_offset + 6] * reg_pw7;
        sum1 += buf_pin[buf_pin_offset + 7] * reg_pw8;
        sum1 += buf_pin[buf_pin_offset + 8] * reg_pw9;

        sum2 += buf_pin[buf_pin_offset + 288] * reg_pw1;
        sum2 += buf_pin[buf_pin_offset + 289] * reg_pw2;
        sum2 += buf_pin[buf_pin_offset + 290] * reg_pw3;
        sum2 += buf_pin[buf_pin_offset + 291] * reg_pw4;
        sum2 += buf_pin[buf_pin_offset + 292] * reg_pw5;
        sum2 += buf_pin[buf_pin_offset + 293] * reg_pw6;
        sum2 += buf_pin[buf_pin_offset + 294] * reg_pw7;
        sum2 += buf_pin[buf_pin_offset + 295] * reg_pw8;
        sum2 += buf_pin[buf_pin_offset + 296] * reg_pw9;

        sum3 += buf_pin[buf_pin_offset + 576] * reg_pw1;
        sum3 += buf_pin[buf_pin_offset + 577] * reg_pw2;
        sum3 += buf_pin[buf_pin_offset + 578] * reg_pw3;
        sum3 += buf_pin[buf_pin_offset + 579] * reg_pw4;
        sum3 += buf_pin[buf_pin_offset + 580] * reg_pw5;
        sum3 += buf_pin[buf_pin_offset + 581] * reg_pw6;
        sum3 += buf_pin[buf_pin_offset + 582] * reg_pw7;
        sum3 += buf_pin[buf_pin_offset + 583] * reg_pw8;
        sum3 += buf_pin[buf_pin_offset + 584] * reg_pw9;

        sum4 += buf_pin[buf_pin_offset + 864] * reg_pw1;
        sum4 += buf_pin[buf_pin_offset + 865] * reg_pw2;
        sum4 += buf_pin[buf_pin_offset + 866] * reg_pw3;
        sum4 += buf_pin[buf_pin_offset + 867] * reg_pw4;
        sum4 += buf_pin[buf_pin_offset + 868] * reg_pw5;
        sum4 += buf_pin[buf_pin_offset + 869] * reg_pw6;
        sum4 += buf_pin[buf_pin_offset + 870] * reg_pw7;
        sum4 += buf_pin[buf_pin_offset + 871] * reg_pw8;
        sum4 += buf_pin[buf_pin_offset + 872] * reg_pw9;

        sum5 += buf_pin[buf_pin_offset + 1152] * reg_pw1;
        sum5 += buf_pin[buf_pin_offset + 1153] * reg_pw2;
        sum5 += buf_pin[buf_pin_offset + 1154] * reg_pw3;
        sum5 += buf_pin[buf_pin_offset + 1155] * reg_pw4;
        sum5 += buf_pin[buf_pin_offset + 1156] * reg_pw5;
        sum5 += buf_pin[buf_pin_offset + 1157] * reg_pw6;
        sum5 += buf_pin[buf_pin_offset + 1158] * reg_pw7;
        sum5 += buf_pin[buf_pin_offset + 1159] * reg_pw8;
        sum5 += buf_pin[buf_pin_offset + 1160] * reg_pw9;

        sum6 += buf_pin[buf_pin_offset + 1440] * reg_pw1;
        sum6 += buf_pin[buf_pin_offset + 1441] * reg_pw2;
        sum6 += buf_pin[buf_pin_offset + 1442] * reg_pw3;
        sum6 += buf_pin[buf_pin_offset + 1443] * reg_pw4;
        sum6 += buf_pin[buf_pin_offset + 1444] * reg_pw5;
        sum6 += buf_pin[buf_pin_offset + 1445] * reg_pw6;
        sum6 += buf_pin[buf_pin_offset + 1446] * reg_pw7;
        sum6 += buf_pin[buf_pin_offset + 1447] * reg_pw8;
        sum6 += buf_pin[buf_pin_offset + 1448] * reg_pw9;

        sum7 += buf_pin[buf_pin_offset + 1728] * reg_pw1;
        sum7 += buf_pin[buf_pin_offset + 1729] * reg_pw2;
        sum7 += buf_pin[buf_pin_offset + 1730] * reg_pw3;
        sum7 += buf_pin[buf_pin_offset + 1731] * reg_pw4;
        sum7 += buf_pin[buf_pin_offset + 1732] * reg_pw5;
        sum7 += buf_pin[buf_pin_offset + 1733] * reg_pw6;
        sum7 += buf_pin[buf_pin_offset + 1734] * reg_pw7;
        sum7 += buf_pin[buf_pin_offset + 1735] * reg_pw8;
        sum7 += buf_pin[buf_pin_offset + 1736] * reg_pw9;

        sum8 += buf_pin[buf_pin_offset + 2016] * reg_pw1;
        sum8 += buf_pin[buf_pin_offset + 2017] * reg_pw2;
        sum8 += buf_pin[buf_pin_offset + 2018] * reg_pw3;
        sum8 += buf_pin[buf_pin_offset + 2019] * reg_pw4;
        sum8 += buf_pin[buf_pin_offset + 2020] * reg_pw5;
        sum8 += buf_pin[buf_pin_offset + 2021] * reg_pw6;
        sum8 += buf_pin[buf_pin_offset + 2022] * reg_pw7;
        sum8 += buf_pin[buf_pin_offset + 2023] * reg_pw8;
        sum8 += buf_pin[buf_pin_offset + 2024] * reg_pw9;
        __syncthreads();
    } 
    //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
    int outOffset3 = z3*out+out1;
    param.pout[outOffset3] = sum3;
    int outOffset4 = z4*out+out1;
    param.pout[outOffset4] = sum4;
    int outOffset5 = z5*out+out1;
    param.pout[outOffset5] = sum5;
    int outOffset6 = z6*out+out1;
    param.pout[outOffset6] = sum6;
    int outOffset7 = z7*out+out1;
    param.pout[outOffset7] = sum7;
    int outOffset8 = z8*out+out1;
    param.pout[outOffset8] = sum8;
}
extern "C" __global__ void myKernelConv2dGpu_3(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,816)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int y = by * b_size_y + ty;
    int z = bz;
    
    int z1 = z;
    int z2 = z+(n/7);
    int z3 = z+(n/7)*2;
    int z4 = z+(n/7)*3;
    int z5 = z+(n/7)*4;
    int z6 = z+(n/7)*5;
    int z7 = z+(n/7)*6;

    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    float sum6 = 0.0;
    float sum7 = 0.0;


    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[1071];
    __shared__ float buf_pw[432];
    
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset3 = z3*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset4 = z4*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset5 = z5*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset6 = z6*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset7 = z7*c*inChannelOffset + posh_ori*w + posw_ori;

    int pw_offset = y*c*weightChannelOffset;
    int out=k*Ox_w; 
    int out1=y*Ox_w + x;
     
     
    int channel,i,j;
    int pin_pos = 0;
    

    //寄存器
    float reg_pw1;
    float reg_pw2;
    float reg_pw3;
    float reg_pw4;
    float reg_pw5;
    float reg_pw6;
    float reg_pw7;
    float reg_pw8;
    float reg_pw9;

    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty+ 153;
    int buf_ty2=buf_pin_offset + ty+ 306;
    int buf_ty3=buf_pin_offset + ty+ 459;
    int buf_ty4=buf_pin_offset + ty+ 612;
    int buf_ty5=buf_pin_offset + ty+ 765;
    int buf_ty6=buf_pin_offset + ty+ 918;
    int buf_tx =buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
        if(ty<weightChannelOffset)
        {
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_ty] = param.pin[pin_pos];
            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_ty1] = param.pin[pin_pos];
            pin_pos = pin_offset3 + pin_ty;
            buf_pin[buf_ty2] = param.pin[pin_pos];
            pin_pos = pin_offset4 + pin_ty;
            buf_pin[buf_ty3] = param.pin[pin_pos];
            pin_pos = pin_offset5 + pin_ty;
            buf_pin[buf_ty4] = param.pin[pin_pos];
            pin_pos = pin_offset6 + pin_ty;
            buf_pin[buf_ty5] = param.pin[pin_pos];
            pin_pos = pin_offset7 + pin_ty;
            buf_pin[buf_ty6] = param.pin[pin_pos];
        }
        if(tx<weightChannelOffset)
        {
            buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        }
        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pin_offset3 += inChannelOffset;
        pin_offset4 += inChannelOffset;
        pin_offset5 += inChannelOffset;
        pin_offset6 += inChannelOffset;
        pin_offset7 += inChannelOffset;

        pw_offset   += weightChannelOffset;
        
        reg_pw1= buf_pw[buf_pw_offset + 0];
        reg_pw2= buf_pw[buf_pw_offset + 1];
        reg_pw3= buf_pw[buf_pw_offset + 2];
        reg_pw4= buf_pw[buf_pw_offset + 3];
        reg_pw5= buf_pw[buf_pw_offset + 4];
        reg_pw6= buf_pw[buf_pw_offset + 5];
        reg_pw7= buf_pw[buf_pw_offset + 6];
        reg_pw8= buf_pw[buf_pw_offset + 7];
        reg_pw9= buf_pw[buf_pw_offset + 8];

        sum1 += buf_pin[buf_pin_offset + 0] * reg_pw1;
        sum1 += buf_pin[buf_pin_offset + 1] * reg_pw2;
        sum1 += buf_pin[buf_pin_offset + 2] * reg_pw3;
        sum1 += buf_pin[buf_pin_offset + 3] * reg_pw4;
        sum1 += buf_pin[buf_pin_offset + 4] * reg_pw5;
        sum1 += buf_pin[buf_pin_offset + 5] * reg_pw6;
        sum1 += buf_pin[buf_pin_offset + 6] * reg_pw7;
        sum1 += buf_pin[buf_pin_offset + 7] * reg_pw8;
        sum1 += buf_pin[buf_pin_offset + 8] * reg_pw9;

        sum2 += buf_pin[buf_pin_offset + 153] * reg_pw1;
        sum2 += buf_pin[buf_pin_offset + 154] * reg_pw2;
        sum2 += buf_pin[buf_pin_offset + 155] * reg_pw3;
        sum2 += buf_pin[buf_pin_offset + 156] * reg_pw4;
        sum2 += buf_pin[buf_pin_offset + 157] * reg_pw5;
        sum2 += buf_pin[buf_pin_offset + 158] * reg_pw6;
        sum2 += buf_pin[buf_pin_offset + 159] * reg_pw7;
        sum2 += buf_pin[buf_pin_offset + 160] * reg_pw8;
        sum2 += buf_pin[buf_pin_offset + 161] * reg_pw9;

        sum3 += buf_pin[buf_pin_offset + 306] * reg_pw1;
        sum3 += buf_pin[buf_pin_offset + 307] * reg_pw2;
        sum3 += buf_pin[buf_pin_offset + 308] * reg_pw3;
        sum3 += buf_pin[buf_pin_offset + 309] * reg_pw4;
        sum3 += buf_pin[buf_pin_offset + 310] * reg_pw5;
        sum3 += buf_pin[buf_pin_offset + 311] * reg_pw6;
        sum3 += buf_pin[buf_pin_offset + 312] * reg_pw7;
        sum3 += buf_pin[buf_pin_offset + 313] * reg_pw8;
        sum3 += buf_pin[buf_pin_offset + 314] * reg_pw9;

        sum4 += buf_pin[buf_pin_offset + 459] * reg_pw1;
        sum4 += buf_pin[buf_pin_offset + 460] * reg_pw2;
        sum4 += buf_pin[buf_pin_offset + 461] * reg_pw3;
        sum4 += buf_pin[buf_pin_offset + 462] * reg_pw4;
        sum4 += buf_pin[buf_pin_offset + 463] * reg_pw5;
        sum4 += buf_pin[buf_pin_offset + 464] * reg_pw6;
        sum4 += buf_pin[buf_pin_offset + 465] * reg_pw7;
        sum4 += buf_pin[buf_pin_offset + 466] * reg_pw8;
        sum4 += buf_pin[buf_pin_offset + 467] * reg_pw9;

        sum5 += buf_pin[buf_pin_offset + 612] * reg_pw1;
        sum5 += buf_pin[buf_pin_offset + 613] * reg_pw2;
        sum5 += buf_pin[buf_pin_offset + 614] * reg_pw3;
        sum5 += buf_pin[buf_pin_offset + 615] * reg_pw4;
        sum5 += buf_pin[buf_pin_offset + 616] * reg_pw5;
        sum5 += buf_pin[buf_pin_offset + 617] * reg_pw6;
        sum5 += buf_pin[buf_pin_offset + 618] * reg_pw7;
        sum5 += buf_pin[buf_pin_offset + 619] * reg_pw8;
        sum5 += buf_pin[buf_pin_offset + 620] * reg_pw9;

        sum6 += buf_pin[buf_pin_offset + 765] * reg_pw1;
        sum6 += buf_pin[buf_pin_offset + 766] * reg_pw2;
        sum6 += buf_pin[buf_pin_offset + 767] * reg_pw3;
        sum6 += buf_pin[buf_pin_offset + 768] * reg_pw4;
        sum6 += buf_pin[buf_pin_offset + 769] * reg_pw5;
        sum6 += buf_pin[buf_pin_offset + 770] * reg_pw6;
        sum6 += buf_pin[buf_pin_offset + 771] * reg_pw7;
        sum6 += buf_pin[buf_pin_offset + 772] * reg_pw8;
        sum6 += buf_pin[buf_pin_offset + 773] * reg_pw9;

        sum7 += buf_pin[buf_pin_offset + 918] * reg_pw1;
        sum7 += buf_pin[buf_pin_offset + 919] * reg_pw2;
        sum7 += buf_pin[buf_pin_offset + 920] * reg_pw3;
        sum7 += buf_pin[buf_pin_offset + 921] * reg_pw4;
        sum7 += buf_pin[buf_pin_offset + 922] * reg_pw5;
        sum7 += buf_pin[buf_pin_offset + 923] * reg_pw6;
        sum7 += buf_pin[buf_pin_offset + 924] * reg_pw7;
        sum7 += buf_pin[buf_pin_offset + 925] * reg_pw8;
        sum7 += buf_pin[buf_pin_offset + 926] * reg_pw9;
        __syncthreads();
    } 
     //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
    int outOffset3 = z3*out+out1;
    param.pout[outOffset3] = sum3;
    int outOffset4 = z4*out+out1;
    param.pout[outOffset4] = sum4;
    int outOffset5 = z5*out+out1;
    param.pout[outOffset5] = sum5;
    int outOffset6 = z6*out+out1;
    param.pout[outOffset6] = sum6;
    int outOffset7 = z7*out+out1;
    param.pout[outOffset7] = sum7;
}
extern "C" __global__ void myKernelConv2dGpu_4(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,1024)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int y = by * b_size_y + ty;
    int z = bz;
    
    int z1 = z;
    int z2 = z+n/4;
    int z3 = z+n/2;
    int z4 = z+(n/4)*3;

    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;


    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    
    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[6272];
    __shared__ float buf_pw[1568];
    
    float reg_pw=0.0;
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset3 = z3*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset4 = z4*c*inChannelOffset + posh_ori*w + posw_ori;

    int pw_offset = y*c*weightChannelOffset;
    int out=k*Ox_w;
    int out1=y*Ox_w + x;  
     
    int channel,i,j;
    int pin_pos = 0;
    int buf_pos = 0;
 
    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty+ 1568;
    int buf_ty2=buf_pin_offset + ty+ 3136;
    int buf_ty3=buf_pin_offset + ty+ 4704;
    int buf_tx=buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
        
        pin_pos = pin_offset1 + pin_ty;
        buf_pin[buf_ty] = param.pin[pin_pos];

        pin_pos = pin_offset2 + pin_ty;
        buf_pin[buf_ty1] = param.pin[pin_pos];

        pin_pos = pin_offset3 + pin_ty;
        buf_pin[buf_ty2] = param.pin[pin_pos];

        pin_pos = pin_offset4 + pin_ty;
        buf_pin[buf_ty3] = param.pin[pin_pos];
        
        if(ty<=16)
        {
            int a=ty+32;
            int pin_ty=(a/s)*w + a%s;
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_pin_offset + a] = param.pin[pin_pos];

            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_pin_offset + a + 1568] = param.pin[pin_pos];

            pin_pos = pin_offset3 + pin_ty;
            buf_pin[buf_pin_offset + a + 3136] = param.pin[pin_pos];

            pin_pos = pin_offset4 + pin_ty;
            buf_pin[buf_pin_offset + a + 4704] = param.pin[pin_pos];
        }
        buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        if(tx<=16)
        {
            int a=tx+32;
            buf_pw[buf_pw_offset + a] = param.pweight[pw_offset + a];
        }

        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pin_offset3 += inChannelOffset;
        pin_offset4 += inChannelOffset;
        pw_offset   += weightChannelOffset;
        
        for(i = 0; i < weightChannelOffset; ++i)
        {
            
            reg_pw= buf_pw[buf_pw_offset + i];
            sum1 += buf_pin[buf_pin_offset + i] * reg_pw;
            sum2 += buf_pin[buf_pin_offset + i + 1568] * reg_pw;
            sum3 += buf_pin[buf_pin_offset + i + 3136] * reg_pw;
            sum4 += buf_pin[buf_pin_offset + i + 4704] * reg_pw;
        }
        __syncthreads();
    } 
    //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
    int outOffset3 = z3*out+out1;
    param.pout[outOffset3] = sum3;
    int outOffset4 = z4*out+out1;
    param.pout[outOffset4] = sum4;
}
extern "C" __global__ void myKernelConv2dGpu_5(mykernelParamType param) __attribute__((amdgpu_flat_work_group_size(1,1024)))
{
    int n = param.n;
    int k = param.k;
    int r = param.r;
    int s = param.s;

    int h = param.h;
    int w = param.w;

    int c = param.c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b_size_x = blockDim.x;
    int b_size_y = blockDim.y;

    int x = bx * b_size_x + tx;
    int Oh = param.Oh;
    int Ow = param.Ow;

    int Ox_w = Oh*Ow;
    int y = by * b_size_y + ty;
    int z = bz;
    
    int z1 = z;
    int z2 = z+n/8;
    int z3 = z+n/4;
    int z4 = z+(n/8)*3;
    int z5 = z+n/2;
    int z6 = z+(n/8)*5;
    int z7 = z+(n/8)*6;
    int z8 = z+(n/8)*7;

    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    float sum5 = 0.0;
    float sum6 = 0.0;
    float sum7 = 0.0;
    float sum8 = 0.0;


    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/Ow;
    int posOw = x%Ow;
        
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    int inChannelOffset = h*w;
    int weightChannelOffset = r*s;
    
    // 申请两个LDS
   
    __shared__ float buf_pin[1152];
    __shared__ float buf_pw[576];
    
    
    // 每个线程将输入矩阵的数据放到 r*s 的连续数据块中
    int buf_pin_offset = tx*weightChannelOffset;
    int buf_pw_offset = ty*weightChannelOffset;

    int pin_offset1 = z1*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset2 = z2*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset3 = z3*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset4 = z4*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset5 = z5*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset6 = z6*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset7 = z7*c*inChannelOffset + posh_ori*w + posw_ori;
    int pin_offset8 = z8*c*inChannelOffset + posh_ori*w + posw_ori;
    
    int pw_offset = y*c*weightChannelOffset; 
    int out=k*Ox_w;
    int out1=y*Ox_w + x; 
     
    int channel;
    int pin_pos = 0;
    

    //寄存器
    float reg_pw1;
    float reg_pw2;
    float reg_pw3;
    float reg_pw4;
    float reg_pw5;
    float reg_pw6;
    float reg_pw7;
    float reg_pw8;
    float reg_pw9;

    int pin_ty=(ty/s)*w + ty%s;
    int buf_ty=buf_pin_offset + ty;
    int buf_ty1=buf_pin_offset + ty+ 144;
    int buf_ty2=buf_pin_offset + ty+ 288;
    int buf_ty3=buf_pin_offset + ty+ 432;
    int buf_ty4=buf_pin_offset + ty+ 576;
    int buf_ty5=buf_pin_offset + ty+ 720;
    int buf_ty6=buf_pin_offset + ty+ 864;
    int buf_ty7=buf_pin_offset + ty+ 1008;
    int buf_tx=buf_pw_offset + tx;
    for(channel = 0; channel<c; ++channel)
    {
        if(ty<weightChannelOffset)
        {
            pin_pos = pin_offset1 + pin_ty;
            buf_pin[buf_ty] = param.pin[pin_pos];
            pin_pos = pin_offset2 + pin_ty;
            buf_pin[buf_ty1] = param.pin[pin_pos];
            pin_pos = pin_offset3 + pin_ty;
            buf_pin[buf_ty2] = param.pin[pin_pos];
            pin_pos = pin_offset4 + pin_ty;
            buf_pin[buf_ty3] = param.pin[pin_pos];
            pin_pos = pin_offset5 + pin_ty;
            buf_pin[buf_ty4] = param.pin[pin_pos];
            pin_pos = pin_offset6 + pin_ty;
            buf_pin[buf_ty5] = param.pin[pin_pos];
            pin_pos = pin_offset7 + pin_ty;
            buf_pin[buf_ty6] = param.pin[pin_pos];
            pin_pos = pin_offset8 + pin_ty;
            buf_pin[buf_ty7] = param.pin[pin_pos];
        }
        if(tx<weightChannelOffset)
        {
            buf_pw[buf_tx] = param.pweight[pw_offset + tx];
        }
        __syncthreads(); 
        pin_offset1 += inChannelOffset;
        pin_offset2 += inChannelOffset;
        pin_offset3 += inChannelOffset;
        pin_offset4 += inChannelOffset;
        pin_offset5 += inChannelOffset;
        pin_offset6 += inChannelOffset;
        pin_offset7 += inChannelOffset;
        pin_offset8 += inChannelOffset;

        pw_offset   += weightChannelOffset;
        
        reg_pw1= buf_pw[buf_pw_offset + 0];
        reg_pw2= buf_pw[buf_pw_offset + 1];
        reg_pw3= buf_pw[buf_pw_offset + 2];
        reg_pw4= buf_pw[buf_pw_offset + 3];
        reg_pw5= buf_pw[buf_pw_offset + 4];
        reg_pw6= buf_pw[buf_pw_offset + 5];
        reg_pw7= buf_pw[buf_pw_offset + 6];
        reg_pw8= buf_pw[buf_pw_offset + 7];
        reg_pw9= buf_pw[buf_pw_offset + 8];

        sum1 += buf_pin[buf_pin_offset + 0] * reg_pw1;
        sum1 += buf_pin[buf_pin_offset + 1] * reg_pw2;
        sum1 += buf_pin[buf_pin_offset + 2] * reg_pw3;
        sum1 += buf_pin[buf_pin_offset + 3] * reg_pw4;
        sum1 += buf_pin[buf_pin_offset + 4] * reg_pw5;
        sum1 += buf_pin[buf_pin_offset + 5] * reg_pw6;
        sum1 += buf_pin[buf_pin_offset + 6] * reg_pw7;
        sum1 += buf_pin[buf_pin_offset + 7] * reg_pw8;
        sum1 += buf_pin[buf_pin_offset + 8] * reg_pw9;

        sum2 += buf_pin[buf_pin_offset + 144] * reg_pw1;
        sum2 += buf_pin[buf_pin_offset + 145] * reg_pw2;
        sum2 += buf_pin[buf_pin_offset + 146] * reg_pw3;
        sum2 += buf_pin[buf_pin_offset + 147] * reg_pw4;
        sum2 += buf_pin[buf_pin_offset + 148] * reg_pw5;
        sum2 += buf_pin[buf_pin_offset + 149] * reg_pw6;
        sum2 += buf_pin[buf_pin_offset + 150] * reg_pw7;
        sum2 += buf_pin[buf_pin_offset + 151] * reg_pw8;
        sum2 += buf_pin[buf_pin_offset + 152] * reg_pw9;

        sum3 += buf_pin[buf_pin_offset + 288] * reg_pw1;
        sum3 += buf_pin[buf_pin_offset + 289] * reg_pw2;
        sum3 += buf_pin[buf_pin_offset + 290] * reg_pw3;
        sum3 += buf_pin[buf_pin_offset + 291] * reg_pw4;
        sum3 += buf_pin[buf_pin_offset + 292] * reg_pw5;
        sum3 += buf_pin[buf_pin_offset + 293] * reg_pw6;
        sum3 += buf_pin[buf_pin_offset + 294] * reg_pw7;
        sum3 += buf_pin[buf_pin_offset + 295] * reg_pw8;
        sum3 += buf_pin[buf_pin_offset + 296] * reg_pw9;

        sum4 += buf_pin[buf_pin_offset + 432] * reg_pw1;
        sum4 += buf_pin[buf_pin_offset + 433] * reg_pw2;
        sum4 += buf_pin[buf_pin_offset + 434] * reg_pw3;
        sum4 += buf_pin[buf_pin_offset + 435] * reg_pw4;
        sum4 += buf_pin[buf_pin_offset + 436] * reg_pw5;
        sum4 += buf_pin[buf_pin_offset + 437] * reg_pw6;
        sum4 += buf_pin[buf_pin_offset + 438] * reg_pw7;
        sum4 += buf_pin[buf_pin_offset + 439] * reg_pw8;
        sum4 += buf_pin[buf_pin_offset + 440] * reg_pw9;

        sum5 += buf_pin[buf_pin_offset + 576] * reg_pw1;
        sum5 += buf_pin[buf_pin_offset + 577] * reg_pw2;
        sum5 += buf_pin[buf_pin_offset + 578] * reg_pw3;
        sum5 += buf_pin[buf_pin_offset + 579] * reg_pw4;
        sum5 += buf_pin[buf_pin_offset + 580] * reg_pw5;
        sum5 += buf_pin[buf_pin_offset + 581] * reg_pw6;
        sum5 += buf_pin[buf_pin_offset + 582] * reg_pw7;
        sum5 += buf_pin[buf_pin_offset + 583] * reg_pw8;
        sum5 += buf_pin[buf_pin_offset + 584] * reg_pw9;

        sum6 += buf_pin[buf_pin_offset + 720] * reg_pw1;
        sum6 += buf_pin[buf_pin_offset + 721] * reg_pw2;
        sum6 += buf_pin[buf_pin_offset + 722] * reg_pw3;
        sum6 += buf_pin[buf_pin_offset + 723] * reg_pw4;
        sum6 += buf_pin[buf_pin_offset + 724] * reg_pw5;
        sum6 += buf_pin[buf_pin_offset + 725] * reg_pw6;
        sum6 += buf_pin[buf_pin_offset + 726] * reg_pw7;
        sum6 += buf_pin[buf_pin_offset + 727] * reg_pw8;
        sum6 += buf_pin[buf_pin_offset + 728] * reg_pw9;

        sum7 += buf_pin[buf_pin_offset + 864] * reg_pw1;
        sum7 += buf_pin[buf_pin_offset + 865] * reg_pw2;
        sum7 += buf_pin[buf_pin_offset + 866] * reg_pw3;
        sum7 += buf_pin[buf_pin_offset + 867] * reg_pw4;
        sum7 += buf_pin[buf_pin_offset + 868] * reg_pw5;
        sum7 += buf_pin[buf_pin_offset + 869] * reg_pw6;
        sum7 += buf_pin[buf_pin_offset + 870] * reg_pw7;
        sum7 += buf_pin[buf_pin_offset + 871] * reg_pw8;
        sum7 += buf_pin[buf_pin_offset + 872] * reg_pw9;

        sum8 += buf_pin[buf_pin_offset + 1008] * reg_pw1;
        sum8 += buf_pin[buf_pin_offset + 1009] * reg_pw2;
        sum8 += buf_pin[buf_pin_offset + 1010] * reg_pw3;
        sum8 += buf_pin[buf_pin_offset + 1011] * reg_pw4;
        sum8 += buf_pin[buf_pin_offset + 1012] * reg_pw5;
        sum8 += buf_pin[buf_pin_offset + 1013] * reg_pw6;
        sum8 += buf_pin[buf_pin_offset + 1014] * reg_pw7;
        sum8 += buf_pin[buf_pin_offset + 1015] * reg_pw8;
        sum8 += buf_pin[buf_pin_offset + 1016] * reg_pw9;
        __syncthreads();
    } 
    //计算输出偏移
    int outOffset1 = z1*out+out1;
    param.pout[outOffset1] = sum1;
    int outOffset2 = z2*out+out1;
    param.pout[outOffset2] = sum2;
    int outOffset3 = z3*out+out1;
    param.pout[outOffset3] = sum3;
    int outOffset4 = z4*out+out1;
    param.pout[outOffset4] = sum4;
    int outOffset5 = z5*out+out1;
    param.pout[outOffset5] = sum5;
    int outOffset6 = z6*out+out1;
    param.pout[outOffset6] = sum6;
    int outOffset7 = z7*out+out1;
    param.pout[outOffset7] = sum7;
    int outOffset8 = z8*out+out1;
    param.pout[outOffset8] = sum8;
}
/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}

/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    if(r==2)
    {
        kernelInfo->blockx   = (outh*outw + 15)/16;                    //blockx  number
        kernelInfo->blocky   = (k+15)/16;                    //blocky  number
        kernelInfo->blockz   = n/8;                    //blockz  number
        kernelInfo->threadx  = 16;                   //threadx number per block
        kernelInfo->thready  = 16;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu;                 //kernel ptr
        
    }
    else if(r==7&&n==2)
    {  
        kernelInfo->blockx   = (outh*outw + 31)/32;                    //blockx  number
        kernelInfo->blocky   = (k+31)/32;                    //blocky  number
        kernelInfo->blockz   = n/2;                    //blockz  number
        kernelInfo->threadx  = 32;                   //threadx number per block
        kernelInfo->thready  = 32;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu_1;                 //kernel ptr
    }
    else if(r==3&&n==128)
    {
        kernelInfo->blockx   = (outh*outw + 31)/32;                    //blockx  number
        kernelInfo->blocky   = (k+31)/32;                    //blocky  number
        kernelInfo->blockz   = n/8;                    //blockz  number
        kernelInfo->threadx  = 32;                   //threadx number per block
        kernelInfo->thready  = 32;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu_2;                 //kernel ptr
    } 
    else if(r==3&&n==49)
    {
        kernelInfo->blockx   = outh*outw/17;                    //blockx  number
        kernelInfo->blocky   = k/48;                    //blocky  number
        kernelInfo->blockz   = n/7;                    //blockz  number
        kernelInfo->threadx  = 17;                   //threadx number per block
        kernelInfo->thready  = 48;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu_3;                 //kernel ptr
    }
    else if(r==7&&n==128)
    {  
        kernelInfo->blockx   = (outh*outw + 31)/32;                    //blockx  number
        kernelInfo->blocky   = (k+31)/32;                    //blocky  number
        kernelInfo->blockz   = n/4;                    //blockz  number
        kernelInfo->threadx  = 32;                   //threadx number per block
        kernelInfo->thready  = 32;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu_4;                 //kernel ptr
    }
    else
    {
        kernelInfo->blockx   = (outh*outw + 15)/16;                    //blockx  number
        kernelInfo->blocky   = (k+63)/64;                    //blocky  number
        kernelInfo->blockz   = n/8;                    //blockz  number
        kernelInfo->threadx  = 16;                   //threadx number per block
        kernelInfo->thready  = 64;                   //thready number per block
        kernelInfo->threadz  = 1;                   //threadz number per block
        kernelInfo->dynmicLdsSize = 0;
        kernelInfo->kernelPtr= (void*)myKernelConv2dGpu_5;                 //kernel ptr
    }
    
    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie              default value 1
    pArgs->c = c;                              //channel number          default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw;       

    return 0;
}