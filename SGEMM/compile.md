`rocm-smi` and `hy-smi --showproductname` for checking the specific hardware structure of DCU.

`hipcc --offload-arch=gfx928 ./MCv1.cpp -o ./MCv1` for compiles the source file directly.

For different computation shape of matrix A, B, C in C = A x B, change the variables M, N, K directly in source file and recompile.