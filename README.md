# BALS: A Blocked Alternating Least Squares Algorithm for Parallel Matrix Factorization

Install: MAGMA must be installed first, and $MAGMA_DIR set to wherever it is compiled or installed.

CUDA Kernel: magma_sals.cu

The code shared in github is a example in updating X over Y matrix about ALS algorithm.

Modify the number of XBLOCK and YBLOCK in Makefile.
For example, xblock=256, yblock=512, batch must be the integral number of xblock.
Run: ./testing_sals -f 32 --batch 1024 --first ***.mtx
