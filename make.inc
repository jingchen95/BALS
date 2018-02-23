# Use 'make -j #' to do # number of parallel jobs ('make -j 8' for 8 jobs)
# When running testers ('make run'), it forces 'make -j 1' to run them serially

CC        = gcc    # C    compiler
CXX       = g++    # C++  compiler
LD        = g++    # Linker
NVCC      = nvcc   # CUDA Compiler

CFLAGS    = -DADD_ -m64 -fopenmp  # C    compiler flags
CXXFLAGS  = -DADD_ -m64 -fopenmp  # C++  compiler flags
LDFLAGS   = -DADD_ -m64 -fopenmp  # Linking flags
NVCCFLAGS = -DADD_ -m64           # CUDA compiler flags

# CUDA architecture flags
# Generate binary code for specific architecture.
# For faster compilation, use only one of these, but can use multiple.
# Fermi  2.0:  -gencode arch=compute_20,code=sm_20
# Kepler 3.0:  -gencode arch=compute_30,code=sm_30
# Kepler 3.5:  -gencode arch=compute_35,code=sm_35
NVCCFLAGS += -gencode arch=compute_35,code=sm_35

# Compiler flags for include path. Used for CC, CXX, NVCC.
CUDADIR=/usr/local/cuda-7.5
MAGMA_DIR=/home/chenjing/magma-2.2.0
#MKLROOT=/home/intel/composer_xe_2015.1.133/mkl
#MKLROOT=/home/intel/mkl/lib/intel64
MKLROOT=/home/chenjing/intel/composer_xe_2015.1.133/mkl/lib/intel64
INC  = \
	-I$(CUDADIR)/include \
	-I$(MAGMA_DIR)/include \
	-I$(MAGMA_DIR)/control \
	-I$(MAGMA_DIR)/testing \
	-I$(MAGMA_DIR)/sparse-iter/include \

# Linker flags for library path and libraries.
LIB  = \
	-L$(CUDADIR)/lib64 -lcurand -lcublas -lcusparse -lcudart \
	-L$(MAGMA_DIR)/testing -ltest \
	-L$(MAGMA_DIR)/lib -lmagma -lmagma_sparse \

# MacOS Accelerate
#LIB += \
#	-framework Accelerate \

# MKL
# may need -lstdc++ -lgfortran
LIB += \
	-L$(MKLROOT) -L/home/chenjing/intel/composer_xe_2015.1.133/compiler/lib/intel64 \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -liomp5 \

# Arguments given to run each command
# -f or --range feature space sizes
RUN_ARGS = -f 16 -f 32 $(HOME)/matrices/rec-yelp-user-business.mtx

# Timeout in seconds for each run
TIMEOUT  = 600
