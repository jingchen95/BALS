# -------------------- tools & flags
include make.inc

# If not set, make CXXFLAGS same as CFLAGS
CXXFLAGS  ?= $(CFLAGS)

INC += \
	-I$(MAGMA_DIR)/include \
	-I$(MAGMA_DIR)/control \
	-I$(MAGMA_DIR)/testing \
	-I$(MAGMA_DIR)/sparse/include \

LIB := \
	$(LIBDIR) $(LIB) \
	-L $(MAGMA_DIR)/lib -lmagma -lmagma_sparse \
	-L $(MAGMA_DIR)/testing \

codegen      := $(MAGMA_DIR)/tools/codegen.py

# -------------------- files
exes := \
	testing_sals \

srcs := \
	magma_sals.cu    \
	magma_sals.h     \
	testing_sals.cpp \

sobjs := testing_sals.o magma_sals.o


# ------------------------------------------------------------ rules
default: testing_sals

all: $(exes)


# -------------------- builds directories
builds: builds/stamp

builds/stamp: generate.csv magma_sals.cu testing_sals.o
	$(beast_build) -c 20 builds $^
	touch $@


# -------------------- objects
.SUFFIXES:

# add dependency on header
testing_sals.o magma_sals.o:   magma_sals.h

ALS_DEFS := -DXBLOCK=700 -DYBLOCK=614 -DP1BATCH=1 -DINDEX=0 -DBLK_K=20 -DBLK_M=10 -DDIM_M=5 -DDIM_N=5 -DDIM_M_A=5 -DDIM_N_A=5 -DTEX_A=0

%.i: %.cu
	$(NVCC) -E $(NVCCFLAGS) $(ALS_DEFS) $(INC) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(ALS_DEFS) $(INC) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ALS_DEFS) $(INC) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) $(ALS_DEFS) $(INC) -std=c99 -Wno-unused-variable -c -o $@ $<


# -------------------- exes
testing_sals: $(sobjs)

# -----
$(exes):
	$(CXX) $(LDFLAGS) $(INC) -o $@ $^ $(LIB)


# -------------------- clean
clean:
	-rm -f $(exes) $(generate_exes) *.o *.pyc generate.c generate_pydecl.c

distclean: clean
	-rm -f *.csv

.DELETE_ON_ERROR:
