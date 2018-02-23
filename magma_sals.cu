#include "magma_internal.h"
#include "magmasparse.h"

#include "template_support.h"

#include "magma_sals.h"

#ifndef TEX_A
#define TEX_A true
#endif

#define DIM_Z min(1024/(dim_m*dim_n),56)

#ifdef BLK_M
    #ifndef BLK_N
    #define BLK_N BLK_M  // square tiles
    #endif
    const int kernel_index = INDEX;
    
    const int blk_m   = BLK_M;
    const int blk_n   = BLK_N;
    const int blk_k   = BLK_K;
    const int dim_m   = DIM_M;
    const int dim_n   = DIM_N;

    const int dim_m_a = DIM_M_A;
    const int dim_n_a = DIM_N_A;
#else
    #define INDEX 0
    const int kernel_index = INDEX;
    
    const int blk_m   = 16;
    const int blk_n   = blk_m;
    const int blk_k   = 8;
    const int dim_m   = 4;
    const int dim_n   = 4;

    const int dim_m_a = 8;
    const int dim_n_a = 8;
#endif

template<
    typename scalar_t,
    typename objectA_t >
__global__ void
magma_sals_kernel_lower1(
    int m,
    int batch_NB,
    int rows,
    int position,
    int f, int new_row,
    objectA_t A, magma_int_t lda,
          scalar_t * __restrict__ C, magma_int_t ldc,
	magma_index_t *offset_s,
	magma_index_t *offset,
	magma_index_t *oindex ,
	magma_index_t *pindex , 
	int sa)
{
    #define   A(i_, j_) fetch< scalar_t >( A, (i_) + (j_)*lda )    
    #define  sY1(i_, j_) ( sY1[ (i_) + (j_)*(blk_m)     ])
    #define  sY2(i_, j_) ( sY2[ (i_) + (j_)*(blk_n)     ])
    #define   C(i_, j_) (  C[ (i_) + (j_)*ldc           ])
    
    const int batchid = blockIdx.z;
    
    const int blx = blockIdx.x;        // block's m position
    const int bly = blockIdx.y;        // block's n position
    if ( blx < bly ) { return; }       // ignore blocks in upper triangle
    
    const int tx  = threadIdx.x;       // thread's m position in C
    const int ty  = threadIdx.y;       // thread's m position in C
    const int tz  = threadIdx.z;
    const int tid = tx + ty * dim_m + tz * dim_m * dim_n;     // thread's number

    scalar_t * C_old;
    C_old = C;
    __shared__ scalar_t sY1[ blk_m * YBLOCK ];
    scalar_t rC[ blk_m/dim_m * blk_n/dim_n * ((XBLOCK-1)/DIM_Z+1) ] = { 0 };

    if(m < batch_NB)
    {
    	if(batchid == gridDim.z-1)
	{
		new_row = m % XBLOCK;
	}	
    }

    if(blx == bly)
    {
	for(int consa = 0; consa < sa; consa++)
	{
		int oft = position + batchid * sa + consa;
		if(oindex[oft + 1] - oindex[oft] > 0)
		{
			for(int k = oindex[oft] + tid; k < oindex[oft + 1]; k += dim_m*dim_n*blockDim.z)
			{
				int of = offset_s[k];
				for(int i=0;i<blk_m;i++)
				{
					sY1(i,k-oindex[oft]) = A(i+blx*blk_m, of);
				}
			}
			__syncthreads();
		
			int pptr = 0;
			while(pptr < new_row)
			{
				int index = (pptr/blockDim.z) * blk_m/dim_m * blk_n/dim_n;
				int remnant = min(blockDim.z, new_row-pptr);
				if(tz < remnant)
				{
					int ed = position * XBLOCK + (batchid * sa + consa) * new_row + pptr + tz;
					int length = pindex[ed+1]-pindex[ed];
					if(length>0)
					{
						#pragma unroll
						for(int k = pindex[ed]; k < pindex[ed+1]; k++)
						{
							int offset_k = offset[k];
							#pragma unroll
							for( int j=0; j < blk_n/dim_n; ++j ) 
							{
								#pragma unroll
								for( int i=0; i < blk_m/dim_m; ++i ) 
								{
									rC[i + j * blk_m/dim_m + index] +=  sY1(tx + i*dim_m, offset_k) * sY1(ty + j*dim_n, offset_k);
								}
							}
						}
					}	
				}
				pptr += blockDim.z;
			}		
		}
	}

	int ptr = 0;
	while(ptr < new_row)
	{
		int index = (ptr/blockDim.z) * (blk_m/dim_m) * (blk_n/dim_n);
		int remnant = min(blockDim.z, new_row-ptr);
		if(tz < remnant)
		{
			C = C_old + (batchid +ptr+ tz) * ldc * f;
			#pragma unroll
			for( int j=0; j < blk_n/dim_n; ++j ) 
			{
				int jj = bly*blk_n + j*dim_n + ty;          // global col index
				#pragma unroll
				for( int i=0; i < blk_m/dim_m; ++i ) 
				{
					int ii = blx*blk_m + i*dim_m + tx;  // global row index
					if(ii >= jj)
					{
						C(ii,jj) = rC[i + j * blk_m/dim_m + index];
					}
				}
			}
		}
		ptr += blockDim.z;
	}
    }
    else
    {
	for(int consa = 0; consa < sa; consa++)
	{
		int oft = position + batchid * sa + consa;
		if(oindex[oft + 1] - oindex[oft] > 0)
		{
			__shared__ scalar_t sY2[ blk_n * YBLOCK ];
			for(int k = oindex[oft] + tid; k < oindex[oft + 1]; k += dim_m*dim_n*blockDim.z)
			{
				int of = offset_s[k];
				for(int i=0;i<blk_m;i++)
				{
					sY1(i,k-oindex[oft]) = A(i+blx*blk_m, of);
				}
				for(int i=0;i<blk_m;i++)
				{
					sY2(i,k-oindex[oft]) = A(i+bly*blk_m, of);
				}
			}
			__syncthreads();
		
			int pptr = 0;
			while(pptr < new_row)
			{
				int index = (pptr/blockDim.z)*blk_m/dim_m*blk_n/dim_n;
				int remnant = min(blockDim.z, new_row-pptr);
				if(tz < remnant)
				{
					int ed = position * XBLOCK + (batchid * sa + consa) * new_row + pptr + tz;
					int length = pindex[ed+1]-pindex[ed];
					if(length>0)
					{
						#pragma unroll
						for(int k = pindex[ed]; k < pindex[ed+1]; k++)
						{
							int offset_k = offset[k];
							#pragma unroll
							for( int j=0; j < blk_n/dim_n; ++j ) 
							{
								#pragma unroll
								for( int i=0; i < blk_m/dim_m; ++i ) 
								{
									rC[i+j*blk_m/dim_m+index] +=  sY1(tx + i*dim_m, offset_k) * sY2(ty + j*dim_n, offset_k);
								}
							}
						}
					}	
				}
				pptr += blockDim.z;
			}		
		}
	}

	int ptr = 0;
	while(ptr < new_row)
	{
		int index = (ptr/blockDim.z)*blk_m/dim_m*blk_n/dim_n;
		int remnant = min(blockDim.z, new_row-ptr);
		if(tz < remnant)
		{
			C = C_old + (batchid + ptr + tz) * ldc * f;
			#pragma unroll
			for( int j=0; j < blk_n/dim_n; ++j ) 
			{
				int jj = bly*blk_n + j*dim_n + ty;          // global col index
				#pragma unroll
				for( int i=0; i < blk_m/dim_m; ++i ) 
				{
					int ii = blx*blk_m + i*dim_m + tx;  // global row index
					if(ii >= jj)
					{
						C(ii,jj) = rC[i+j*blk_m/dim_m+index];
					}
				}
			}
		}
		ptr += blockDim.z;
	}
    }
}

template<
    typename scalar_t,
    typename objectA_t >
__global__ void
magma_sals_kernel_lower2 (int f, objectA_t A, magma_int_t lda, const magma_index_t * __restrict__ R_rowptr, const magma_index_t * __restrict__ R_colind, scalar_t * __restrict__ C, magma_int_t ldc)
{
    	#define   A(i_, j_) fetch< scalar_t >( A, (i_) + (j_)*lda )
    	#define  sA(i_, j_) ( sA[ (i_) + (j_)*(blk_m+1)     ])
    	#define  sB(i_, j_) ( sB[ (i_) + (j_)*(blk_n+1)     ])
    	#define   C(i_, j_) (  C[ (i_) + (j_)*ldc           ])
    	#define  rC(i_, j_) ( rC[ (i_) + (j_)*(blk_m/dim_m_a) ])
    
    	const int batchid = blockIdx.z;
    
    	const int blx = blockIdx.x;        // block's m position
    	const int bly = blockIdx.y;        // block's n position
    	if ( blx < bly ) { return; }       // ignore blocks in upper triangle
    
    	const int tx  = threadIdx.x;       // thread's m position in C
    	const int ty  = threadIdx.y;       // thread's m position in C
    	const int tid = tx + ty*dim_m_a;     // thread's number
    
    	const int txA = tid % dim_m_a;     // thread's m position for loading A
    	const int tyA = tid / dim_m_a;     // thread's n position for loading A
    
    	const scalar_t c_one = make< scalar_t >( 1, 0 );
    
    	__shared__ scalar_t sA[ (blk_m+1) * blk_k ];  
  
    	// initialize C = 0
    	scalar_t rC[ (blk_m/dim_m_a)*(blk_n/dim_n_a) ] = {0};

    	if ( blx == bly ) 
	{
        	int ptr = R_rowptr[batchid];
        	while( ptr < R_rowptr[batchid+1] - (blk_k - 1) ) 
		{
            		#pragma unroll
            		for( int k=0; k < blk_k; k += dim_n_a ) 
			{
                		int kk = R_colind[ptr + k + tyA];
                		#pragma unroll
                		for( int i=0; i < blk_m; i += dim_m_a ) 
				{
                    			sA(i+txA, k+tyA) = A(i+txA + blx*blk_m, kk);
                		}
            		}
           		__syncthreads();
            
            		// Multiply Cu += A * Ru * A^T
            		#pragma unroll
            		for( int k=0; k < blk_k; ++k ) 
			{
                		#pragma unroll
                		for( int j=0; j < blk_n/dim_n_a; ++j ) 
				{
                    			#pragma unroll
                    			for( int i=0; i < blk_m/dim_m_a; ++i ) 
					{
						rC(i,j) += sA(tx + i*dim_m_a, k) * sA(ty + j*dim_n_a, k);
                    			}
                		}
            		}
            		__syncthreads(); 
            		ptr += blk_k;
        	}
        
        	if ( ptr < R_rowptr[batchid+1] ) 
		{
            		int part = R_rowptr[batchid+1] - ptr;

            		// Load A dev->shmem
           		#pragma unroll
		    	for( int k=0; k < blk_k; k += dim_n_a ) 
			{
				if ( k + tyA < part ) 
				{
				    	int kk = R_colind[ptr + k + tyA];
				    	#pragma unroll
				    	for( int i=0; i < blk_m; i += dim_m_a ) 
					{
				        	sA(i+txA, k+tyA) = A(i+txA + blx*blk_m, kk);
				    	}
				}
		    	}
            		__syncthreads();
            
		    	// Multiply Cu += A * Ru * A^T
		    	for( int k=0; k < part; ++k ) 
			{
				#pragma unroll
				for( int j=0; j < blk_n/dim_n_a; ++j ) 
				{
				    	#pragma unroll
				    	for( int i=0; i < blk_m/dim_m_a; ++i ) 
					{
						rC(i,j) += sA(tx + i*dim_m_a, k) * sA(ty + j*dim_n_a, k);
				    	}
				}
		    	}
        	}
    
		// save results, only lower triangle
		C += batchid*ldc*f;
		#pragma unroll
		for( int j=0; j < blk_n/dim_n_a; ++j ) 
		{
		    	int jj = bly*blk_n + j*dim_n_a + ty; 
			#pragma unroll
			for( int i=0; i < blk_m/dim_m_a; ++i ) 
			{
				int ii = blx*blk_m + i*dim_m_a + tx;  
				if ( ii >= jj ) 
				{
				        C(ii,jj) = rC(i,j);
				}
		    	}
		}
    	}
    	else 
	{
		__shared__ scalar_t sB[ (blk_n+1) * blk_k ];
      
        	int ptr = R_rowptr[batchid];
        	while( ptr < R_rowptr[batchid+1] - (blk_k - 1) ) 
		{
            		#pragma unroll
            		for( int k=0; k < blk_k; k += dim_n_a ) 
			{
				int kk = R_colind[ptr + k + tyA];
				#pragma unroll
				for( int i=0; i < blk_m; i += dim_m_a ) 
				{
				    sA(i+txA, k+tyA) = A(i+txA + blx*blk_m, kk);
				}
				#pragma unroll
				for( int i=0; i < blk_m; i += dim_m_a ) 
				{
				    sB(i+txA, k+tyA) = A(i+txA + bly*blk_m, kk);
				}
            		}
            		__syncthreads();
           
            		// Multiply Cu += A * Ru * A^T
            		#pragma unroll
            		for( int k=0; k < blk_k; ++k ) 
			{
				#pragma unroll
				for( int j=0; j < blk_n/dim_n_a; ++j ) 
				{
				    	#pragma unroll
				    	for( int i=0; i < blk_m/dim_m_a; ++i ) 
					{
						rC(i,j) += sA(tx + i*dim_m_a, k) * sB(ty + j*dim_n_a, k);
				    	}
				}
            		}
            		__syncthreads();           
	
            		ptr += blk_k;
        	}
        
		if ( ptr < R_rowptr[batchid+1] ) 
		{
		    	// partial < blk_k number of columns
		    	// still assumes full blk_m number of rows
		    	int part = R_rowptr[batchid+1] - ptr;
		 
		    	// Load A dev->shmem, and B (i.e., A^T) dev->shmem
		   	#pragma unroll
		    	for( int k=0; k < blk_k; k += dim_n_a ) 
			{
				if ( k + tyA < part ) 
				{
				    	int kk = R_colind[ptr + k + tyA];
				    	#pragma unroll
				    	for( int i=0; i < blk_m; i += dim_m_a ) 
					{
				        	sA(i+txA, k+tyA) = A(i+txA + blx*blk_m, kk);
				    	}
				    	#pragma unroll
				    	for( int i=0; i < blk_m; i += dim_m_a ) 
					{
				        	sB(i+txA, k+tyA) = A(i+txA + bly*blk_m, kk);
				    	}
				}
		    	}
		    	__syncthreads();
		   
		    	for( int k=0; k < part; ++k ) 
			{
		        	#pragma unroll
		       		for( int j=0; j < blk_n/dim_n_a; ++j ) 
				{
			    		#pragma unroll
		            		for( int i=0; i < blk_m/dim_m_a; ++i) 
					{
						rC(i,j) += sA(tx + i*dim_m_a, k) * sB(ty + j*dim_n_a, k);
		            		}
		        	}
		    	}
	
		}
  
        	// save results, whole off-diagonal block
        	C += batchid*ldc*f;
        	#pragma unroll
        	for( int j=0; j < blk_n/dim_n_a; ++j ) 
		{
            		int jj = bly*blk_n + j*dim_n_a + ty;          
                	#pragma unroll
                	for( int i=0; i < blk_m/dim_m_a; ++i ) 
			{
                    		int ii = blx*blk_m + i*dim_m_a + tx;  
                    		C(ii,jj) = rC(i,j);
                	}
        	}
}


extern "C" magma_int_t
magma_sals1(
    magma_int_t rows,
    magma_int_t batch_NB,
    magma_int_t position,
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,   
    magma_int_t Am,
    magma_int_t An,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magma_int_t Cm,
    magma_int_t Cn,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaIndex_ptr doffset_s, magmaIndex_ptr doffset, 
    magmaIndex_ptr doindex, magmaIndex_ptr dpindex,
    magma_int_t threadblock_size,
    magma_int_t sa,
    magma_int_t new_row,
    magma_queue_t queue )
{
    	if ( ! TEX_A ) 
    	{
        	assert( ldda >= magma_ceildiv(f, blk_m)*blk_m );
    	}
    	typedef typename memory_traits< float, TEX_A >::object_const_restrict_type objectA_t;    
    	objectA_t objA;
    	setup_memory( Am, An, dA, ldda, &objA );
    
    	dim3 threads( dim_m, dim_n, DIM_Z);
    	dim3 grid( magma_ceildiv(f, blk_m), magma_ceildiv(f, blk_n), threadblock_size);
    	magma_sals_kernel_lower1<<< grid, threads, 0, magma_queue_get_cuda_stream(queue) >>>(m, batch_NB, rows, position, f, new_row, objA, ldda, dC, lddc, doffset_s, doffset, doindex, dpindex, sa);
	cudaDeviceSynchronize(); 
    	teardown_memory( objA );
    	return 0;
}

extern "C" magma_int_t
magma_sals2(
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,
    magma_int_t Am,
    magma_int_t An,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    const magma_s_matrix dR,
    magma_int_t Cm,
    magma_int_t Cn,
    magmaFloat_ptr dC, magma_int_t lddc,
    magma_queue_t queue )
{
    	if ( ! TEX_A ) 
	{
		assert( ldda >= magma_ceildiv(f, blk_m)*blk_m );
    	}
    	typedef typename memory_traits< float, TEX_A >::object_const_restrict_type objectA_t;    
    	objectA_t objA;
    	setup_memory( Am, An, dA, ldda, &objA );
    
    	dim3 grid( magma_ceildiv(f, blk_m), magma_ceildiv(f, blk_n), m );
    	dim3 threads( dim_m_a, dim_n_a);

    	magma_sals_kernel_lower2<<< grid, threads, 0, magma_queue_get_cuda_stream(queue) >>>( f, objA, ldda, dR.drow, dR.dcol, dC, lddc);
    	cudaDeviceSynchronize();    
    	teardown_memory( objA );
    	return 0;
}


// ------------------------------------------------------------
extern "C" void
magma_sals_print_index()
{
    printf( "%5d  (%2d  %2d  %2d.  %2d  %2d) ", kernel_index, blk_m, blk_n, blk_k, dim_m, dim_m_a);
}
