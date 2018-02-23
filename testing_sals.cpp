/*
    Copyright 2009-2016, University of Tennesse. See COPYRIGHT file.
    
    @author Hartwig Anzt
    @author Mark Gates
    @author Jakub Kurzak
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include <omp.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

// in control
#include "magma_threadsetting.h"

// in beast
#include "magma_sals.h"

// ---------------------------------------------------------------------------
// Forms matrices C_blk = A * diag( alpha * R(blk,:)     ) * A^H + AA^H + lambda*I
// and   vectors  x_blk = A * diag( alpha * R(blk,:) + 1 ) * bool( R(blk,:) ).
// for blk = 1, ..., m, where A is a set of row vectors.
//
// A   is f by n
// AAT is f by f
// R   is m by n
// C   is (f by f) by m; each C is (f by f)
// X   is f by m
//
// simple CPU implementation for accuracy (not performance!) comparison
//
// This implementation (zals1) uses inline ssyrk-like loops code (no BLAS calls).

extern "C" magma_int_t
testing_sals1(
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,
    
    magma_int_t Am,
    magma_int_t An,
    float const * __restrict__ A, magma_int_t lda,
    float alpha,
    //magma_s_matrix const R,
    // pass fields separately to get __restrict__ on all pointers
    const magma_index_t      * __restrict__ R_rowptr,
    const magma_index_t      * __restrict__ R_colind,
    const float * __restrict__ R_values,
    magma_int_t AAm,
    magma_int_t AAn,
    float * __restrict__ AAT, magma_int_t ldaa,
    float lambda,
    magma_int_t Cm,
    magma_int_t Cn,
    float * __restrict__ C, magma_int_t ldc,
    magma_int_t Xm,
    magma_int_t Xn,
    float * __restrict__ X, magma_int_t ldx )
{
    #define   A(i_, j_) (  A[ (i_) + (j_)*lda  ])
    #define AAT(i_, j_) (AAT[ (i_) + (j_)*ldaa ])
    #define   C(i_, j_) (  C[ (i_) + (j_)*ldc  ])
    #define   X(i_, j_) (  X[ (i_) + (j_)*ldx  ])
    
    assert( f > 0 );
    assert( m > 0 );
    assert( n > 0 );
    
    assert( Am  == f );
    assert( An  == n );
    assert( lda >= f );
    
    //assert( R.storage_type == Magma_CSR );
    //assert( R.num_rows >= m );  // allow subset of rows
    //assert( R.num_cols == n );
    
    assert( AAm  == f );
    assert( AAn  == f );
    assert( ldaa >= f );
    
    assert( Cm  == f   );
    assert( Cn  == f   );  // re-use one matrix  //f*m );
    assert( ldc >= f   );
    
    assert( Xm  == f );
    assert( Xn  == m );
    assert( ldx >= f );
    
    const float c_one = MAGMA_S_ONE;
    //const magma_int_t ione = 1;
    
    float cui;
    
    for( int blk=0; blk < m; ++blk ) {
        // zero Ci and xi
        for( int j=0; j < f; ++j ) {
            for( int i=0; i < f; ++i ) {
                //C(i,j) = AAT(i,j);
                C(i,j) = MAGMA_S_ZERO;
            }
            //C(j,j) += lambda;  // add regularization to diagonal
        }
        for( int i=0; i < f; ++i ) {
            X[i] = MAGMA_S_ZERO;
        }
        // accumulate rank-K update into Ci, where K is number of non-zeros in row blk of R.
        // accumulate rhs into xi.
        for( int k = R_rowptr[blk]; k < R_rowptr[blk+1]; ++k ) {
            int kk = R_colind[k];
            //cui = alpha * R_values[k];
            for( int j=0; j < f; ++j ) {
                for( int i=j; i < f; ++i ) {
                    C(i,j) += A(i,kk) * A(j,kk);
                }
            }
            for( int i=0; i < f; ++i ) {
                X[i] += (c_one + cui) * A(i,kk);
            }
        }
        
        // re-use one matrix  
        C += f*ldc;  // shift to next f-by-f Ci matrix
        X +=   ldx;  // shift to next rhs
    }
    
    #undef A
    #undef AAT
    #undef C
    #undef X
    
    return 0;
}
// end testing_sals1


// ---------------------------------------------------------------------------
// Forms matrices C_blk = A * diag( alpha * R(blk,:)     ) * A^H + AA^H + lambda*I
// and   vectors  x_blk = A * diag( alpha * R(blk,:) + 1 ) * bool( R(blk,:) ).
// for blk = 1, ..., m, where A is a set of row vectors.
//
// A   is f by n
// AAT is f by f
// R   is m by n
// C   is (f by f) by m; each C is (f by f)
// X   is f by m
//
// simple CPU implementation for accuracy (not performance!) comparison
//
// This implementation (zals2) uses sgemm BLAS call.
// It assembles matrices into work, of size lwork >= 2*ldw*max( rank_i ),
// where rank_i is the number of rows in the i-th row of A.
// Using non-symmetric sgemm, this does twice the required flops.

extern "C" magma_int_t
testing_sals2(
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,
    
    magma_int_t Am,
    magma_int_t An,
    float const * __restrict__ A, magma_int_t lda,
    float alpha,
    //magma_s_matrix const R,
    // pass fields separately to get __restrict__ on all pointers
    const magma_index_t      * __restrict__ R_rowptr,
    const magma_index_t      * __restrict__ R_colind,
    const float * __restrict__ R_values,
    magma_int_t AAm,
    magma_int_t AAn,
    float * __restrict__ AAT, magma_int_t ldaa,
    float lambda,
    magma_int_t Cm,
    magma_int_t Cn,
    float * __restrict__ C, magma_int_t ldc,
    magma_int_t Xm,
    magma_int_t Xn,
    float * __restrict__ X, magma_int_t ldx,
    float * __restrict__ work, magma_int_t lwork )
{
    #define     A(i_, j_) (    A[ (i_) + (j_)*lda  ])
    #define   AAT(i_, j_) (  AAT[ (i_) + (j_)*ldaa ])
    #define     C(i_, j_) (    C[ (i_) + (j_)*ldc  ])
    #define     X(i_, j_) (    X[ (i_) + (j_)*ldx  ])
    #define  work(i_, j_) ( work[ (i_) + (j_)*ldw  ])
    #define work2(i_, j_) (work2[ (i_) + (j_)*ldw  ])
    
    assert( f > 0 );
    assert( m > 0 );
    assert( n > 0 );
    
    assert( Am  == f );
    assert( An  == n );
    assert( lda >= f );
    
    //assert( R.storage_type == Magma_CSR );
    //assert( R.num_rows >= m );  // allow subset of rows
    //assert( R.num_cols == n );
    
    assert( AAm  == f );
    assert( AAn  == f );
    assert( ldaa >= f );
    
    assert( Cm  == f   );
    assert( Cn  == f   );  // re-use one matrix  //f*m );
    assert( ldc >= f   );
    
    assert( Xm  == f );
    assert( Xn  == m );
    assert( ldx >= f );
    
    const float c_one = MAGMA_S_ONE;
    //const magma_int_t ione = 1;
    
    float *work2;
    float cui;
    magma_int_t ldw = f;
    
    for( int blk=0; blk < m; ++blk ) {
        // zero Ci and xi
        for( int j=0; j < f; ++j ) {
            for( int i=0; i < f; ++i ) {
                C(i,j) = AAT(i,j);
            }
            C(j,j) += lambda;  // add regularization to diagonal
        }
        for( int i=0; i < f; ++i ) {
            X[i] = MAGMA_S_ZERO;
        }
        // accumulate rank-K update into Ci, where K is number of non-zeros in row blk of R.
        // accumulate rhs into xi.
        int rank = R_rowptr[blk+1] - R_rowptr[blk];
        assert( lwork >= 2*ldw*rank );
        work2 = work + ldw*rank;
        int j = 0;
        for( int k = R_rowptr[blk]; k < R_rowptr[blk+1]; ++k ) {
            int kk = R_colind[k];
            cui = alpha * R_values[k];
            for( int i=0; i < f; ++i ) {
                work(i,j)  = A(i,kk)*cui;  // apply diagonal scaling
                work2(i,j) = A(i,kk);
                X[i] += (c_one + cui) * A(i,kk);
            }
            ++j;
        }
        blasf77_sgemm( "no", "conj", &f, &f, &rank, &c_one, work, &ldw, work2, &ldw, &c_one, C, &ldc );
        
        // reset upper triangle
        for( int j=0; j < f; ++j ) {
            for( int i=0; i < j; ++i ) {
                C(i,j) = AAT(i,j);
            }
        }
        
        // re-use one matrix  //C += f*ldc;  // shift to next f-by-f Ci matrix
        X +=   ldx;  // shift to next rhs
    }
    
    #undef A
    #undef AAT
    #undef C
    #undef X
    
    return 0;
}
// end testing_sals2


// ---------------------------------------------------------------------------
// Forms matrices C_blk = A * diag( alpha * R(blk,:)     ) * A^H + AA^H + lambda*I
// and   vectors  x_blk = A * diag( alpha * R(blk,:) + 1 ) * bool( R(blk,:) ).
// for blk = 1, ..., m, where A is a set of row vectors.
//
// A   is f by n
// AAT is f by f
// R   is m by n
// C   is (f by f) by m; each C is (f by f)
// X   is f by m
//
// simple CPU implementation for accuracy (not performance!) comparison
//
// This implementation (zals3) uses ssyrk BLAS call.
// It assembles matrices into work, of size lwork >= ldw*max( rank_i ),
// where rank_i is the number of rows in the i-th row of A.
// Using ssyrk, this does the correct number of flops,
// but gets the correct answer only if all non-zero A_ij == 1.

extern "C" magma_int_t
testing_sals(
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,
    
    magma_int_t Am,
    magma_int_t An,
    float const * __restrict__ A, magma_int_t lda,
    float alpha,
    //magma_s_matrix const R,
    // pass fields separately to get __restrict__ on all pointers
    const magma_index_t      * __restrict__ R_rowptr,
    const magma_index_t      * __restrict__ R_colind,
    const float * __restrict__ R_values,
    magma_int_t AAm,
    magma_int_t AAn,
    float * __restrict__ AAT, magma_int_t ldaa,
    float lambda,
    magma_int_t Cm,
    magma_int_t Cn,
    float * __restrict__ C, magma_int_t ldc,
    magma_int_t Xm,
    magma_int_t Xn,
    float * __restrict__ X, magma_int_t ldx,
    float * __restrict__ work, magma_int_t lwork )
{
    #define     A(i_, j_) (    A[ (i_) + (j_)*lda  ])
    #define   AAT(i_, j_) (  AAT[ (i_) + (j_)*ldaa ])
    #define     C(i_, j_) (    C[ (i_) + (j_)*ldc  ])
    #define     X(i_, j_) (    X[ (i_) + (j_)*ldx  ])
    #define  work(i_, j_) ( work[ (i_) + (j_)*ldw  ])
    
    assert( f > 0 );
    assert( m > 0 );
    assert( n > 0 );
    
    assert( Am  == f );
    assert( An  == n );
    assert( lda >= f );
    
    //assert( R.storage_type == Magma_CSR );
    //assert( R.num_rows >= m );  // allow subset of rows
    //assert( R.num_cols == n );
    
    assert( AAm  == f );
    assert( AAn  == f );
    assert( ldaa >= f );
    
    assert( Cm  == f   );
    assert( Cn  == f   );  // re-use one matrix  //f*m );
    assert( ldc >= f   );
    
    assert( Xm  == f );
    assert( Xn  == m );
    assert( ldx >= f );
    
    const float c_one = MAGMA_S_ONE;
    const float d_one = MAGMA_D_ONE;
    //const magma_int_t ione = 1;
    
    float cui;
    magma_int_t ldw = f;
    
    for( int blk=0; blk < m; ++blk ) {
        // zero Ci and xi
        for( int j=0; j < f; ++j ) {
            for( int i=0; i < f; ++i ) {
                C(i,j) = AAT(i,j);
            }
            C(j,j) += lambda;  // add regularization to diagonal
        }
        for( int i=0; i < f; ++i ) {
            X[i] = MAGMA_S_ZERO;
        }
        // accumulate rank-K update into Ci, where K is number of non-zeros in row blk of R.
        // accumulate rhs into xi.
        int rank = R_rowptr[blk+1] - R_rowptr[blk];
        assert( lwork >= ldw*rank );
        int j = 0;
        for( int k = R_rowptr[blk]; k < R_rowptr[blk+1]; ++k ) {
            int kk = R_colind[k];
            cui = alpha * R_values[k];
            for( int i=0; i < f; ++i ) {
                //work(i,j)  = A(i,kk)*cui;  // apply diagonal scaling
                work(i,j) = A(i,kk);
                X[i] += (c_one + cui) * A(i,kk);
            }
            ++j;
        }
        blasf77_ssyrk( "lower", "no", &f, &rank, &d_one, &work(0,0), &ldw, &d_one, C, &ldc );
        
        // re-use one matrix  //C += f*ldc;  // shift to next f-by-f Ci matrix
        X +=   ldx;  // shift to next rhs
    }
    
    #undef A
    #undef AAT
    #undef C
    #undef X
    
    return 0;
}
// end testing_sals

extern "C" int
cal_off(magma_int_t batch_nb, magma_int_t rows, magma_int_t cols, magma_int_t off, magma_int_t nnz,  const magma_index_t * __restrict__ R_rowptr, const magma_index_t * __restrict__ R_colind, magma_index_t *oindex, magma_index_t *pindex, magma_index_t *offset, magma_index_t *offset_s, int iterx, int itery)
{
	int num = 0, num1=0;
	int tem = 0, tem1=0;
	int h = 0;
	int v = 1;
	int *meoffset;
	meoffset = (int *)malloc(nnz*sizeof(int));

	for(int i = 0; i < nnz; i++)
	{
		offset[i] = -1;
                offset_s[i] = -1;
		meoffset[i] = -1;
        }
	pindex[0] = 0;
	oindex[0] = 0;

	for(int t=0;t<iterx;t++)
	{
		for(int j=0;j<itery;j++)
		{
			for(int row=t*XBLOCK;row<(t+1)*XBLOCK;row++)
			{
				if(row<rows)
				{
					int o = 0;
					for(int s=R_rowptr[row]+o;s<R_rowptr[row+1];s++)
					{
						if((R_colind[s] >= YBLOCK*j) && (R_colind[s]<YBLOCK*(j+1)))
						{
							offset[h] = R_colind[s];
							meoffset[h] = R_colind[s];
							h++;
							o++;
						}
						else
						{
							if(R_colind[s] >= YBLOCK*(j+1))
								break;
						}
					}
					pindex[v] = pindex[v-1]+o;
					v++; 
				}
			}
		}
	}

	if((rows % XBLOCK == 0) || (XBLOCK == 1))
        {
		int hh = 0;
		for(int i = 0; i < iterx * itery; i++)
		{
			offset_s[hh] = offset[pindex[i*XBLOCK]];
			hh++;
			int redundant = 1;
			for(int t = pindex[i*XBLOCK]+1; t < pindex[(i+1)*XBLOCK]; t++)
			{
				int dd = 0;
				for(int pp = pindex[i*XBLOCK]; pp < t; pp++)
				{
					if(offset[t] == offset[pp])
					{	
						dd++;	
						break;
					}
				}
				if(dd == 0)
				{
					offset_s[hh] = offset[t];
					redundant++;
					hh++;
				}
			}
			oindex[i+1] = oindex[i] + redundant;
		}
	}
	else
	{
		int hh = 0;
                for(int i = 0; i < (iterx-1) * itery; i++)
                {
			if(pindex[(i+1)*XBLOCK] == pindex[i*XBLOCK])
			{
				oindex[i+1] = oindex[i];
			}
			else
			{
				offset_s[hh] = offset[pindex[i*XBLOCK]];
                        	hh++;
                        	int redundant = 1;
                        	for(int t = pindex[i*XBLOCK]+1; t < pindex[(i+1)*XBLOCK]; t++)
                        	{
		                        int dd = 0;
		                        for(int pp = pindex[i*XBLOCK]; pp < t; pp++)
		                        {
		                                if(offset[t] == offset[pp])
		                                {
		                                        dd++;
		                                        break;
		                                }
		                        }
		                        if(dd == 0)
		                        {
		                                offset_s[hh] = offset[t];
		                                redundant++;
		                                hh++;
		                        }
                        	}
                        	oindex[i+1] = oindex[i] + redundant;
			}
                }
		for(int i=0;i<itery;i++)
                {
			if(pindex[(iterx-1)*itery*XBLOCK+(i+1)*(rows%XBLOCK)] == pindex[(iterx-1)*itery*XBLOCK+i*(rows%XBLOCK)])
			{
				oindex[(iterx-1)*itery+i+1] = oindex[(iterx-1)*itery+i];
			}
			else
			{
				offset_s[hh] = offset[pindex[(iterx-1)*itery*XBLOCK+i*(rows%XBLOCK)]];
				hh++;
                        	int redundant = 1;
		                for(int t=pindex[(iterx-1)*itery*XBLOCK+i*(rows%XBLOCK)]+1;t<pindex[(iterx-1)*itery*XBLOCK+(i+1)*(rows%XBLOCK)];t++)
		                {
					int dd = 0;
					for(int pp = pindex[(iterx-1)*itery*XBLOCK+i*(rows%XBLOCK)]; pp < t; pp++)
					{
						if(offset[t] == offset[pp])
		                                {
		                                        dd++;
		                                        break;
		                                }
					}
					if(dd == 0)
					{
						offset_s[hh] = offset[t];
		                                redundant++;
		                                hh++;
					}
				}
				oindex[(iterx-1)*itery+i+1] = oindex[(iterx-1)*itery+i] + redundant;
			}
		}	
	}

	for(int i=0;i<nnz;i++)
	{
		offset[i] = meoffset[i];
	}
	free(meoffset);
	
	if((rows%XBLOCK == 0) || (XBLOCK == 1))
	{ 
		for(int i = 0; i < rows*itery; i++)
		{
			for(int k=pindex[i];k < pindex[i+1];k++)
			{
				for(int j=oindex[i/XBLOCK];j<oindex[i/XBLOCK+1];j++)
				{
					if(offset[k] == offset_s[j])
					{
						offset[k] = j-oindex[i/XBLOCK];
						break;
					}
				}
			}
		}
	}
	else
	{
		for(int i = 0; i < (iterx-1)*XBLOCK*itery; i++)
		{
			for(int k=pindex[i];k < pindex[i+1];k++)
			{
				for(int j=oindex[i/XBLOCK];j<oindex[i/XBLOCK+1];j++)
				{
					if(offset[k] == offset_s[j])
					{
						offset[k] = j-oindex[i/XBLOCK];
						break;
					}
				}
			}
		}
		int new_row = rows%XBLOCK;
		int dsa = (iterx-1)*XBLOCK*itery;
		int zxc = (iterx-1)*itery - 1;
		for(int i = 0; i < new_row * itery; i++)
		{
			if(i%new_row == 0)
			{
				zxc=zxc+1;	
			}
			for(int k=pindex[dsa+i];k < pindex[dsa+i+1];k++)
			{
				for(int j=oindex[zxc];j<oindex[zxc+1];j++)
				{
					if(offset[k] == offset_s[j])
					{
						offset[k] = j-oindex[zxc];
						break;
					}
				}
			}
		}
	}	
}

extern "C" int
Reuse_order(magma_int_t batch_NB, magma_int_t rows, const magma_index_t * __restrict__ R_rowptr, magma_index_t *oindex, int as, int sa, magma_index_t *r)
{
	int *reuse;
	reuse = (int *)malloc(((rows-1)/batch_NB+1)*sizeof(int));
	for(int i = 0; i < (rows-1)/batch_NB+1; i++)
	{
		reuse[i] = 0;
	}
		
	// Occasion 1: Full batch
	for(int i = 0; i < rows/batch_NB; i++)
	{
		int ind = i * batch_NB / XBLOCK * sa;
		for(int j = 0; j < batch_NB/XBLOCK; j++)
		{
			reuse[i] += (R_rowptr[i * batch_NB + XBLOCK * (j+1)] - R_rowptr[i * batch_NB + XBLOCK * j]) - (oindex[ind + sa * (j+1)] - oindex[ind + sa * j]);
		}	
	}
	
	// Occasion 2: Part
	int accom = (rows/batch_NB)*batch_NB;  // Finished row number
	int batch_IB = rows - (rows/batch_NB)*batch_NB;
	if(batch_IB < XBLOCK)
	{
		reuse[(rows-1)/batch_NB] = (R_rowptr[rows] - R_rowptr[accom]) - (oindex[as*sa] - oindex[accom/XBLOCK*sa]);
	}
	else
	{
		for(int j = 0; j < batch_IB/XBLOCK; j++)
		{
			reuse[(rows-1)/batch_NB] += (R_rowptr[(rows-batch_IB) + XBLOCK*(j+1)] - R_rowptr[(rows-batch_IB) + XBLOCK*j]) - (oindex[(rows-batch_IB)/XBLOCK*sa + sa*(j+1)] - oindex[(rows-batch_IB)/XBLOCK*sa + sa*j]);
		}
		reuse[(rows-1)/batch_NB] += (R_rowptr[rows] - R_rowptr[rows/XBLOCK * XBLOCK]) - (oindex[as*sa] - oindex[rows/XBLOCK*sa]);
	}

	int h = 0;
	int tem = 0;
	int num = 0;
	for(int i = 0; i < (rows-1)/batch_NB+1; i++)
	{
		tem = reuse[0];
		num = 0;
		for(int pp = 1; pp < (rows-1)/batch_NB+1; pp++)
		{
			if(reuse[pp] > tem)	
			{
				tem = reuse[pp];
				num = pp;
			}
		}
		reuse[num] = -1;
		r[h] = num;
		h++;
	}
	free(reuse);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- testing
*/
int main( int argc, char** argv )
{
    magma_init();
    
    magma_queue_t queue;
    magma_queue_create( &queue );
    magmablasSetKernelStream( queue );
    
    const float c_zero    = MAGMA_S_ZERO;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const float d_one     = 1;
    const float d_zero    = 0;
    const magma_int_t ione = 1;
    
    real_Double_t time=0, gpu_time=0, cpu_time=0, sum_time;
    real_Double_t read_time=0, setup_time1, setup_time2, setup_time3, alloc_time, overhead_time;
    real_Double_t cpu_ssyrk1_time, cpu_ssyrk2_time;
    real_Double_t gpu_ssyrk1_time, gpu_ssyrk2_time;
    real_Double_t cpu_zals1_time, cpu_zals2_time;
    real_Double_t gpu_zals1_time, gpu_zals2_time;
    real_Double_t part1 = 0, part2 = 0, part3 = 0, part4 = 0;
    
    magma_int_t ISEED[4] = {0,0,0,1};
    float lambda = MAGMA_S_MAKE( 10, 0 );
    float alpha  = MAGMA_S_MAKE(  1, 0 );
    magma_int_t size;
    magma_int_t info;
    
    float work[1];
    magma_index_t *drowptr;
    
    magma_int_t niter   = 1;   // number of times to repeat test
    magma_int_t verbose = 0;
    magma_int_t align   = 16;
    bool first   = true;
    bool second  = true;
    
    float tol = 1000 * lapackf77_slamch("E");
    if ( verbose >= 1 ) { printf( "tol %.2e\n", tol ); }
    
    // largest batch to do -- limits the memory required
    magma_int_t batch_NB = 32768;
   
    const magma_int_t max_r_tests = 1000;
    magma_int_t fsize[ max_r_tests ] = { 32, 0 };  // rows in X, Y, A (rank)
    magma_int_t fsize_tests = 0;
    
    const magma_int_t max_thread_tests = 20;
    magma_int_t threads[ max_thread_tests ] = { 0 };
    magma_int_t nthread_tests = 0;
    
    const magma_int_t max_files = 100;
    char* files[ max_files ] = { NULL };
    magma_int_t nfiles = 0;
    
    int cpu_version = 1;
    
    // parse command line options
    const char* usage =
    "Usage: %s [options] matrix-files.mtx\n"
    "\n"
    "-h               help\n"
    "-f #             set rank (feature space size) of Y*Ru*Y^T, repeatable, default %d\n"
    "--range start:stop:step\n"
    "                 loop over inclusive range of feature space sizes, repeatable\n"
    "--batch #        size of GPU batch to do in one magma_sals call, default %d\n"
    "-v               increase verboseness, repeatable\n"
    "--verbose #      set verboseness to level, default %d\n"
    "--niter #        repeat number of times, default %d\n"
    "--align #        round ldda up to alignment, default %d\n"
    "--nthread #      set OpenMP number of threads, repeatable, default %d\n"
    "--first          do only first  ALS (Y*Ru*Y^T)\n"
    "--second         do only second ALS (X*Ri*X^T)\n"
    "--cpu-version #  use CPU version testing_sals1 (loops), 2 (sgemm), or 3 (ssyrk), default %d\n";
    
    for( int arg=1; arg < argc; ++arg ) {
        if ( strcmp("-h", argv[arg]) == 0 ) {
            printf( usage, argv[0], fsize[0], batch_NB, verbose, niter, align,
                    threads[0], cpu_version );
            exit(0);
        }
        else if ( strcmp("-f", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            fsize[ fsize_tests ] = atoi( argv[arg] );
            magma_assert( fsize[ fsize_tests ] > 0, "f=%s is invalid\n", argv[arg] );
            ++fsize_tests;
        }
        else if ( strcmp("--range", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            int start, stop, step;
            info = sscanf( argv[arg], "%d:%d:%d", &start, &stop, &step );
            magma_assert( info == 3 && start > 0 && stop > 0 && step != 0, "range=%s is invalid\n", argv[arg] );
            while( start <= stop && fsize_tests < max_r_tests ) {
                fsize[ fsize_tests++ ] = start;
                start += step;
            }
        }
        else if ( strcmp("--batch", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            batch_NB = atoi( argv[arg] );
            magma_assert( batch_NB > 0, "batch=%s is invalid\n", argv[arg] );
        }
        else if ( strcmp("-v", argv[arg]) == 0 ) {
            ++verbose;
        }
        else if ( strcmp("--verbose", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            verbose = atoi( argv[arg] );
            magma_assert( verbose >= 0, "verbose=%s is invalid\n", argv[arg] );
        }
        else if ( strcmp("--niter", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            niter = atoi( argv[arg] );
            magma_assert( niter >= 1, "niter=%s is invalid\n", argv[arg] );
        }
        else if ( strcmp("--align", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            align = atoi( argv[arg] );
            magma_assert( align >= 1, "align=%s is invalid\n", argv[arg] );
        }
        else if ( strcmp("--nthread", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            magma_assert( nthread_tests < max_thread_tests, "nthread=%s: too many --nthread options\n", argv[arg] );
            threads[ nthread_tests ] = atoi( argv[arg] );
            magma_assert( threads[ nthread_tests ] >= 1, "nthread=%s is invalid\n", argv[arg] );
            ++nthread_tests;
        }
        else if ( strcmp("--first", argv[arg]) == 0 ) {
            first  = true;
            second = false;
        }
        else if ( strcmp("--second", argv[arg]) == 0 ) {
            first  = false;
            second = true;
        }
        else if ( strcmp("--cpu-version", argv[arg]) == 0 && arg+1 < argc ) {
            ++arg;
            cpu_version = atoi( argv[arg] );
        }
        else if ( strncmp("-", argv[arg], 1) == 0 ) {
            fprintf( stderr, "Unknown option: %s\n", argv[arg] );
            exit(1);
        }
        else {
            // default is matrix file
            magma_assert( nfiles < max_files, "file %s: too many files\n", argv[arg] );
            files[ nfiles ] = argv[arg];
            ++nfiles;
        }
    } // end arg
    fsize_tests   = max( 1, fsize_tests   );  // at least default f=32 size
    nthread_tests = max( 1, nthread_tests );  // at least default nthread=0, i.e., use default OMP_NUM_THREADS
    
    printf( "# Usage: %s [options] matrix-files\n", argv[0] );
    printf( "# Options: --verbose #, -f #, --range start:stop:step, --batch #, --niter #, --align #, --nthread #, --first, --second.\n" );

    for( int ifile=0; ifile < nfiles; ++ifile ) {
        printf("# ==================================================================================================================================\n");
        magma_s_matrix hR, hRT, dR, dRT;
        read_time = magma_sync_wtime(0);
        info = magma_s_csr_mtx( &hR, files[ifile], queue );
        read_time = magma_sync_wtime(0) - read_time;
        if ( info != 0 ) {
            printf( "File %s failed: magma_s_csr_mtx returned error %d: %s.\n",
                    files[ifile], (int) info, magma_strerror( info ));
            continue;
        }
        
        for( int i=0; i < hR.nnz; ++i ) {
            hR.val[i] = MAGMA_S_ONE;
        }
        
        setup_time1 = magma_sync_wtime(0);
        info = magma_smtransfer(  hR,  &dR,  Magma_CPU, Magma_DEV, queue );
        if ( info != 0 ) {
            printf( "magma_smtransfer error %s (%d)\n", magma_strerror(info), info );
            exit(1);
        }
        setup_time1 = magma_sync_wtime(0) - setup_time1;
        
        setup_time2 = magma_sync_wtime(0);
        info = magma_smtranspose( dR,  &dRT, queue );
        if ( info != 0 ) {
            printf( "magma_smtranspose error %s (%d)\n", magma_strerror(info), info );
            exit(1);
        }
        setup_time2 = magma_sync_wtime(0) - setup_time2;
        
        setup_time3 = magma_sync_wtime(0);
        info = magma_smtransfer(  dRT, &hRT, Magma_DEV, Magma_CPU, queue );
        if ( info != 0 ) {
            printf( "magma_smtransfer error %s (%d)\n", magma_strerror(info), info );
            exit(1);
        }
        setup_time3 = magma_sync_wtime(0) - setup_time3;
        
        int max_row_nnz = 0;
        for( int i=0; i < hR.num_rows; ++i ) {
            max_row_nnz = max( max_row_nnz, hR.row[i+1] - hR.row[i] );
        }
        
        int max_col_nnz = 0;
        for( int i=0; i < hRT.num_rows; ++i ) {
            max_col_nnz = max( max_col_nnz, hRT.row[i+1] - hRT.row[i] );
        }
        printf( "# M %d, N %d, nnz %d\n", hR.num_rows, hR.num_cols, hR.nnz);
        
        printf("# idx  (blk_m blk_n blk_k  dim_part1 dim_part2 )  vers  nthread  nnz    batch_NB    X_block   Y_block   M    N    f  CPU YY^T Gflop/s (sec)    XX^T Gflop/s (sec)  GPU YY^T Gflop/s (sec)    XX^T Gflop/s (sec)  CPU YRY^T Gflop/s (sec)  GPU YRY^T Gflop/s (sec)  A1 error\n");
        printf("# ===================================================================================================================================\n");
        
        for( int itest=0; itest < fsize_tests; ++itest ) {
            //printf( "----------\n" );
            magma_int_t f = fsize[ itest ];
            magma_int_t ldf  = f;   // ldx, ldy, ldc all same, based on f
            magma_int_t lddf = magma_roundup( ldf, align );
            
            for( int ithread=0; ithread < nthread_tests; ++ithread ) {
                if ( threads[ ithread ] > 0 ) {
                    if ( verbose >= 1 ) { printf( "nthread %d\n", threads[ ithread ] ); }
                    omp_set_num_threads( threads[ ithread ] );
                }
                else {
                    #pragma omp parallel
                    {
                        threads[ ithread ] = omp_get_num_threads();
                    }
                }
                magma_int_t nthread = threads[ ithread ];
                
                // each thread needs own workspace
                float *hX, *hY, *hA, *hYYT, *hwork;
                float *hA1_cpu;
		float *hA1_magma;
		magma_index_t *oindex, *pindex, *offset, *offset_s;
		
                magmaFloat_ptr dX, dY, dA, dYYT;
                magmaFloat_ptr dwork;
		magmaIndex_ptr doindex, dpindex, doffset,doffset_s;
		magmaIndex_ptr doft, dofft;
                
                magma_int_t m = dR.num_rows;
                magma_int_t n = dR.num_cols;
		magma_int_t as = (m-1)/XBLOCK+1;
	    	magma_int_t sa = (n-1)/YBLOCK+1;
		magma_int_t off = as * sa;   //number of warp
                magma_int_t max_mn = max(m,n);
                batch_NB = min( batch_NB, max_mn );
                magma_int_t nbatch = magma_ceildiv( max(m,n), batch_NB );
        	if ( verbose >= 1 ) { printf( "nbatch=%d.\n",nbatch ); }        
                magma_int_t lwork = 2 * ldf * max( max_row_nnz, max_col_nnz );
                magma_int_t ldwork = max( f, max_mn );
                
                // allocate dense matrices
                alloc_time = magma_sync_wtime(0);
                if ( verbose >= 1 ) { printf( "allocate dense\n" ); }
		TESTING_CHECK( magma_index_malloc_cpu( &oindex, off+1));
		TESTING_CHECK( magma_index_malloc_cpu( &pindex, m * sa + 1 ));
		TESTING_CHECK( magma_index_malloc_cpu( &offset, hR.nnz));
		TESTING_CHECK( magma_index_malloc_cpu( &offset_s, hR.nnz ));
		
		if(verbose >= 1) { printf("Data Structure Tranfer\n"); }
		real_Double_t datastruc_time = magma_sync_wtime(0);	
		cal_off(batch_NB, m, n, off, hR.nnz, hR.row, hR.col, oindex, pindex, offset, offset_s, as, sa);
		real_Double_t to_time = magma_sync_wtime(0) - datastruc_time;
		if(verbose>=1) {printf("Data Structure Transfer Time: %11.6f s.\n", to_time);}
		
		TESTING_CHECK( magma_index_malloc( &doindex, off+1));
		TESTING_CHECK( magma_index_malloc( &dpindex, m * sa + 1 ));
		TESTING_CHECK( magma_index_malloc( &doffset, hR.nnz));
		TESTING_CHECK( magma_index_malloc( &doffset_s, oindex[off]));
		
		magma_index_setvector( off+1, oindex, 1 ,doindex, 1);
		magma_index_setvector( m*sa + 1, pindex, 1 ,dpindex, 1);
		magma_index_setvector( hR.nnz, offset, 1 ,doffset, 1);
		magma_index_setvector( oindex[off], offset_s, 1 ,doffset_s, 1);
		
		magma_index_t *reusenum_perbatch;	
		TESTING_CHECK( magma_index_malloc_cpu( &reusenum_perbatch, (m-1)/batch_NB+1 ));
		if ( verbose >= 1 ) { printf( "Order array according to data reuse time per batch\n" ); }
		real_Double_t reuse_time = magma_sync_wtime(0);
		Reuse_order(batch_NB, m, hR.row, oindex, as, sa, reusenum_perbatch);
		real_Double_t order_time = magma_sync_wtime(0) - reuse_time;
		if(verbose>=1) {printf("Reordering reuse array (reuse time per batch) Time: %11.6f s.\n", order_time);}
	
		magma_free_cpu( oindex );
		magma_free_cpu( pindex );
		magma_free_cpu( offset );
		magma_free_cpu( offset_s );

		real_Double_t  zals1_gflops = FLOPS_SSYRK( hR.nnz, f ) / 1e9;  // Y Ru Y^T
                real_Double_t  zals2_gflops = FLOPS_SSYRK( hR.nnz, f ) / 1e9;  // X Ri X^T
                real_Double_t ssyrk1_gflops = FLOPS_SSYRK( n, f )      / 1e9;  // Y Y^T
                real_Double_t ssyrk2_gflops = FLOPS_SSYRK( m, f )      / 1e9;  // X X^T
                real_Double_t gflops = ssyrk1_gflops + ssyrk2_gflops
                                     +  zals1_gflops +  zals2_gflops;
                if ( verbose >= 1 ) {
                    printf( "f:%d, Gflop count: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f = total %8.4f (per iteration)\n",
                            f, ssyrk1_gflops, ssyrk2_gflops,
                             zals1_gflops,  zals2_gflops,
                                   gflops );
                }

                //TESTING_CHECK( magma_smalloc_cpu( &hwork,     lwork * nthread));
                TESTING_CHECK( magma_smalloc_cpu( &hX,        ldf * m ));
                TESTING_CHECK( magma_smalloc_cpu( &hY,        ldf * n ));
                //TESTING_CHECK( magma_smalloc_cpu( &hA,        ldf * f * batch_NB* nthread ));
                //TESTING_CHECK( magma_smalloc_cpu( &hA1_cpu,   ldf * f * nbatch ));
                TESTING_CHECK( magma_smalloc_cpu( &hA1_magma, ldf * f * nbatch ));
                TESTING_CHECK( magma_smalloc_cpu( &hYYT,      ldf * f ));

                TESTING_CHECK( magma_smalloc( &dX,    lddf * m ));
                TESTING_CHECK( magma_smalloc( &dY,    lddf * n ));
                TESTING_CHECK( magma_smalloc( &dA,    lddf * f * batch_NB ));
                TESTING_CHECK( magma_smalloc( &dYYT,  lddf * f ));
                TESTING_CHECK( magma_smalloc( &dwork, ldwork ));
		
		alloc_time = magma_sync_wtime(0) - alloc_time;               
 
                // print header for this matrix
                size_t Rsize  = ((dR.num_rows+1  + dR.nnz)  * sizeof(magma_index_t) + dR.nnz   * sizeof(float));
                size_t RTsize = ((dRT.num_rows+1 + dRT.nnz) * sizeof(magma_index_t) + dRT.nnz  * sizeof(float));
                size_t Xsize  = lddf*m          * sizeof(float);
                size_t Ysize  = lddf*n          * sizeof(float);
                size_t Asize  = lddf*f*batch_NB * sizeof(float);
                float meg    = 1024.*1024.;
                if ( verbose >= 1 ) 
		{
                    printf( "R %d by %d, nnz %d, f %d, %.2f MiB, RT %.2f MiB, 2X %.2f MiB, 2Y %.2f MiB, A %.2f MiB, total %.2f MiB\n",
                            m, n, dR.nnz, f,
                            Rsize/meg, RTsize/meg, 2*Xsize/meg, 2*Ysize/meg, Asize/meg,
                            (Rsize + RTsize + 2*Xsize + 2*Ysize + Asize)/meg ); 
                    printf( "setup time: read %8.4f, transfer %8.4f, transpose %8.4f, transfer %8.4f, allocate %8.4f\n",
                            read_time, setup_time1, setup_time2, setup_time3, alloc_time );
                }
                
                // count flops
         /*       real_Double_t  zals1_gflops = FLOPS_SSYRK( hR.nnz, f ) / 1e9;  // Y Ru Y^T
                real_Double_t  zals2_gflops = FLOPS_SSYRK( hR.nnz, f ) / 1e9;  // X Ri X^T
                real_Double_t ssyrk1_gflops = FLOPS_SSYRK( n, f )      / 1e9;  // Y Y^T
                real_Double_t ssyrk2_gflops = FLOPS_SSYRK( m, f )      / 1e9;  // X X^T
                real_Double_t gflops = ssyrk1_gflops + ssyrk2_gflops
                                     +  zals1_gflops +  zals2_gflops;
                if ( verbose >= 1 ) {
                    printf( "Gflop count: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f = total %8.4f (per iteration)\n",
                            ssyrk1_gflops, ssyrk2_gflops,
                             zals1_gflops,  zals2_gflops,
                                   gflops );
                }	
	*/
                for( int iter=0; iter < niter; ++iter ) {
                    // random initial guess in Y matrix
                    if ( verbose >= 1 ) { printf( "init Y\n" ); }
                    size = ldf * n;
                    lapackf77_slarnv( &ione, ISEED, &size, hY );
                    size = ldf * m;
                    lapackf77_slarnv( &ione, ISEED, &size, hX );
        	               
                    magma_ssetmatrix( f, m, hX, ldf, dX, lddf );
                    magma_ssetmatrix( f, n, hY, ldf, dY, lddf );
                    
                    /* ====================================================================
                       Performs operation using MAGMA
                       =================================================================== */
                    if ( verbose >= 1 ) { printf( "\nmagma\n" ); }
                    fflush(0);
                    magmablas_slaset( MagmaFull, f, f*batch_NB, c_zero, c_zero, dA,   lddf );  // A   = 0
                    magmablas_slaset( MagmaFull, f, f,          c_zero, c_zero, dYYT, lddf );  // YYT = 0
                    
                    overhead_time = 0;
                    gpu_ssyrk1_time  = 0;
                    gpu_ssyrk2_time  = 0;
                    gpu_zals1_time   = 0;
                    gpu_zals2_time   = 0;
                    gpu_time = magma_sync_wtime(0);
                    if ( first ) {
                        // --------------------------------------------------
                        if ( verbose >= 1 ) { printf( "magma: Au = Y Ru Y^T\n" ); }
                        
                        // Y*Y^T
                        if ( verbose >= 1 ) { printf( "ssyrk\n" ); }
                        time = magma_sync_wtime(0);
                        magma_ssyrk( MagmaLower, MagmaNoTrans, f, n, d_one, dY, lddf, d_zero, dYYT, lddf );
                        gpu_ssyrk1_time += magma_sync_wtime(0) - time;
                        
                        // scale so YYT does not dominate ALS, to make ALS errors discernable
                        float AAnorm = magmablas_slansy( MagmaMaxNorm, MagmaLower, f, dYYT, lddf, dwork, ldwork );
                        size = lddf*f;
                        AAnorm = 1. / AAnorm;
                        magma_sscal( size, AAnorm, dYYT, ione );

			printf("a.\n");		

			drowptr = dR.drow; 
                        for( int i=0; i < m; i += batch_NB ) 
			{
				int NUM = 0;
                            	int batch_IB = min( batch_NB, m-i );
			    	int th_size = (batch_IB-1)/XBLOCK+1;
			    	int pos = 0;
			    	if(i!=0)
			    	{
					pos = sa*((i-1)/XBLOCK+1);
			    	}
			    
			    	int new_row = XBLOCK;
			        dR.drow = drowptr + i;
                         
				for(int ll = 0; ll < P1BATCH; ll++)
				{
					if(i/batch_NB == reusenum_perbatch[ll])
					{
						part1 = magma_sync_wtime(0);
                            			magma_sals1( m, batch_NB, pos, f, batch_IB, n, f, n,  dY, lddf, f, f*batch_IB, dA, lddf, doffset_s, doffset, doindex, dpindex, th_size, sa, new_row, queue);
						part3 += magma_sync_wtime(0) - part1;
						NUM += 1;
						break;
					}
				}
				if(NUM == 0)
				{
					part2 = magma_sync_wtime(0);
					magma_sals2( f, batch_IB, n, f, n, dY, lddf, dR, f, f*batch_IB, dA, lddf, queue);
					part4 += magma_sync_wtime(0) - part2;
				}	
                            	time = magma_sync_wtime(0);
				magma_sgetmatrix( f, f, dA, lddf, hA1_magma + (i/batch_NB)*ldf*f, ldf );
				overhead_time += magma_sync_wtime(0) - time;    	
                        }
			gpu_zals1_time = part3 + part4;

                    }
                    if ( second ) {
/*                       // --------------------------------------------------
                        if ( verbose >= 1 ) { printf( "magma: Ai = X Ri X^T\n" ); }
                        
                        // X*X^T
                        if ( verbose >= 1 ) { printf( "ssyrk\n" ); }
                        time = magma_sync_wtime(0);
                        magma_ssyrk( MagmaLower, MagmaNoTrans, f, m, d_one, dX, lddf, d_zero, dYYT, lddf );
                        gpu_ssyrk2_time += magma_sync_wtime(0) - time;
                        
                        // scale so XXT does not dominate ALS, to make ALS errors discernable
                        float AAnorm = magmablas_slansy( MagmaMaxNorm, MagmaLower, f, dYYT, lddf, dwork, ldwork );
                        size = lddf*f;
                        AAnorm = 1. / AAnorm;
                        magma_sscal( size, AAnorm, dYYT, ione );
                        
                        // save and adjust dR rowptr to allow small batches in magma_sals2
                        drowptr = dRT.drow;
                        for( int i=0; i < n; i += batch_NB ) {
                            int batch_IB = min( batch_NB, n-i );
                            if ( verbose >= 1 ) { printf( "zals: X Ri X^T\n" ); }
                            dRT.drow = drowptr + i;
                            time = magma_sync_wtime(0);
                            magma_sals( f, batch_IB, m,
                                         f, m,  dX,   lddf,
                                         alpha, dRT,
                                         f, f,  dYYT, lddf,  // i.e., XX^T
                                         lambda,
                                         f, f*batch_IB, dA,          lddf,
                                         f,   batch_IB, dY + i*lddf, lddf,
                                         queue );
                            gpu_zals2_time += magma_sync_wtime(0) - time;
                            
                            // save last matrix in each batch to verify
                            time = magma_sync_wtime(0);
                            magma_sgetmatrix( f, f, dA + (batch_IB-1)*lddf*f, lddf, hA2_magma + (i/batch_NB)*ldf*f, ldf );
                            overhead_time += magma_sync_wtime(0) - time;
                        }
                        dRT.drow = drowptr;
*/
                    }
                    gpu_time = magma_sync_wtime(0) - gpu_time;
                    sum_time = gpu_ssyrk1_time + gpu_ssyrk2_time + gpu_zals1_time + gpu_zals2_time + overhead_time;
                    if ( verbose >= 1 ) { printf( "magma done\n" ); }
                    if ( verbose >= 1 ) {
                        printf( "gpu seconds: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f = sum %8.4f, total %8.4f\n",
                                gpu_ssyrk1_time, gpu_ssyrk2_time,
                                gpu_zals1_time,  gpu_zals2_time,
                                sum_time, gpu_time );
                        printf( "gpu Gflop/s: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f, total %8.4f\n",
                                ssyrk1_gflops / gpu_ssyrk1_time,
                                ssyrk2_gflops / gpu_ssyrk2_time,
                                 zals1_gflops / gpu_zals1_time,
                                 zals2_gflops / gpu_zals2_time,
                                       gflops / gpu_time );
                    }
                    // warn if sum > 2% different than total -- something isn't getting timed
                    if ( verbose >= 1 && fabs(gpu_time - sum_time) / gpu_time > 0.02 ) {
                        printf( "WARNING: GPU sum %10.6f != total time %10.6f\n", sum_time, gpu_time );
                    }
                    fflush(0);
                    
                    /* ====================================================================
                       Performs operation on CPU
                       =================================================================== */
                    if ( verbose >= 1 ) { printf( "\ncpu\n" ); }
                    fflush(0);
                    //size = f*nthread;  // only need one matrix workspace per thread
                    size = f * batch_NB * nthread;
                    lapackf77_slaset( "Full", &f, &size, &c_zero, &c_zero, hA,   &ldf );  // A   = 0
                    lapackf77_slaset( "Full", &f, &f,    &c_zero, &c_zero, hYYT, &ldf );  // YYT = 0
                    
                    magma_int_t lapack_nthread = magma_get_lapack_numthreads();
                    
                    // for convenience checking results, CPU and GPU now use same batch_NB
                    // // divide into at least 4 tasks per thread, up to batch_NB (4000) in each task
                    // cpu_batch_NB = min( batch_NB, max( 1, min(m,n) / (4*nthread) ));
                    if ( verbose >= 1 ) 
		    {
                         printf( "CPU nthread %d, cpu_batch_NB %d\n", nthread, batch_NB );
                    }
                    
                    cpu_ssyrk1_time  = 0;
                    cpu_ssyrk2_time  = 0;
                    cpu_zals1_time   = 0;
                    cpu_zals2_time   = 0;
                    cpu_time = magma_wtime();
                    if ( first ) {
                        // --------------------------------------------------
                        if ( verbose >= 1 ) { printf( "cpu: Au = Y Ru Y^T\n" ); }
                        
                        // Y*Y^T
                        time = magma_wtime();
                        blasf77_ssyrk( "Lower", "NoTrans", &f, &n, &d_one, hY, &ldf, &d_zero, hYYT, &ldf );
                        cpu_ssyrk1_time += magma_wtime() - time;
                        
                        // scale so YYT does not dominate ALS, to make ALS errors discernable
                        float AAnorm = lapackf77_slansy( "M", "L", &f, hYYT, &ldf, work );
                        size = ldf*f;
                        AAnorm = 1. / AAnorm;
                        blasf77_sscal( &size, &AAnorm, hYYT, &ione );
                        
                        #pragma omp parallel
                        {
                            magma_set_lapack_numthreads( 2 );
                        }
                        time = magma_wtime();
                        #pragma omp parallel for
                        for( int i=0; i < m; i += batch_NB ) {
                            int batch_IB = min( batch_NB, m-i );  // local!
                            int tid = omp_get_thread_num();
                            if ( verbose >= 1 ) { printf( "zals: Y Ru Y^T i %d, tid %d\n", i, tid ); }
                            switch( cpu_version ) {
                            case 1:
                                testing_sals1( f, batch_IB, n,
                                               f, n,  hY, ldf,
                                               alpha, hR.row+i, hR.col, hR.val,
                                               f, f,  hYYT, ldf,
                                               lambda,
                                               f, f,        hA + tid*batch_NB*ldf*f, ldf,
                                               f, batch_IB, hX + i*ldf,     ldf );
                                break;
                            
                            case 2:
                                testing_sals2( f, batch_IB, n,
                                               f, n,  hY, ldf,
                                               alpha, hR.row+i, hR.col, hR.val,
                                               f, f,  hYYT, ldf,
                                               lambda,
                                               f, f,        hA + tid*ldf*f, ldf,
                                               f, batch_IB, hX + i*ldf,     ldf,
                                               &hwork[ lwork * tid ], lwork );
                                break;
                            
                            case 3:
                                testing_sals( f, batch_IB, n,
                                               f, n,  hY, ldf,
                                               alpha, hR.row+i, hR.col, hR.val,
                                               f, f,  hYYT, ldf,
                                               lambda,
                                               f, f,        hA + tid*ldf*f, ldf,
                                               f, batch_IB, hX + i*ldf,     ldf,
                                               &hwork[ lwork * tid ], lwork );
                                break;
                            
                            default:
                                fprintf( stderr, "unknown version %d\n", cpu_version );
                                exit(1);
                                break;
                            }
                            // save last matrix in each batch to verify
                            lapackf77_slacpy( "full", &f, &f, hA + tid*batch_NB*ldf*f, &ldf, hA1_cpu + (i/batch_NB)*ldf*f, &ldf );
                        }
                        cpu_zals1_time += magma_wtime() - time;
                        #pragma omp parallel
                        {
                            magma_set_lapack_numthreads( lapack_nthread );
                        }
                    }
                    if ( second ) {
                        // --------------------------------------------------
                        if ( verbose >= 1 ) { printf( "cpu: Ai = X Ri X^T\n" ); }
                        
                        // X*X^T
                        time = magma_wtime();
                        blasf77_ssyrk( "Lower", "NoTrans", &f, &m, &d_one, hX, &ldf, &d_zero, hYYT, &ldf );
                        cpu_ssyrk2_time += magma_wtime() - time;
                        
                        // scale so XXT does not dominate ALS, to make ALS errors discernable
                        float AAnorm = lapackf77_slansy( "M", "L", &f, hYYT, &ldf, work );
                        size = ldf*f;
                        AAnorm = 1. / AAnorm;
                        blasf77_sscal( &size, &AAnorm, hYYT, &ione );
                        
                        #pragma omp parallel
                        {
                            magma_set_lapack_numthreads( 1 );
                        }
                        time = magma_wtime();
                        #pragma omp parallel for
                        for( int i=0; i < n; i += batch_NB ) {
                            int batch_IB = min( batch_NB, n-i );  // local!
                            int tid = omp_get_thread_num();
                            if ( verbose >= 1 ) { printf( "zals: Y Ru Y^T i %d, tid %d\n", i, tid ); }
                            switch( cpu_version ) {
                            case 1:
                                testing_sals1( f, batch_IB, m,
                                               f, m,  hX, ldf,
                                               alpha, hRT.row+i, hRT.col, hRT.val,
                                               f, f,  hYYT, ldf,  // i.e., XX^T
                                               lambda,
                                               f, f,        hA + tid*ldf*f, ldf,
                                               f, batch_IB, hY + i*ldf,     ldf );
                                break;
                            
                            case 2:
                                testing_sals2( f, batch_IB, m,
                                               f, m,  hX, ldf,
                                               alpha, hRT.row+i, hRT.col, hRT.val,
                                               f, f,  hYYT, ldf,  // i.e., XX^T
                                               lambda,
                                               f, f,        hA + tid*ldf*f, ldf,
                                               f, batch_IB, hY + i*ldf,     ldf,
                                               &hwork[ lwork * tid ], lwork );
                                break;
                            
                            case 3:
                                testing_sals( f, batch_IB, m,
                                               f, m,  hX, ldf,
                                               alpha, hRT.row+i, hRT.col, hRT.val,
                                               f, f,  hYYT, ldf,  // i.e., XX^T
                                               lambda,
                                               f, f,        hA + tid*ldf*f, ldf,
                                               f, batch_IB, hY + i*ldf,     ldf,
                                               &hwork[ lwork * tid ], lwork );
                                break;
                            
                            default:
                                fprintf( stderr, "unknown version %d\n", cpu_version );
                                exit(1);
                                break;
                            }
                            // save first matrix in each batch to verify
		            lapackf77_slacpy( "full", &f, &f, hA + tid*batch_NB*ldf*f, &ldf, hA1_cpu + (i/batch_NB)*ldf*f, &ldf );
                        }
                        cpu_zals2_time += magma_wtime() - time;
                        #pragma omp parallel
                        {
                            magma_set_lapack_numthreads( lapack_nthread );
                        }
                    }
                    cpu_time = magma_wtime() - cpu_time;
                    sum_time = cpu_ssyrk1_time + cpu_ssyrk2_time + cpu_zals1_time + cpu_zals2_time;
                    if ( verbose >= 1 ) { printf( "cpu done\n" ); }
                    if ( verbose >= 1 ) {
                        printf( "cpu seconds: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f = sum %8.4f, total %8.4f\n",
                                cpu_ssyrk1_time, cpu_ssyrk2_time,
                                cpu_zals1_time,  cpu_zals2_time,
                                sum_time, cpu_time );
                        printf( "cpu Gflop/s: ssyrk %8.4f, %8.4f, zals %8.4f, %8.4f, total %8.4f\n",
                                ssyrk1_gflops / cpu_ssyrk1_time,
                                ssyrk2_gflops / cpu_ssyrk2_time,
                                 zals1_gflops / cpu_zals1_time,
                                 zals2_gflops / cpu_zals2_time,
                                       gflops / cpu_time );
                    }
                    // warn if sum > 2% different than total -- something isn't getting timed
                    if ( verbose >= 1 && fabs(cpu_time - sum_time) / cpu_time > 0.02 ) {
                        printf( "WARNING: CPU sum %10.6f != total time %10.6f\n", sum_time, cpu_time );
                    }
                    fflush(0);
           
                    /* =====================================================================
                       Check the result compared to CPU
                       =================================================================== */
                    //printf( "hA1_cpu   = " );  magma_sprint( f, nbatch*f, hA1_cpu,   ldf );
                    //printf( "hA1_magma = " );  magma_sprint( f, nbatch*f, hA1_magma, ldf );
                    float A1norm, errorA1;
                    size = ldf * f * magma_ceildiv(m,batch_NB);
                    blasf77_saxpy( &size, &c_neg_one, hA1_cpu, &ione, hA1_magma, &ione );
                    size =       f * magma_ceildiv(m,batch_NB);
                    A1norm   = lapackf77_slange( "M", &f, &size, hA1_cpu  , &ldf, work );
                    errorA1  = lapackf77_slange( "M", &f, &size, hA1_magma, &ldf, work );
                    errorA1 /= A1norm;
                    //printf( "hA1_diff  = " );  magma_sprint( f, nbatch*f, hA1_magma, ldf );
                    
/*                  //printf( "hA2_cpu   = " );  magma_sprint( f, nbatch*f, hA2_cpu,   ldf );
                    //printf( "hA2_magma = " );  magma_sprint( f, nbatch*f, hA2_magma, ldf );
                    float A2norm, errorA2;
                    size = ldf*f * magma_ceildiv(n,batch_NB);
                    blasf77_saxpy( &size, &c_neg_one, hA2_cpu, &ione, hA2_magma, &ione );
                    size =     f * magma_ceildiv(n,batch_NB);
                    A2norm   = lapackf77_slange( "M", &f, &size, hA2_cpu  , &ldf, work );
                    errorA2  = lapackf77_slange( "M", &f, &size, hA2_magma, &ldf, work );
                    //printf( "A2 norm %.4e, error %.4e, rel error %.4e\n", A2norm, errorA2, errorA2/A2norm );
                    errorA2 /= A2norm;
                    //printf( "hA2_diff  = " );  magma_sprint( f, nbatch*f, hA2_magma, ldf );
*/                    
              //      magma_sals_print_index();
              //      vers thr  nnz  m    n    f    ssyrk1          ssyrk2          ssyrk1          ssyrk2          zals1           zals2           zals1           zals2           err1   err2   ok
                    //printf( "%4d  %3d  %7d  %7d  %7d   %7d   %7d  %7d  %3d  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %8.2e  %8.2e  %s\n",
		    printf( "%4d  %3d  %7d  %7d  %7d   %7d   %7d  %7d  %3d  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %7.2f (%11.6f)  %11.6f %11.6f %8.2e  %s\n",
			    (int) cpu_version,
                            (int) nthread, hR.nnz, batch_NB, XBLOCK, YBLOCK, (int) m, (int) n, (int) f,
                            
                            (ssyrk1_gflops / cpu_ssyrk1_time), cpu_ssyrk1_time,
                            (ssyrk2_gflops / cpu_ssyrk2_time), cpu_ssyrk2_time,
                            
                            (ssyrk1_gflops / gpu_ssyrk1_time), gpu_ssyrk1_time,
                            (ssyrk2_gflops / gpu_ssyrk2_time), gpu_ssyrk2_time,
                            
                            (zals1_gflops / cpu_zals1_time), cpu_zals1_time,
                            //(zals2_gflops / cpu_zals2_time), cpu_zals2_time,
                            
                            (zals1_gflops / gpu_zals1_time), gpu_zals1_time,
                            //(zals2_gflops / gpu_zals2_time), gpu_zals2_time,

			    part3, part4,
                            
                            errorA1,
			    // errorA2, 
			    (errorA1 < tol ? "ok" : "failed"));

                    fflush(0); 

                }  // end iter
                if ( niter > 1 ) {
                    printf( "\n" );
                }
                
                // free dense matrices
	//	magma_free_cpu( hwork );
                magma_free_cpu( hX   );
                magma_free_cpu( hY   );
	  //    magma_free_cpu( hA   );
          //    magma_free_cpu( hA1_cpu );
	        magma_free_cpu( hA1_magma );
	        magma_free_cpu( hYYT );
		magma_free_cpu(reusenum_perbatch);
	        magma_free( dX    );
	        magma_free( dY    );
	        magma_free( dA    );
		magma_free( dYYT  );
		magma_free( dwork );
		magma_free( doindex );
		magma_free( dpindex );
		magma_free( doffset );
		magma_free( doffset_s );              
            }  // end ithread
            if ( nthread_tests > 1 ) {
                printf( "# -----\n" );
            }
        }  // end itest
        
        // free sparse matrices
        magma_smfree( &hR,  queue );
        magma_smfree( &hRT, queue );
        magma_smfree( &dR,  queue );
        magma_smfree( &dRT, queue );
    }  // end ifile   
    magma_queue_destroy( queue );
    magma_finalize();
    return 0;
}
// end main
