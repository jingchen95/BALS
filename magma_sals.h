#ifndef MAGMA_SALS_H
#define MAGMA_SALS_H

#include <magma.h>

extern "C" magma_int_t
magma_sals1(
    magma_int_t rows,
    magma_int_t batch_NB,
    magma_int_t i,
    magma_int_t f,
    magma_int_t m,
    magma_int_t n,
    magma_int_t Am,
    magma_int_t An,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magma_int_t Cm,
    magma_int_t Cn,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaIndex_ptr doffset,magmaIndex_ptr doffset_s,
    magmaIndex_ptr doindex,magmaIndex_ptr dpindex,
    magma_int_t threadblock_size,
    magma_int_t sa,
    magma_int_t new_row,
    magma_queue_t queue );


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
    	magma_queue_t queue );

extern "C" void
magma_sals_print_index();

#endif        //  #ifndef MAGMA_SALS_H
