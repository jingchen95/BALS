/*
    Copyright 2009-2016, University of Tennesse. See COPYRIGHT file.
    
    @author Hartwig Anzt
    @author Mark Gates
    @author Jakub Kurzak
*/

#ifndef TRAITS_H
#define TRAITS_H

#ifndef CUBLAS_V2_H_
#include <cublas.h>
#endif


// ------------------------------------------------------------
// additional struct for vector float-complex.
typedef struct {
    cuFloatComplex x, y;
} cuFloatComplex2;



// ------------------------------------------------------------
// scalar_traits<type> encapsulates information and types specific to each scalar type:
// float, double, float-complex, double-complex.
// value_type    is the actual type, e.g., cuFloatComplex.
// base_type     is the type of each companent, e.g., float for cuFloatComplex.
// is_real       is boolean flag whether type is real (true) or complex (false).
// flops_per_fma is number of multiplies and adds per fused multiply add (fma), e.g., 8 for double-complex.
// prefix        is BLAS/LAPACK routine prefix (s=float, d=double, c=float-complex, z=double-complex).

// --------------------
template< typename scalar_t >
class scalar_traits;

// --------------------
template<>
class scalar_traits< float > {
public:
    typedef float       value_type;
    typedef float       base_type;
    
    static const bool  is_real       = true;
    static const int   flops_per_fma = 2;  // 1 mul + 1 add
    static const char  prefix        = 's';
};

// --------------------
template<>
class scalar_traits< double > {
public:
    typedef double       value_type;
    typedef double       base_type;
    
    static const bool  is_real       = true;
    static const int   flops_per_fma = 2;  // 1 mul + 1 add
    static const char  prefix        = 'd';
};

// --------------------
template<>
class scalar_traits< cuFloatComplex > {
public:
    typedef cuFloatComplex value_type;
    typedef float          base_type;
    
    static const bool  is_real       = false;
    static const int   flops_per_fma = 8;  // 6 mul + 2 add
    static const char  prefix        = 'c';
};

// --------------------
template<>
class scalar_traits< cuDoubleComplex > {
public:
    typedef cuDoubleComplex value_type;
    typedef double          base_type;
    
    static const bool  is_real       = false;
    static const int   flops_per_fma = 8;  // 6 mul + 2 add
    static const char  prefix        = 'z';
};



// ------------------------------------------------------------
// vector_traits< scalar_type, dim_vec > encapsulates information and types
// specific to each vector type:
// float, float2, float4, double, double2, float-complex, float-complex2, double-complex.
// For consistent code, a scalar is a vector of length 1.
// vector_type    is the vector type.
// array_type     is the vector/array union. [TODO]
// channel_[xyzw] are sizes for cudaArray channels.
template< typename scalar_t, int dim_vec >
class vector_traits;

template<>
class vector_traits< float, 1 > {
public:
    typedef float       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 0;
    static const int channel_z = 0;
    static const int channel_w = 0;
};

// --------------------
template<>
class vector_traits< float, 2 > {
public:
    typedef float2       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 0;
    static const int channel_w = 0;
};

// --------------------
template<>
class vector_traits< float, 4 > {
public:
    typedef float4       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 32;
    static const int channel_w = 32;
};

// --------------------
template<>
class vector_traits< double, 1 > {
public:
    typedef double       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 0;
    static const int channel_w = 0;
};

// --------------------
template<>
class vector_traits< double, 2 > {
public:
    typedef double2       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 32;
    static const int channel_w = 32;
};

// --------------------
template<>
class vector_traits< cuFloatComplex, 1 > {
public:
    typedef cuFloatComplex       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 0;
    static const int channel_w = 0;
};

// --------------------
template<>
class vector_traits< cuFloatComplex, 2 > {
public:
    typedef cuFloatComplex2       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 32;
    static const int channel_w = 32;
};

// --------------------
template<>
class vector_traits< cuDoubleComplex, 1 > {
public:
    typedef cuDoubleComplex       vector_type;
    
    static const int channel_x = 32;
    static const int channel_y = 32;
    static const int channel_z = 32;
    static const int channel_w = 32;
};



// ------------------------------------------------------------
// define memory type as either pointer or texture.

template< typename vector_t, bool tex >
class memory_traits;

// partial specialization for 1D textures
template< typename vector_t >
class memory_traits< vector_t, true > {
public:
    typedef cudaTextureObject_t object_type;
    typedef cudaTextureObject_t object_const_type;
    typedef cudaTextureObject_t object_const_restrict_type;
    typedef cudaTextureObject_t object_restrict_type;
};

// partial specialization for pointers
template< typename vector_t >
class memory_traits< vector_t, false > {
public:
    typedef vector_t*                     object_type;
    typedef vector_t const*               object_const_type;
    typedef vector_t const* __restrict__  object_const_restrict_type;
    typedef vector_t      * __restrict__  object_restrict_type;
};



// ------------------------------------------------------------
// make<type>(a,b) returns (a + bi) for complex types or (a) for real types.

// generic version for float, double
template< typename scalar_t >
__host__ __device__ inline
scalar_t make( typename scalar_traits<scalar_t>::base_type a,
               typename scalar_traits<scalar_t>::base_type b )
{
    return a;
}

// specialization for cuFloatComplex
template<>
__host__ __device__ inline
cuFloatComplex make( float a, float b )
{
    return make_cuFloatComplex(a,b);
}

// specialization for cuDoubleComplex
template<>
__host__ __device__ inline
cuDoubleComplex make( double a, double b )
{
    return make_cuDoubleComplex(a,b);
}



// ------------------------------------------------------------
// fetch( A, index ) returns A[index]
// works for A as texture or pointer, for float, float4, double, float2/complex-float, double2/complex-double.

// ---------------- for 1D textures
// generic version for float, float4, float2/cuFloatComplex
template< typename vector_t >
__device__ inline
vector_t fetch( cudaTextureObject_t tex, int index )
{
    return tex1Dfetch<vector_t>( tex, index );
}

// specialization for cuFloatComplex2 (essentially same as float4)
template<>
__device__ inline
cuFloatComplex2 fetch( cudaTextureObject_t tex, int index )
{
    float4 x = tex1Dfetch<float4>( tex, index );
    cuFloatComplex2 y = {{ x.x, x.y }, { x.z, x.w }};
    return y;
}

// specialization for double
template<>
__device__ inline
double fetch( cudaTextureObject_t tex, int index )
{
    int2 v = tex1Dfetch<int2>( tex, index );
    return __hiloint2double( v.y, v.x );
}

// specialization for double2/cuDoubleComplex
template<>
__device__ inline
cuDoubleComplex fetch( cudaTextureObject_t tex, int index )
{
    int4 v = tex1Dfetch<int4>( tex, index );
    return make<cuDoubleComplex>( __hiloint2double( v.y, v.x ),
                                  __hiloint2double( v.w, v.z ) );
}

// ---------------- for arrays
// generic version for float, float2/cuFloatComplex, float4, double, double2/cuDoubleComplex
template< typename vector_t >
__device__ inline
vector_t fetch( const vector_t * __restrict__ A, int index )
{
    return A[index];
}


// ------------------------------------------------------------
// Functions to setup & teardown pointers & textures.
// For pointers, just sets objA = A.

// --------------------
// version for regular pointers
// "A" must be const here to match beast_gemm( ..., float const* A, ... )
// TODO what about un-aligned arrays?
// TODO define scalar_t from scalar_type of vector_t.
template< typename scalar_t >
void setup_memory(
    magma_int_t m, magma_int_t n,
    scalar_t const* A, magma_int_t lda,
    scalar_t const* __restrict__ * objA )
{
    //static bool first = true;
    //if ( first ) {
    //    printf( "%s pointer\n", __func__ );
    //    first = false;
    //}
    
    *objA = (scalar_t const*) A;
}

template< typename scalar_t >
void teardown_memory( scalar_t const* __restrict__ objA )
{
    // pass: nothing to do
}

// --------------------
// version for textures
// "A" must be const here to match beast_gemm( ..., float const* A, ... )
template< typename scalar_t >
void setup_memory(
    magma_int_t m, magma_int_t n,
    scalar_t const* A, magma_int_t lda,
    cudaTextureObject_t* objA )
{
    //static bool first = true;
    //if ( first ) {
    //    printf( "%s texture\n", __func__ );
    //    first = false;
    //}
    
    // Create channel.
    cudaChannelFormatDesc channel_desc;
    channel_desc = cudaCreateChannelDesc(
        vector_traits< scalar_t, 1 >::channel_x,
        vector_traits< scalar_t, 1 >::channel_y,
        vector_traits< scalar_t, 1 >::channel_z,
        vector_traits< scalar_t, 1 >::channel_w,
        cudaChannelFormatKindSigned );
    
    // Specify texture object parameters.
    struct cudaTextureDesc texDesc;
    memset( &texDesc, 0, sizeof(texDesc) );
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModePoint;
    texDesc.readMode       = cudaReadModeElementType;

    // Create resource descriptor.
    struct cudaResourceDesc resDescA;
    memset( &resDescA, 0, sizeof(resDescA) );
    resDescA.resType                = cudaResourceTypeLinear;
    resDescA.res.linear.devPtr      = (void*)A;
    resDescA.res.linear.desc        = channel_desc;
    resDescA.res.linear.sizeInBytes = (size_t)lda*n * sizeof(scalar_t);
    
    // Create texture object.
    cudaCreateTextureObject( objA, &resDescA, &texDesc, NULL );
}

static inline
void teardown_memory( cudaTextureObject_t objA )
{
    cudaDestroyTextureObject( objA );
}

#endif        //  #ifndef TRAITS_H
