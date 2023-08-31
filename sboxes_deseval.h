#pragma once

#ifndef DESEVAL_H
#define DESEVAL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__NVCC__) && !defined(__CUDACC__)
#define __device__
#define __host__
#define __inline__
#define __forceinline__
#include <math.h>
#define log2(x) (floor(log10((float)(x))/log10(2.0f)+0.5))
#else /* !defined(__NVCC__) && !defined(__CUDACC__) */
#define log2(x) (64-__clzll((x)-1))
#endif /* !defined(__NVCC__) && !defined(__CUDACC__) */

#define bitsizeof(x)          (sizeof(x)<<3)
#define bitcountof(x)         (bitsizeof(x)==64?6:(bitsizeof(x)==32?5:((size_t)(log2(bitsizeof(x))+0.5))))

#if defined(WIN32) && !defined(__NVCC__)
/* Windows / Linux */
#if 64
typedef unsigned __int64      sboxes_bitslice;
#else /* 32 */
typedef unsigned long         sboxes_bitslice;
#endif /* 32 */
typedef unsigned long long    sboxes_deskey;
typedef unsigned long long    sboxes_block;

#else /* defined(WIN32) && !defined(__NVCC__) */
/* NVIDIA */
#if 0
/* 128bit slice */
typedef uint128_t            sboxes_bitslice;
#define sboxes_slicebitcount  7
#define sboxes_slicebitsize   128
#define sboxes_sliceoff       ((sboxes_bitslice)0ULLL)
#define sboxes_sliceon        ((sboxes_bitslice)~0ULLL)
#elif 0
/* 64bit slice */
typedef uint64_t             sboxes_bitslice;
#define sboxes_slicebitcount  6
#define sboxes_slicebitsize   64
#define sboxes_sliceoff       ((sboxes_bitslice)0ULL)
#define sboxes_sliceon        ((sboxes_bitslice)~0ULL)
#else
/* 32bit slice */
typedef uint32_t             sboxes_bitslice;
#define sboxes_slicebitcount  5
#define sboxes_slicebitsize   32
#define sboxes_sliceoff       ((sboxes_bitslice)0UL)
#define sboxes_sliceon        ((sboxes_bitslice)~0UL)
#endif /* 0 */
typedef uint64_t              sboxes_deskey;
typedef uint64_t              sboxes_block;
#endif /* defined(WIN32) && !defined(__NVCC__) */

#ifndef sboxes_slicebitcount
#define sboxes_slicebitcount  (bitcountof(sboxes_bitslice))
#endif /* sboxes_slicebitcount */
#ifndef sboxes_slicebitsize
#define sboxes_slicebitsize   (bitsizeof(sboxes_bitslice))
#endif /* sboxes_slicebitsize */
#ifndef sboxes_sliceoff
#define sboxes_sliceoff       ((sboxes_bitslice)0UL)
#endif /* sboxes_sliceoff */
#ifndef sboxes_sliceon
#define sboxes_sliceon        (~(sboxes_sliceoff))
#endif /* sboxes_sliceon */

#define sboxes_char2key(key)  ((sboxes_deskey)((((((((((((((( \
                              (sboxes_deskey)(key)[0])<<8) | (sboxes_deskey)(key)[1])<<8) | (sboxes_deskey)(key)[2])<<8) | (sboxes_deskey)(key)[3])<<8) | \
                              (sboxes_deskey)(key)[4])<<8) | (sboxes_deskey)(key)[5])<<8) | (sboxes_deskey)(key)[6])<<8) | (sboxes_deskey)(key)[7])<<0)

/* External functions */

extern __device__ __host__ __forceinline__ void            sboxes_desslice_tdec(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k1h)[56], const sboxes_bitslice (&k2l)[56]);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_tenc(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k1h)[56], const sboxes_bitslice (&k2l)[56]);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_dec(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k)[56]);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_enc(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k)[56]);
extern __device__ __host__ __forceinline__ sboxes_bitslice sboxes_desslice_eval(sboxes_bitslice (&t)[64], const sboxes_bitslice (&c)[64]);
extern __device__ __host__ __forceinline__ sboxes_bitslice sboxes_desslice_eval64(sboxes_bitslice (&t)[64], const sboxes_block c);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_set(sboxes_bitslice (&t)[64], const sboxes_bitslice (&p)[64]);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_set64(sboxes_bitslice (&t)[64], const sboxes_block p64);
extern __device__ __host__ __forceinline__ void            sboxes_desslice_get(sboxes_bitslice (&c)[64], const sboxes_bitslice (&t)[64]);
extern __device__ __host__ __forceinline__ sboxes_block    sboxes_desslice_get64(const sboxes_bitslice (&t)[64]);
extern __device__ __host__ __forceinline__ sboxes_bitslice sboxes_deseval(const sboxes_bitslice (&p)[64], const sboxes_bitslice (&c)[64], const sboxes_bitslice (&k)[56]);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_clr_parity(const sboxes_deskey val64);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_set_parity(const sboxes_deskey val64);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_key_found(const sboxes_bitslice (&key)[56], const sboxes_bitslice	slice);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_set_low_keys(sboxes_bitslice	(&key)[56]);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_add_keys(sboxes_bitslice (&key)[56], const sboxes_deskey val);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_inc_keys(sboxes_bitslice (&key)[56]);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_inc_high_keys(sboxes_bitslice	(&key)[56]);
extern __device__ __host__ __forceinline__ void            sboxes_add_keys_fast(sboxes_bitslice (&key)[56], const sboxes_deskey val);
extern __device__ __host__ __forceinline__ void            sboxes_inc_keys_fast(sboxes_bitslice (&key)[56]);
extern __device__ __host__ __forceinline__ void            sboxes_set_key64(sboxes_bitslice (&k)[56], const sboxes_deskey k64);
extern __device__ __host__ __forceinline__ sboxes_deskey   sboxes_get_key64(const sboxes_bitslice (&k)[56], const sboxes_bitslice sliceMask);
extern __device__ __host__ __forceinline__ void            sboxes_set_data(sboxes_bitslice (&p)[64], const sboxes_block p64);
extern __device__ __host__ __forceinline__ sboxes_block    sboxes_get_data(const sboxes_bitslice (&p)[64]);
extern __device__ __host__ __forceinline__ void            sboxes_unroll_bits(sboxes_bitslice *pp, sboxes_bitslice *pc, sboxes_bitslice *pk, const sboxes_block p, const sboxes_block c, const sboxes_deskey k);

#ifdef __cplusplus
}
#endif

#endif /* DESEVAL_H */
