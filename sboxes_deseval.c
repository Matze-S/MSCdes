#pragma once

/*
 * Generated S-box files.
 *
 * Produced by Matthew Kwan - March 1998
 */

#include "sboxes_deseval.h"

#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
#include "sbox.h"
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 */

__device__ __host__ __forceinline__ static void _s1(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s1(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice x[64];
	x[ 1] = ~a4;	          x[ 2] = ~a1;	          x[ 3] = a4    ^ a3;	  x[ 4] = x[ 3] ^ x[ 2];	
  x[ 5] = a3    | x[ 2];	x[ 6] = x[ 5] & x[ 1];	x[ 7] = a6    | x[ 6]; x[ 8] = x[ 4] ^ x[ 7];	
  x[ 9] = x[ 1] | x[ 2];	x[10] = a6    & x[ 9]; x[11] = x[ 7] ^ x[10]; x[12] =    a2 | x[11];	
  x[13] = x[ 8] ^ x[12];	x[14] = x[ 9] ^ x[13];	x[15] = a6    | x[14]; x[16] = x[ 1] ^ x[15]; 
  x[17] =        ~x[14]; x[18] = x[17] & x[ 3];	x[19] = a2    | x[18]; x[20] = x[16] ^ x[19]; 
  x[21] = a5    | x[20];	x[22] = x[13] ^ x[21];
  *out4 ^= x[22];
	x[23] = a3    | x[ 4];	x[24] =        ~x[23]; x[25] = a6    | x[24]; x[26] = x[ 6] ^ x[25];
  x[27] = x[ 1] & x[ 8]; x[28] = a2    | x[27]; x[29] = x[26] ^ x[28]; x[30] = x[ 1] | x[ 8];
  x[31] = x[30] ^ x[ 6]; x[32] = x[ 5] & x[14]; x[33] = x[32] ^ x[ 8]; x[34] = a2    & x[33];
  x[35] = x[31] ^ x[34]; x[36] = a5    | x[35]; x[37] = x[29] ^ x[36];
  *out1 ^= x[37];
  x[38] = a3    & x[10]; x[39] = x[38] | x[ 4]; x[40] = a3    & x[33]; x[41] = x[40] ^ x[25];
  x[42] = a2    | x[41]; x[43] = x[39] ^ x[42]; x[44] = a3    | x[26]; x[45] = x[44] ^ x[14];
  x[46] = a1    | x[ 8]; x[47] = x[46] ^ x[20]; x[48] = a2    | x[47]; x[49] = x[45] ^ x[48];
  x[50] = a5    & x[49]; x[51] = x[43] ^ x[50];
  *out2 ^= x[51];
  x[52] = x[ 8] ^ x[40]; x[53] = a3    ^ x[11]; x[54] = x[53] & x[ 5]; x[55] = a2    | x[54];
  x[56] = x[52] ^ x[55]; x[57] = a6    | x[ 4]; x[58] = x[57] ^ x[38]; x[59] = x[13] & x[56];
  x[60] = a2    & x[59]; x[61] = x[58] ^ x[60]; x[62] = a5    & x[61]; x[63] = x[56] ^ x[62];
  *out3 ^= x[63];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s2(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s2(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
  sboxes_bitslice x[57];
	x[ 1] =        ~a5;    x[ 2] = ~a1;            x[ 3] = a5    ^ a6;    x[ 4] = x[ 3] ^ x[ 2];
	x[ 5] = x[ 4] ^ a2;    x[ 6] = a6    | x[ 1]; x[ 7] = x[ 6] | x[ 2]; x[ 8] = a2    & x[ 7];
	x[ 9] = a6    ^ x[ 8]; x[10] = a3    & x[ 9]; x[11] = x[ 5] ^ x[10]; x[12] = a2    & x[ 9];
	x[13] = a5    ^ x[ 6]; x[14] = a3    | x[13]; x[15] = x[12] ^ x[14]; x[16] = a4    & x[15];
	x[17] = x[11] ^ x[16];
	*out2 ^= x[17];
	x[18] = a5    | a1;    x[19] = a6    | x[18]; x[20] = x[13] ^ x[19]; x[21] = x[20] ^ a2;
	x[22] = a6    | x[ 4]; x[23] = x[22] & x[17]; x[24] = a3    | x[23]; x[25] = x[21] ^ x[24];
	x[26] = a6    | x[ 2]; x[27] = a5    & x[ 2]; x[28] = a2    | x[27]; x[29] = x[26] ^ x[28];
	x[30] = x[ 3] ^ x[27]; x[31] = x[ 2] ^ x[19]; x[32] = a2    & x[31]; x[33] = x[30] ^ x[32];
	x[34] = a3    & x[33]; x[35] = x[29] ^ x[34]; x[36] = a4    | x[35]; x[37] = x[25] ^ x[36];
  *out3 ^= x[37];
  x[38] = x[21] & x[32]; x[39] = x[38] ^ x[ 5]; x[40] = a1    | x[15]; x[41] = x[40] ^ x[13];
	x[42] = a3    | x[41]; x[43] = x[39] ^ x[42]; x[44] = x[28] | x[41]; x[45] = a4    & x[44];
	x[46] = x[43] ^ x[45];
	*out1 ^= x[46];
  x[47] = x[19] & x[21]; x[48] = x[47] ^ x[26]; x[49] = a2    & x[33]; x[50] = x[49] ^ x[21];
	x[51] = a3    & x[50]; x[52] = x[48] ^ x[51]; x[53] = x[18] & x[28]; x[54] = x[53] & x[50];
	x[55] = a4    | x[54]; x[56] = x[52] ^ x[55];
	*out4 ^= x[56];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s3(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s3(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[58];
	x[ 1] =        ~a5;    x[ 2] =  ~a6;           x[ 3] = a5    & a3;    x[ 4] = x[ 3] ^ a6;
	x[ 5] = a4    & x[ 1]; x[ 6] = x[ 4] ^ x[ 5]; x[ 7] = x[ 6] ^ a2;    x[ 8] = a3    & x[ 1];
	x[ 9] = a5    ^ x[ 2]; x[10] = a4    | x[ 9]; x[11] = x[ 8] ^ x[10]; x[12] = x[ 7] & x[11];
	x[13] = a5    ^ x[11]; x[14] = x[13] | x[ 7]; x[15] = a4    & x[14]; x[16] = x[12] ^ x[15];
	x[17] = a2    & x[16]; x[18] = x[11] ^ x[17]; x[19] = a1    & x[18]; x[20] = x[ 7] ^ x[19];
	*out4 ^= x[20];
	x[21] = a3    ^ a4;    x[22] = x[21] ^ x[ 9]; x[23] = x[ 2] | x[ 4]; x[24] = x[23] ^ x[ 8];
	x[25] = a2    | x[24]; x[26] = x[22] ^ x[25]; x[27] = a6    ^ x[23]; x[28] = x[27] | a4;
	x[29] = a3    ^ x[15]; x[30] = x[29] | x[ 5]; x[31] = a2    | x[30]; x[32] = x[28] ^ x[31];
	x[33] = a1    | x[32]; x[34] = x[26] ^ x[33];
	*out1 ^= x[34];
	x[35] = a3    ^ x[ 9]; x[36] = x[35] | x[ 5]; x[37] = x[ 4] | x[29]; x[38] = x[37] ^ a4;
	x[39] = a2    | x[38]; x[40] = x[36] ^ x[39]; x[41] = a6    & x[11]; x[42] = x[41] | x[ 6];
	x[43] = x[34] ^ x[38]; x[44] = x[43] ^ x[41]; x[45] = a2    & x[44]; x[46] = x[42] ^ x[45];
	x[47] = a1    | x[46]; x[48] = x[40] ^ x[47];
	*out3 ^= x[48];
	x[49] = x[ 2] | x[38]; x[50] = x[49] ^ x[13]; x[51] = x[27] ^ x[28]; x[52] = a2    | x[51];
	x[53] = x[50] ^ x[52]; x[54] = x[12] & x[23]; x[55] = x[54] & x[52]; x[56] = a1    | x[55];
	x[57] = x[53] ^ x[56];
	*out2 ^= x[57];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s4(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s4(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[43];
	x[ 1] =        ~a1;     x[ 2] =        ~a3;    x[ 3] = a1    | a3;     x[ 4] = a5    & x[ 3];
	x[ 5] = x[ 1] ^ x[ 4]; x[ 6] = a2    | a3;     x[ 7] = x[ 5] ^ x[ 6]; x[ 8] = a1    & a5;
	x[ 9] = x[ 8] ^ x[ 3]; x[10] = a2    & x[ 9]; x[11] = a5    ^ x[10]; x[12] = a4    & x[11];
	x[13] = x[ 7] ^ x[12]; x[14] = x[ 2] ^ x[ 4]; x[15] = a2    & x[14]; x[16] = x[ 9] ^ x[15];
	x[17] = x[ 5] & x[14]; x[18] = a5    ^ x[ 2]; x[19] = a2    | x[18]; x[20] = x[17] ^ x[19];
	x[21] = a4    | x[20]; x[22] = x[16] ^ x[21]; x[23] = a6    & x[22]; x[24] = x[13] ^ x[23];
	*out2 ^= x[24];
	x[25] =        ~x[13]; x[26] = a6    | x[22]; x[27] = x[25] ^ x[26];
	*out1 ^= x[27];
	x[28] = a2    & x[11]; x[29] = x[28] ^ x[17]; x[30] = a3    ^ x[10]; x[31] = x[30] ^ x[19];
	x[32] = a4    & x[31]; x[33] = x[29] ^ x[32]; x[34] = x[25] ^ x[33]; x[35] = a2    & x[34];
	x[36] = x[24] ^ x[35]; x[37] = a4    | x[34]; x[38] = x[36] ^ x[37]; x[39] = a6    & x[38];
	x[40] = x[33] ^ x[39];
	*out4 ^= x[40];
	x[41] = x[26] ^ x[38]; x[42] = x[41] ^ x[40];
  *out3 ^= x[42];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s5(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s5(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[63];
	x[ 1] =        ~a6;    x[ 2] =        ~a3;     x[ 3] = x[ 1] | x[ 2]; x[ 4] = x[ 3] ^ a4;
	x[ 5] = a1    & x[ 3]; x[ 6] = x[ 4] ^ x[ 5]; x[ 7] = a6    | a4;    x[ 8] = x[ 7] ^ a3;
	x[ 9] = a3    | x[ 7]; x[10] = a1    | x[ 9]; x[11] = x[ 8] ^ x[10]; x[12] = a5    & x[11];
	x[13] = x[ 6] ^ x[12]; x[14] =        ~x[ 4]; x[15] = x[14] & a6;    x[16] = a1    | x[15];
	x[17] = x[ 8] ^ x[16]; x[18] = a5    | x[17]; x[19] = x[10] ^ x[18]; x[20] = a2    | x[19];
	x[21] = x[13] ^ x[20];
	*out3 ^= x[21];
	x[22] = x[ 2] | x[15]; x[23] = x[22] ^ a6;    x[24] = a4    ^ x[22]; x[25] = a1    & x[24];
	x[26] = x[23] ^ x[25]; x[27] = a1    ^ x[11]; x[28] = x[27] & x[22]; x[29] = a5    | x[28];
	x[30] = x[26] ^ x[29]; x[31] = a4    | x[27]; x[32] =        ~x[31]; x[33] = a2    | x[32];
	x[34] = x[30] ^ x[33];
	*out2 ^= x[34];
	x[35] = x[ 2] ^ x[15]; x[36] = a1    & x[35]; x[37] = x[14] ^ x[36]; x[38] = x[ 5] ^ x[ 7];
	x[39] = x[38] & x[34]; x[40] = a5    | x[39]; x[41] = x[37] ^ x[40]; x[42] = x[ 2] ^ x[ 5];
	x[43] = x[42] & x[16]; x[44] = x[ 4] & x[27]; x[45] = a5    & x[44]; x[46] = x[43] ^ x[45];
	x[47] = a2    | x[46]; x[48] = x[41] ^ x[47];
	*out1 ^= x[48];
	x[49] = x[24] & x[48]; x[50] = x[49] ^ x[ 5]; x[51] = x[11] ^ x[30]; x[52] = x[51] | x[50];
	x[53] = a5    & x[52]; x[54] = x[50] ^ x[53]; x[55] = x[14] ^ x[19]; x[56] = x[55] ^ x[34];
	x[57] = x[ 4] ^ x[16]; x[58] = x[57] & x[30]; x[59] = a5    & x[58]; x[60] = x[56] ^ x[59];
	x[61] = a2    | x[60]; x[62] = x[54] ^ x[61];
	*out4 ^= x[62];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s6(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s6(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[58];
	x[ 1] =        ~a2;    x[ 2] =        ~a5;     x[ 3] = a2    ^ a6;     x[ 4] = x[ 3] ^ x[ 2];
	x[ 5] = x[ 4] ^ a1;    x[ 6] = a5    & a6;     x[ 7] = x[ 6] | x[ 1]; x[ 8] = a5    & x[ 5];
	x[ 9] = a1    & x[ 8]; x[10] = x[ 7] ^ x[ 9]; x[11] = a4    & x[10];  x[12] = x[ 5] ^ x[11];
	x[13] = a6    ^ x[10]; x[14] = x[13] & a1;    x[15] = a2    & a6;     x[16] = x[15] ^ a5;
	x[17] = a1    & x[16]; x[18] = x[ 2] ^ x[17]; x[19] = a4    | x[18]; x[20] = x[14] ^ x[19];
	x[21] = a3    & x[20]; x[22] = x[12] ^ x[21];
	*out2 ^= x[22];
	x[23] = a6    ^ x[18]; x[24] = a1    & x[23]; x[25] = a5    ^ x[24]; x[26] = a2    ^ x[17];
	x[27] = x[26] | x[ 6]; x[28] = a4    & x[27]; x[29] = x[25] ^ x[28]; x[30] =        ~x[26];
	x[31] = a6    | x[29]; x[32] =        ~x[31]; x[33] = a4    & x[32]; x[34] = x[30] ^ x[33];
	x[35] = a3    & x[34]; x[36] = x[29] ^ x[35];
	*out4 ^= x[36];
	x[37] = x[ 6] ^ x[34]; x[38] = a5    & x[23]; x[39] = x[38] ^ x[ 5]; x[40] = a4    | x[39];
	x[41] = x[37] ^ x[40]; x[42] = x[16] | x[24]; x[43] = x[42] ^ x[ 1]; x[44] = x[15] ^ x[24];
	x[45] = x[44] ^ x[31]; x[46] = a4    | x[45]; x[47] = x[43] ^ x[46]; x[48] = a3    | x[47];
	x[49] = x[41] ^ x[48];
	*out1 ^= x[49];
	x[50] = x[ 5] | x[38]; x[51] = x[50] ^ x[ 6]; x[52] = x[ 8] & x[31]; x[53] = a4    | x[52];
	x[54] = x[51] ^ x[53]; x[55] = x[30] & x[43]; x[56] = a3    | x[55];
	x[57] = x[54] ^ x[56];
	*out3 ^= x[57];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ void static _s7(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s7(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[58];
	x[ 1] =        ~a2;    x[ 2] =        ~a5;     x[ 3] = a2    & a4;    x[ 4] = x[ 3] ^ a5;
	x[ 5] = x[ 4] ^ a3;    x[ 6] = a4    & x[ 4]; x[ 7] = x[ 6] ^ a2;    x[ 8] = a3    & x[ 7];
	x[ 9] = a1    ^ x[ 8]; x[10] = a6    | x[ 9]; x[11] = x[ 5] ^ x[10]; x[12] = a4    & x[ 2];
	x[13] = x[12] | a2;    x[14] = a2    | x[ 2]; x[15] = a3    & x[14]; x[16] = x[13] ^ x[15];
	x[17] = x[ 6] ^ x[11]; x[18] = a6    | x[17]; x[19] = x[16] ^ x[18]; x[20] = a1    & x[19];
	x[21] = x[11] ^ x[20];
	*out1 ^= x[21];
	x[22] = a2    | x[21]; x[23] = x[22] ^ x[ 6]; x[24] = x[23] ^ x[15]; x[25] = x[ 5] ^ x[ 6];
	x[26] = x[25] | x[12]; x[27] = a6    | x[26]; x[28] = x[24] ^ x[27]; x[29] = x[ 1] & x[19];
	x[30] = x[23] & x[26]; x[31] = a6    & x[30]; x[32] = x[29] ^ x[31]; x[33] = a1    | x[32];
	x[34] = x[28] ^ x[33];
	*out4 ^= x[34];
	x[35] = a4    & x[16]; x[36] = x[35] | x[ 1]; x[37] = a6    & x[36]; x[38] = x[11] ^ x[37];
	x[39] = a4    & x[13]; x[40] = a3    | x[ 7]; x[41] = x[39] ^ x[40]; x[42] = x[ 1] | x[24];
	x[43] = a6    | x[42]; x[44] = x[41] ^ x[43]; x[45] = a1    | x[44]; x[46] = x[38] ^ x[45];
	*out2 ^= x[46];
	x[47] = x[ 8] ^ x[44]; x[48] = x[ 6] ^ x[15]; x[49] = a6    | x[48]; x[50] = x[47] ^ x[49];
	x[51] = x[19] ^ x[44]; x[52] = a4    ^ x[25]; x[53] = x[52] & x[46]; x[54] = a6    & x[53];
	x[55] = x[51] ^ x[54]; x[56] = a1    | x[55]; x[57] = x[50] ^ x[56];
	*out3 ^= x[57];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _s8(
  sboxes_bitslice	a1   , sboxes_bitslice	a2   , sboxes_bitslice	a3   ,
  sboxes_bitslice	a4   , sboxes_bitslice	a5   , sboxes_bitslice	a6   ,
  sboxes_bitslice	*out1, sboxes_bitslice	*out2, sboxes_bitslice	*out3, sboxes_bitslice	*out4) {
#if sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500
  return s8(a1, a2, a3, a4, a5, a6, out1, out2, out3, out4);
#else /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
	sboxes_bitslice	x[55];

	x[ 1] =        ~a1    ; x[ 2] =        ~a4;    x[ 3] = a3    ^ x[ 1]; x[ 4] = a3    | x[ 1];
	x[ 5] = x[ 4] ^ x[ 2]; x[ 6] = a5    | x[ 5]; x[ 7] = x[ 3] ^ x[ 6]; x[ 8] = x[ 1] | x[ 5];
	x[ 9] = x[ 2] ^ x[ 8]; x[10] = a5    & x[ 9]; x[11] = x[ 8] ^ x[10]; x[12] = a2    & x[11];
	x[13] = x[ 7] ^ x[12]; x[14] = x[ 6] ^ x[ 9]; x[15] = x[ 3] & x[ 9]; x[16] = a5    & x[ 8];
	x[17] = x[15] ^ x[16]; x[18] = a2    | x[17]; x[19] = x[14] ^ x[18]; x[20] = a6    | x[19];
	x[21] = x[13] ^ x[20];
	*out1 ^= x[21];
	x[22] = a5    | x[ 3]; x[23] = x[22] & x[ 2]; x[24] =        ~a3;    x[25] = x[24] & x[ 8];
	x[26] = a5    & x[ 4]; x[27] = x[25] ^ x[26]; x[28] = a2    | x[27]; x[29] = x[23] ^ x[28];
	x[30] = a6    & x[29]; x[31] = x[13] ^ x[30];
	*out4 ^= x[31];
	x[32] = x[ 5] ^ x[ 6]; x[33] = x[32] ^ x[22]; x[34] = a4    | x[13]; x[35] = a2    & x[34];
	x[36] = x[33] ^ x[35]; x[37] = a1    & x[33]; x[38] = x[37] ^ x[ 8]; x[39] = a1    ^ x[23];
	x[40] = x[39] & x[ 7]; x[41] = a2    & x[40]; x[42] = x[38] ^ x[41]; x[43] = a6    | x[42];
	x[44] = x[36] ^ x[43];
	*out3 ^= x[44];
	x[45] = a1    ^ x[10]; x[46] = x[45] ^ x[22]; x[47] =        ~x[ 7]; x[48] = x[47] & x[ 8];
	x[49] = a2    | x[48]; x[50] = x[46] ^ x[49]; x[51] = x[19] ^ x[29]; x[52] = x[51] | x[38];
	x[53] = a6    & x[52]; x[54] = x[50] ^ x[53];
	*out2 ^= x[54];
#endif /* sboxes_slicebitcount == 5 && sboxes_slicebitsize == 32 && __CUDA_ARCH__ >= 500 */
}

__device__ __host__ __forceinline__ static void _r1(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[47], r[ 0] ^ k[11], r[ 1] ^ k[26], r[ 2] ^ k[ 3], r[ 3] ^ k[13], r[ 4] ^ k[41], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[27], r[ 4] ^ k[ 6], r[ 5] ^ k[54], r[ 6] ^ k[48], r[ 7] ^ k[39], r[ 8] ^ k[19], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[53], r[ 8] ^ k[25], r[ 9] ^ k[33], r[10] ^ k[34], r[11] ^ k[17], r[12] ^ k[ 5], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[ 4], r[12] ^ k[55], r[13] ^ k[24], r[14] ^ k[32], r[15] ^ k[40], r[16] ^ k[20], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[36], r[16] ^ k[31], r[17] ^ k[21], r[18] ^ k[ 8], r[19] ^ k[23], r[20] ^ k[52], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[14], r[20] ^ k[29], r[21] ^ k[51], r[22] ^ k[ 9], r[23] ^ k[35], r[24] ^ k[30], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[ 2], r[24] ^ k[37], r[25] ^ k[22], r[26] ^ k[ 0], r[27] ^ k[42], r[28] ^ k[38], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[16], r[28] ^ k[43], r[29] ^ k[44], r[30] ^ k[ 1], r[31] ^ k[ 7], r[ 0] ^ k[28], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r2(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[54], l[ 0] ^ k[18], l[ 1] ^ k[33], l[ 2] ^ k[10], l[ 3] ^ k[20], l[ 4] ^ k[48], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[34], l[ 4] ^ k[13], l[ 5] ^ k[ 4], l[ 6] ^ k[55], l[ 7] ^ k[46], l[ 8] ^ k[26], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[ 3], l[ 8] ^ k[32], l[ 9] ^ k[40], l[10] ^ k[41], l[11] ^ k[24], l[12] ^ k[12], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[11], l[12] ^ k[ 5], l[13] ^ k[ 6], l[14] ^ k[39], l[15] ^ k[47], l[16] ^ k[27], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[43], l[16] ^ k[38], l[17] ^ k[28], l[18] ^ k[15], l[19] ^ k[30], l[20] ^ k[ 0], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[21], l[20] ^ k[36], l[21] ^ k[31], l[22] ^ k[16], l[23] ^ k[42], l[24] ^ k[37], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[ 9], l[24] ^ k[44], l[25] ^ k[29], l[26] ^ k[ 7], l[27] ^ k[49], l[28] ^ k[45], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[23], l[28] ^ k[50], l[29] ^ k[51], l[30] ^ k[ 8], l[31] ^ k[14], l[ 0] ^ k[35], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r3(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[11], r[ 0] ^ k[32], r[ 1] ^ k[47], r[ 2] ^ k[24], r[ 3] ^ k[34], r[ 4] ^ k[ 5], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[48], r[ 4] ^ k[27], r[ 5] ^ k[18], r[ 6] ^ k[12], r[ 7] ^ k[ 3], r[ 8] ^ k[40], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[17], r[ 8] ^ k[46], r[ 9] ^ k[54], r[10] ^ k[55], r[11] ^ k[13], r[12] ^ k[26], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[25], r[12] ^ k[19], r[13] ^ k[20], r[14] ^ k[53], r[15] ^ k[ 4], r[16] ^ k[41], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[ 2], r[16] ^ k[52], r[17] ^ k[42], r[18] ^ k[29], r[19] ^ k[44], r[20] ^ k[14], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[35], r[20] ^ k[50], r[21] ^ k[45], r[22] ^ k[30], r[23] ^ k[ 1], r[24] ^ k[51], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[23], r[24] ^ k[31], r[25] ^ k[43], r[26] ^ k[21], r[27] ^ k[ 8], r[28] ^ k[ 0], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[37], r[28] ^ k[ 9], r[29] ^ k[38], r[30] ^ k[22], r[31] ^ k[28], r[ 0] ^ k[49], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r4(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[25], l[ 0] ^ k[46], l[ 1] ^ k[ 4], l[ 2] ^ k[13], l[ 3] ^ k[48], l[ 4] ^ k[19], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[ 5], l[ 4] ^ k[41], l[ 5] ^ k[32], l[ 6] ^ k[26], l[ 7] ^ k[17], l[ 8] ^ k[54], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[ 6], l[ 8] ^ k[ 3], l[ 9] ^ k[11], l[10] ^ k[12], l[11] ^ k[27], l[12] ^ k[40], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[39], l[12] ^ k[33], l[13] ^ k[34], l[14] ^ k[10], l[15] ^ k[18], l[16] ^ k[55], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[16], l[16] ^ k[ 7], l[17] ^ k[ 1], l[18] ^ k[43], l[19] ^ k[31], l[20] ^ k[28], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[49], l[20] ^ k[ 9], l[21] ^ k[ 0], l[22] ^ k[44], l[23] ^ k[15], l[24] ^ k[38], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[37], l[24] ^ k[45], l[25] ^ k[ 2], l[26] ^ k[35], l[27] ^ k[22], l[28] ^ k[14], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[51], l[28] ^ k[23], l[29] ^ k[52], l[30] ^ k[36], l[31] ^ k[42], l[ 0] ^ k[ 8], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r5(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[39], r[ 0] ^ k[ 3], r[ 1] ^ k[18], r[ 2] ^ k[27], r[ 3] ^ k[ 5], r[ 4] ^ k[33], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[19], r[ 4] ^ k[55], r[ 5] ^ k[46], r[ 6] ^ k[40], r[ 7] ^ k[ 6], r[ 8] ^ k[11], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[20], r[ 8] ^ k[17], r[ 9] ^ k[25], r[10] ^ k[26], r[11] ^ k[41], r[12] ^ k[54], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[53], r[12] ^ k[47], r[13] ^ k[48], r[14] ^ k[24], r[15] ^ k[32], r[16] ^ k[12], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[30], r[16] ^ k[21], r[17] ^ k[15], r[18] ^ k[ 2], r[19] ^ k[45], r[20] ^ k[42], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[ 8], r[20] ^ k[23], r[21] ^ k[14], r[22] ^ k[31], r[23] ^ k[29], r[24] ^ k[52], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[51], r[24] ^ k[ 0], r[25] ^ k[16], r[26] ^ k[49], r[27] ^ k[36], r[28] ^ k[28], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[38], r[28] ^ k[37], r[29] ^ k[ 7], r[30] ^ k[50], r[31] ^ k[ 1], r[ 0] ^ k[22], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r6(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[53], l[ 0] ^ k[17], l[ 1] ^ k[32], l[ 2] ^ k[41], l[ 3] ^ k[19], l[ 4] ^ k[47], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[33], l[ 4] ^ k[12], l[ 5] ^ k[ 3], l[ 6] ^ k[54], l[ 7] ^ k[20], l[ 8] ^ k[25], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[34], l[ 8] ^ k[ 6], l[ 9] ^ k[39], l[10] ^ k[40], l[11] ^ k[55], l[12] ^ k[11], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[10], l[12] ^ k[ 4], l[13] ^ k[ 5], l[14] ^ k[13], l[15] ^ k[46], l[16] ^ k[26], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[44], l[16] ^ k[35], l[17] ^ k[29], l[18] ^ k[16], l[19] ^ k[ 0], l[20] ^ k[ 1], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[22], l[20] ^ k[37], l[21] ^ k[28], l[22] ^ k[45], l[23] ^ k[43], l[24] ^ k[ 7], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[38], l[24] ^ k[14], l[25] ^ k[30], l[26] ^ k[ 8], l[27] ^ k[50], l[28] ^ k[42], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[52], l[28] ^ k[51], l[29] ^ k[21], l[30] ^ k[ 9], l[31] ^ k[15], l[ 0] ^ k[36], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r7(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[10], r[ 0] ^ k[ 6], r[ 1] ^ k[46], r[ 2] ^ k[55], r[ 3] ^ k[33], r[ 4] ^ k[ 4], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[47], r[ 4] ^ k[26], r[ 5] ^ k[17], r[ 6] ^ k[11], r[ 7] ^ k[34], r[ 8] ^ k[39], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[48], r[ 8] ^ k[20], r[ 9] ^ k[53], r[10] ^ k[54], r[11] ^ k[12], r[12] ^ k[25], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[24], r[12] ^ k[18], r[13] ^ k[19], r[14] ^ k[27], r[15] ^ k[ 3], r[16] ^ k[40], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[31], r[16] ^ k[49], r[17] ^ k[43], r[18] ^ k[30], r[19] ^ k[14], r[20] ^ k[15], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[36], r[20] ^ k[51], r[21] ^ k[42], r[22] ^ k[ 0], r[23] ^ k[ 2], r[24] ^ k[21], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[52], r[24] ^ k[28], r[25] ^ k[44], r[26] ^ k[22], r[27] ^ k[ 9], r[28] ^ k[ 1], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[ 7], r[28] ^ k[38], r[29] ^ k[35], r[30] ^ k[23], r[31] ^ k[29], r[ 0] ^ k[50], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r8(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[24], l[ 0] ^ k[20], l[ 1] ^ k[ 3], l[ 2] ^ k[12], l[ 3] ^ k[47], l[ 4] ^ k[18], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[ 4], l[ 4] ^ k[40], l[ 5] ^ k[ 6], l[ 6] ^ k[25], l[ 7] ^ k[48], l[ 8] ^ k[53], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[ 5], l[ 8] ^ k[34], l[ 9] ^ k[10], l[10] ^ k[11], l[11] ^ k[26], l[12] ^ k[39], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[13], l[12] ^ k[32], l[13] ^ k[33], l[14] ^ k[41], l[15] ^ k[17], l[16] ^ k[54], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[45], l[16] ^ k[ 8], l[17] ^ k[ 2], l[18] ^ k[44], l[19] ^ k[28], l[20] ^ k[29], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[50], l[20] ^ k[38], l[21] ^ k[ 1], l[22] ^ k[14], l[23] ^ k[16], l[24] ^ k[35], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[ 7], l[24] ^ k[42], l[25] ^ k[31], l[26] ^ k[36], l[27] ^ k[23], l[28] ^ k[15], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[21], l[28] ^ k[52], l[29] ^ k[49], l[30] ^ k[37], l[31] ^ k[43], l[ 0] ^ k[ 9], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r9(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[ 6], r[ 0] ^ k[27], r[ 1] ^ k[10], r[ 2] ^ k[19], r[ 3] ^ k[54], r[ 4] ^ k[25], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[11], r[ 4] ^ k[47], r[ 5] ^ k[13], r[ 6] ^ k[32], r[ 7] ^ k[55], r[ 8] ^ k[ 3], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[12], r[ 8] ^ k[41], r[ 9] ^ k[17], r[10] ^ k[18], r[11] ^ k[33], r[12] ^ k[46], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[20], r[12] ^ k[39], r[13] ^ k[40], r[14] ^ k[48], r[15] ^ k[24], r[16] ^ k[ 4], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[52], r[16] ^ k[15], r[17] ^ k[ 9], r[18] ^ k[51], r[19] ^ k[35], r[20] ^ k[36], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[ 2], r[20] ^ k[45], r[21] ^ k[ 8], r[22] ^ k[21], r[23] ^ k[23], r[24] ^ k[42], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[14], r[24] ^ k[49], r[25] ^ k[38], r[26] ^ k[43], r[27] ^ k[30], r[28] ^ k[22], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[28], r[28] ^ k[ 0], r[29] ^ k[ 1], r[30] ^ k[44], r[31] ^ k[50], r[ 0] ^ k[16], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r10(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[20], l[ 0] ^ k[41], l[ 1] ^ k[24], l[ 2] ^ k[33], l[ 3] ^ k[11], l[ 4] ^ k[39], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[25], l[ 4] ^ k[ 4], l[ 5] ^ k[27], l[ 6] ^ k[46], l[ 7] ^ k[12], l[ 8] ^ k[17], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[26], l[ 8] ^ k[55], l[ 9] ^ k[ 6], l[10] ^ k[32], l[11] ^ k[47], l[12] ^ k[ 3], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[34], l[12] ^ k[53], l[13] ^ k[54], l[14] ^ k[ 5], l[15] ^ k[13], l[16] ^ k[18], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[ 7], l[16] ^ k[29], l[17] ^ k[23], l[18] ^ k[38], l[19] ^ k[49], l[20] ^ k[50], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[16], l[20] ^ k[ 0], l[21] ^ k[22], l[22] ^ k[35], l[23] ^ k[37], l[24] ^ k[ 1], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[28], l[24] ^ k[ 8], l[25] ^ k[52], l[26] ^ k[ 2], l[27] ^ k[44], l[28] ^ k[36], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[42], l[28] ^ k[14], l[29] ^ k[15], l[30] ^ k[31], l[31] ^ k[ 9], l[ 0] ^ k[30], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r11(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[34], r[ 0] ^ k[55], r[ 1] ^ k[13], r[ 2] ^ k[47], r[ 3] ^ k[25], r[ 4] ^ k[53], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[39], r[ 4] ^ k[18], r[ 5] ^ k[41], r[ 6] ^ k[ 3], r[ 7] ^ k[26], r[ 8] ^ k[ 6], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[40], r[ 8] ^ k[12], r[ 9] ^ k[20], r[10] ^ k[46], r[11] ^ k[ 4], r[12] ^ k[17], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[48], r[12] ^ k[10], r[13] ^ k[11], r[14] ^ k[19], r[15] ^ k[27], r[16] ^ k[32], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[21], r[16] ^ k[43], r[17] ^ k[37], r[18] ^ k[52], r[19] ^ k[ 8], r[20] ^ k[ 9], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[30], r[20] ^ k[14], r[21] ^ k[36], r[22] ^ k[49], r[23] ^ k[51], r[24] ^ k[15], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[42], r[24] ^ k[22], r[25] ^ k[ 7], r[26] ^ k[16], r[27] ^ k[31], r[28] ^ k[50], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[ 1], r[28] ^ k[28], r[29] ^ k[29], r[30] ^ k[45], r[31] ^ k[23], r[ 0] ^ k[44], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r12(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[48], l[ 0] ^ k[12], l[ 1] ^ k[27], l[ 2] ^ k[ 4], l[ 3] ^ k[39], l[ 4] ^ k[10], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[53], l[ 4] ^ k[32], l[ 5] ^ k[55], l[ 6] ^ k[17], l[ 7] ^ k[40], l[ 8] ^ k[20], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[54], l[ 8] ^ k[26], l[ 9] ^ k[34], l[10] ^ k[ 3], l[11] ^ k[18], l[12] ^ k[ 6], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[ 5], l[12] ^ k[24], l[13] ^ k[25], l[14] ^ k[33], l[15] ^ k[41], l[16] ^ k[46], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[35], l[16] ^ k[ 2], l[17] ^ k[51], l[18] ^ k[ 7], l[19] ^ k[22], l[20] ^ k[23], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[44], l[20] ^ k[28], l[21] ^ k[50], l[22] ^ k[ 8], l[23] ^ k[38], l[24] ^ k[29], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[ 1], l[24] ^ k[36], l[25] ^ k[21], l[26] ^ k[30], l[27] ^ k[45], l[28] ^ k[ 9], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[15], l[28] ^ k[42], l[29] ^ k[43], l[30] ^ k[ 0], l[31] ^ k[37], l[ 0] ^ k[31], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r13(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[ 5], r[ 0] ^ k[26], r[ 1] ^ k[41], r[ 2] ^ k[18], r[ 3] ^ k[53], r[ 4] ^ k[24], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[10], r[ 4] ^ k[46], r[ 5] ^ k[12], r[ 6] ^ k[ 6], r[ 7] ^ k[54], r[ 8] ^ k[34], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[11], r[ 8] ^ k[40], r[ 9] ^ k[48], r[10] ^ k[17], r[11] ^ k[32], r[12] ^ k[20], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[19], r[12] ^ k[13], r[13] ^ k[39], r[14] ^ k[47], r[15] ^ k[55], r[16] ^ k[ 3], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[49], r[16] ^ k[16], r[17] ^ k[38], r[18] ^ k[21], r[19] ^ k[36], r[20] ^ k[37], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[31], r[20] ^ k[42], r[21] ^ k[ 9], r[22] ^ k[22], r[23] ^ k[52], r[24] ^ k[43], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[15], r[24] ^ k[50], r[25] ^ k[35], r[26] ^ k[44], r[27] ^ k[ 0], r[28] ^ k[23], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[29], r[28] ^ k[ 1], r[29] ^ k[ 2], r[30] ^ k[14], r[31] ^ k[51], r[ 0] ^ k[45], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r14(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[19], l[ 0] ^ k[40], l[ 1] ^ k[55], l[ 2] ^ k[32], l[ 3] ^ k[10], l[ 4] ^ k[13], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[24], l[ 4] ^ k[ 3], l[ 5] ^ k[26], l[ 6] ^ k[20], l[ 7] ^ k[11], l[ 8] ^ k[48], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[25], l[ 8] ^ k[54], l[ 9] ^ k[ 5], l[10] ^ k[ 6], l[11] ^ k[46], l[12] ^ k[34], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[33], l[12] ^ k[27], l[13] ^ k[53], l[14] ^ k[ 4], l[15] ^ k[12], l[16] ^ k[17], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[ 8], l[16] ^ k[30], l[17] ^ k[52], l[18] ^ k[35], l[19] ^ k[50], l[20] ^ k[51], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[45], l[20] ^ k[ 1], l[21] ^ k[23], l[22] ^ k[36], l[23] ^ k[ 7], l[24] ^ k[ 2], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[29], l[24] ^ k[ 9], l[25] ^ k[49], l[26] ^ k[31], l[27] ^ k[14], l[28] ^ k[37], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[43], l[28] ^ k[15], l[29] ^ k[16], l[30] ^ k[28], l[31] ^ k[38], l[ 0] ^ k[ 0], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r15(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(r[31] ^ k[33], r[ 0] ^ k[54], r[ 1] ^ k[12], r[ 2] ^ k[46], r[ 3] ^ k[24], r[ 4] ^ k[27], &l[ 8], &l[16], &l[22], &l[30]);
  _s2(r[ 3] ^ k[13], r[ 4] ^ k[17], r[ 5] ^ k[40], r[ 6] ^ k[34], r[ 7] ^ k[25], r[ 8] ^ k[ 5], &l[12], &l[27], &l[ 1], &l[17]);
  _s3(r[ 7] ^ k[39], r[ 8] ^ k[11], r[ 9] ^ k[19], r[10] ^ k[20], r[11] ^ k[ 3], r[12] ^ k[48], &l[23], &l[15], &l[29], &l[ 5]);
  _s4(r[11] ^ k[47], r[12] ^ k[41], r[13] ^ k[10], r[14] ^ k[18], r[15] ^ k[26], r[16] ^ k[ 6], &l[25], &l[19], &l[ 9], &l[ 0]);
  _s5(r[15] ^ k[22], r[16] ^ k[44], r[17] ^ k[ 7], r[18] ^ k[49], r[19] ^ k[ 9], r[20] ^ k[38], &l[ 7], &l[13], &l[24], &l[ 2]);
  _s6(r[19] ^ k[ 0], r[20] ^ k[15], r[21] ^ k[37], r[22] ^ k[50], r[23] ^ k[21], r[24] ^ k[16], &l[ 3], &l[28], &l[10], &l[18]);
  _s7(r[23] ^ k[43], r[24] ^ k[23], r[25] ^ k[ 8], r[26] ^ k[45], r[27] ^ k[28], r[28] ^ k[51], &l[31], &l[11], &l[21], &l[ 6]);
  _s8(r[27] ^ k[ 2], r[28] ^ k[29], r[29] ^ k[30], r[30] ^ k[42], r[31] ^ k[52], r[ 0] ^ k[14], &l[ 4], &l[26], &l[14], &l[20]);
  return;
}

__device__ __host__ __forceinline__ static void _r16(sboxes_bitslice l[32], sboxes_bitslice r[32], const sboxes_bitslice (&k)[56]) {
  _s1(l[31] ^ k[40], l[ 0] ^ k[ 4], l[ 1] ^ k[19], l[ 2] ^ k[53], l[ 3] ^ k[ 6], l[ 4] ^ k[34], &r[ 8], &r[16], &r[22], &r[30]);
  _s2(l[ 3] ^ k[20], l[ 4] ^ k[24], l[ 5] ^ k[47], l[ 6] ^ k[41], l[ 7] ^ k[32], l[ 8] ^ k[12], &r[12], &r[27], &r[ 1], &r[17]);
  _s3(l[ 7] ^ k[46], l[ 8] ^ k[18], l[ 9] ^ k[26], l[10] ^ k[27], l[11] ^ k[10], l[12] ^ k[55], &r[23], &r[15], &r[29], &r[ 5]);
  _s4(l[11] ^ k[54], l[12] ^ k[48], l[13] ^ k[17], l[14] ^ k[25], l[15] ^ k[33], l[16] ^ k[13], &r[25], &r[19], &r[ 9], &r[ 0]);
  _s5(l[15] ^ k[29], l[16] ^ k[51], l[17] ^ k[14], l[18] ^ k[ 1], l[19] ^ k[16], l[20] ^ k[45], &r[ 7], &r[13], &r[24], &r[ 2]);
  _s6(l[19] ^ k[ 7], l[20] ^ k[22], l[21] ^ k[44], l[22] ^ k[ 2], l[23] ^ k[28], l[24] ^ k[23], &r[ 3], &r[28], &r[10], &r[18]);
  _s7(l[23] ^ k[50], l[24] ^ k[30], l[25] ^ k[15], l[26] ^ k[52], l[27] ^ k[35], l[28] ^ k[31], &r[31], &r[11], &r[21], &r[ 6]);
  _s8(l[27] ^ k[ 9], l[28] ^ k[36], l[29] ^ k[37], l[30] ^ k[49], l[31] ^ k[ 0], l[ 0] ^ k[21], &r[ 4], &r[26], &r[14], &r[20]);
  return;
}
 
__device__ __host__ __forceinline__ void sboxes_desslice_tdec(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k1h)[56], const sboxes_bitslice (&k2l)[56]) {
  _r16(&t[0], &t[32], k1h);  _r15(&t[0], &t[32], k1h);  _r14(&t[0], &t[32], k1h);  _r13(&t[0], &t[32], k1h);
  _r12(&t[0], &t[32], k1h);  _r11(&t[0], &t[32], k1h);  _r10(&t[0], &t[32], k1h);   _r9(&t[0], &t[32], k1h);
   _r8(&t[0], &t[32], k1h);   _r7(&t[0], &t[32], k1h);   _r6(&t[0], &t[32], k1h);   _r5(&t[0], &t[32], k1h);
   _r4(&t[0], &t[32], k1h);   _r3(&t[0], &t[32], k1h);   _r2(&t[0], &t[32], k1h);   _r1(&t[0], &t[32], k1h);

   _r1(&t[0], &t[32], k2l);   _r2(&t[0], &t[32], k2l);   _r3(&t[0], &t[32], k2l);   _r4(&t[0], &t[32], k2l);
   _r5(&t[0], &t[32], k2l);   _r6(&t[0], &t[32], k2l);   _r7(&t[0], &t[32], k2l);   _r8(&t[0], &t[32], k2l);
   _r9(&t[0], &t[32], k2l);  _r10(&t[0], &t[32], k2l);  _r11(&t[0], &t[32], k2l);  _r12(&t[0], &t[32], k2l);
  _r13(&t[0], &t[32], k2l);  _r14(&t[0], &t[32], k2l);  _r15(&t[0], &t[32], k2l);  _r16(&t[0], &t[32], k2l);

  _r16(&t[0], &t[32], k1h);  _r15(&t[0], &t[32], k1h);  _r14(&t[0], &t[32], k1h);  _r13(&t[0], &t[32], k1h);
  _r12(&t[0], &t[32], k1h);  _r11(&t[0], &t[32], k1h);  _r10(&t[0], &t[32], k1h);   _r9(&t[0], &t[32], k1h);
   _r8(&t[0], &t[32], k1h);   _r7(&t[0], &t[32], k1h);   _r6(&t[0], &t[32], k1h);   _r5(&t[0], &t[32], k1h);
   _r4(&t[0], &t[32], k1h);   _r3(&t[0], &t[32], k1h);   _r2(&t[0], &t[32], k1h);   _r1(&t[0], &t[32], k1h);
  return;
}

__device__ __host__ __forceinline__ void sboxes_desslice_tenc(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k1h)[56], const sboxes_bitslice (&k2l)[56]) {
   _r1(&t[32], &t[0], k1h);   _r2(&t[32], &t[0], k1h);   _r3(&t[32], &t[0], k1h);   _r4(&t[32], &t[0], k1h);
   _r5(&t[32], &t[0], k1h);   _r6(&t[32], &t[0], k1h);   _r7(&t[32], &t[0], k1h);   _r8(&t[32], &t[0], k1h);
   _r9(&t[32], &t[0], k1h);  _r10(&t[32], &t[0], k1h);  _r11(&t[32], &t[0], k1h);  _r12(&t[32], &t[0], k1h);
  _r13(&t[32], &t[0], k1h);  _r14(&t[32], &t[0], k1h);  _r15(&t[32], &t[0], k1h);  _r16(&t[32], &t[0], k1h);

  _r16(&t[32], &t[0], k2l);  _r15(&t[32], &t[0], k2l);  _r14(&t[32], &t[0], k2l);  _r13(&t[32], &t[0], k2l);
  _r12(&t[32], &t[0], k2l);  _r11(&t[32], &t[0], k2l);  _r10(&t[32], &t[0], k2l);   _r9(&t[32], &t[0], k2l);
   _r8(&t[32], &t[0], k2l);   _r7(&t[32], &t[0], k2l);   _r6(&t[32], &t[0], k2l);   _r5(&t[32], &t[0], k2l);
   _r4(&t[32], &t[0], k2l);   _r3(&t[32], &t[0], k2l);   _r2(&t[32], &t[0], k2l);   _r1(&t[32], &t[0], k2l);

   _r1(&t[32], &t[0], k1h);   _r2(&t[32], &t[0], k1h);   _r3(&t[32], &t[0], k1h);   _r4(&t[32], &t[0], k1h);
   _r5(&t[32], &t[0], k1h);   _r6(&t[32], &t[0], k1h);   _r7(&t[32], &t[0], k1h);   _r8(&t[32], &t[0], k1h);
   _r9(&t[32], &t[0], k1h);  _r10(&t[32], &t[0], k1h);  _r11(&t[32], &t[0], k1h);  _r12(&t[32], &t[0], k1h);
  _r13(&t[32], &t[0], k1h);  _r14(&t[32], &t[0], k1h);  _r15(&t[32], &t[0], k1h);  _r16(&t[32], &t[0], k1h);
  return;
}

__device__ __host__ __forceinline__ void sboxes_desslice_dec(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k)[56]) {
  _r16(&t[0], &t[32], k);  _r15(&t[0], &t[32], k);  _r14(&t[0], &t[32], k);  _r13(&t[0], &t[32], k);
  _r12(&t[0], &t[32], k);  _r11(&t[0], &t[32], k);  _r10(&t[0], &t[32], k);   _r9(&t[0], &t[32], k);
   _r8(&t[0], &t[32], k);   _r7(&t[0], &t[32], k);   _r6(&t[0], &t[32], k);   _r5(&t[0], &t[32], k);
   _r4(&t[0], &t[32], k);   _r3(&t[0], &t[32], k);   _r2(&t[0], &t[32], k);   _r1(&t[0], &t[32], k); // L[64]..33 / R[32]..1
  return;
}

__device__ __host__ __forceinline__ void sboxes_desslice_enc(sboxes_bitslice (&t)[64], const sboxes_bitslice (&k)[56]) {
   _r1(&t[32], &t[0], k);   _r2(&t[32], &t[0], k);   _r3(&t[32], &t[0], k);   _r4(&t[32], &t[0], k);
   _r5(&t[32], &t[0], k);   _r6(&t[32], &t[0], k);   _r7(&t[32], &t[0], k);   _r8(&t[32], &t[0], k);
   _r9(&t[32], &t[0], k);  _r10(&t[32], &t[0], k);  _r11(&t[32], &t[0], k);  _r12(&t[32], &t[0], k);
  _r13(&t[32], &t[0], k);  _r14(&t[32], &t[0], k);  _r15(&t[32], &t[0], k);  _r16(&t[32], &t[0], k); // L[64]..33 / R[32]..1
  return;
}

__device__ __host__ __forceinline__ sboxes_bitslice sboxes_desslice_eval(sboxes_bitslice (&t)[64], const sboxes_bitslice (&c)[64]) {
  sboxes_bitslice result = sboxes_sliceon;

  if ((result &= ~(t[32+ 8] ^ c[ 5])) == 0 || (result &= ~(t[32+16] ^ c[ 3])) == 0 ||
      (result &= ~(t[32+22] ^ c[51])) == 0 || (result &= ~(t[32+30] ^ c[49])) == 0 ||
      (result &= ~(t[32+12] ^ c[37])) == 0 || (result &= ~(t[32+27] ^ c[25])) == 0 ||
      (result &= ~(t[32+ 1] ^ c[15])) == 0 || (result &= ~(t[32+17] ^ c[11])) == 0 ||      // t[32+45],60,34,50 <=> C38,26,16,12 ?
      (result &= ~(t[32+23] ^ c[59])) == 0 || (result &= ~(t[32+15] ^ c[61])) == 0 ||
      (result &= ~(t[32+29] ^ c[41])) == 0 || (result &= ~(t[32+ 5] ^ c[47])) == 0 ||
      (result &= ~(t[32+25] ^ c[ 9])) == 0 || (result &= ~(t[32+19] ^ c[27])) == 0 ||
      (result &= ~(t[32+ 9] ^ c[13])) == 0 || (result &= ~(t[32+ 0] ^ c[ 7])) == 0) return result;

  if ((result &= ~(t[32+ 7] ^ c[63])) == 0 || (result &= ~(t[32+13] ^ c[45])) == 0 ||
      (result &= ~(t[32+24] ^ c[ 1])) == 0 || (result &= ~(t[32+ 2] ^ c[23])) == 0 ||
      (result &= ~(t[32+ 3] ^ c[31])) == 0 || (result &= ~(t[32+28] ^ c[33])) == 0 ||
      (result &= ~(t[32+10] ^ c[21])) == 0 || (result &= ~(t[32+18] ^ c[19])) == 0 ||
      (result &= ~(t[32+31] ^ c[57])) == 0 || (result &= ~(t[32+11] ^ c[29])) == 0 ||
      (result &= ~(t[32+21] ^ c[43])) == 0 || (result &= ~(t[32+ 6] ^ c[55])) == 0 ||
      (result &= ~(t[32+ 4] ^ c[39])) == 0 || (result &= ~(t[32+26] ^ c[17])) == 0 ||
      (result &= ~(t[32+14] ^ c[53])) == 0 || (result &= ~(t[32+20] ^ c[35])) == 0) return result;

  if ((result &= ~(t[ 0+ 8] ^ c[ 4])) == 0 || (result &= ~(t[ 0+16] ^ c[ 2])) == 0 ||
      (result &= ~(t[ 0+22] ^ c[50])) == 0 || (result &= ~(t[ 0+30] ^ c[48])) == 0 ||
      (result &= ~(t[ 0+12] ^ c[36])) == 0 || (result &= ~(t[ 0+27] ^ c[24])) == 0 ||
      (result &= ~(t[ 0+ 1] ^ c[14])) == 0 || (result &= ~(t[ 0+17] ^ c[10])) == 0 ||
      (result &= ~(t[ 0+23] ^ c[58])) == 0 || (result &= ~(t[ 0+15] ^ c[60])) == 0 ||
      (result &= ~(t[ 0+29] ^ c[40])) == 0 || (result &= ~(t[ 0+ 5] ^ c[46])) == 0 ||
      (result &= ~(t[ 0+25] ^ c[ 8])) == 0 || (result &= ~(t[ 0+19] ^ c[26])) == 0 ||
      (result &= ~(t[ 0+ 9] ^ c[12])) == 0 || (result &= ~(t[ 0+ 0] ^ c[ 6])) == 0) return result;

  if ((result &= ~(t[ 0+ 7] ^ c[62])) == 0 || (result &= ~(t[ 0+13] ^ c[44])) == 0 ||
      (result &= ~(t[ 0+24] ^ c[ 0])) == 0 || (result &= ~(t[ 0+ 2] ^ c[22])) == 0 ||
      (result &= ~(t[ 0+ 3] ^ c[30])) == 0 || (result &= ~(t[ 0+28] ^ c[32])) == 0 ||
      (result &= ~(t[ 0+10] ^ c[20])) == 0 || (result &= ~(t[ 0+18] ^ c[18])) == 0 ||
      (result &= ~(t[ 0+31] ^ c[56])) == 0 || (result &= ~(t[ 0+11] ^ c[28])) == 0 ||
      (result &= ~(t[ 0+21] ^ c[42])) == 0 || (result &= ~(t[ 0+ 6] ^ c[54])) == 0 ||
      (result &= ~(t[ 0+ 4] ^ c[38])) == 0 || (result &= ~(t[ 0+26] ^ c[16])) == 0 ||
      (result &= ~(t[ 0+14] ^ c[52])) == 0 || (result &= ~(t[ 0+20] ^ c[34])) == 0) return result;

  return result;
}

__device__ __host__ __forceinline__ sboxes_bitslice sboxes_desslice_eval64(sboxes_bitslice (&t)[64], const sboxes_block c) {
  sboxes_bitslice result = sboxes_sliceon;

  if ((result &= ~(t[32+ 8] ^ ((c & (1ULL <<  5)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 || 
      (result &= ~(t[32+16] ^ ((c & (1ULL <<  3)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+22] ^ ((c & (1ULL << 51)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+30] ^ ((c & (1ULL << 49)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+12] ^ ((c & (1ULL << 37)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+27] ^ ((c & (1ULL << 25)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 1] ^ ((c & (1ULL << 15)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+17] ^ ((c & (1ULL << 11)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||      // t[32+45],60,34,50 <=> C38,26,16,12 ?
      (result &= ~(t[32+23] ^ ((c & (1ULL << 59)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+15] ^ ((c & (1ULL << 61)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+29] ^ ((c & (1ULL << 41)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 5] ^ ((c & (1ULL << 47)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+25] ^ ((c & (1ULL <<  9)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+19] ^ ((c & (1ULL << 27)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 9] ^ ((c & (1ULL << 13)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 0] ^ ((c & (1ULL <<  7)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0) return result;

  if ((result &= ~(t[32+ 7] ^ ((c & (1ULL << 63)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+13] ^ ((c & (1ULL << 45)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+24] ^ ((c & (1ULL <<  1)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 2] ^ ((c & (1ULL << 23)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 3] ^ ((c & (1ULL << 31)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+28] ^ ((c & (1ULL << 33)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+10] ^ ((c & (1ULL << 21)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+18] ^ ((c & (1ULL << 19)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+31] ^ ((c & (1ULL << 57)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+11] ^ ((c & (1ULL << 29)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+21] ^ ((c & (1ULL << 43)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 6] ^ ((c & (1ULL << 55)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+ 4] ^ ((c & (1ULL << 39)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+26] ^ ((c & (1ULL << 17)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+14] ^ ((c & (1ULL << 53)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[32+20] ^ ((c & (1ULL << 35)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0) return result;

  if ((result &= ~(t[ 0+ 8] ^ ((c & (1ULL <<  4)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+16] ^ ((c & (1ULL <<  2)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+22] ^ ((c & (1ULL << 50)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+30] ^ ((c & (1ULL << 48)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+12] ^ ((c & (1ULL << 36)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+27] ^ ((c & (1ULL << 24)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 1] ^ ((c & (1ULL << 14)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+17] ^ ((c & (1ULL << 10)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+23] ^ ((c & (1ULL << 58)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+15] ^ ((c & (1ULL << 60)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+29] ^ ((c & (1ULL << 40)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 5] ^ ((c & (1ULL << 46)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+25] ^ ((c & (1ULL <<  8)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+19] ^ ((c & (1ULL << 26)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 9] ^ ((c & (1ULL << 12)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 0] ^ ((c & (1ULL <<  6)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0) return result;

  if ((result &= ~(t[ 0+ 7] ^ ((c & (1ULL << 62)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+13] ^ ((c & (1ULL << 44)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+24] ^ ((c & (1ULL <<  0)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 2] ^ ((c & (1ULL << 22)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 3] ^ ((c & (1ULL << 30)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+28] ^ ((c & (1ULL << 32)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+10] ^ ((c & (1ULL << 20)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+18] ^ ((c & (1ULL << 18)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+31] ^ ((c & (1ULL << 56)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+11] ^ ((c & (1ULL << 28)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+21] ^ ((c & (1ULL << 42)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 6] ^ ((c & (1ULL << 54)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+ 4] ^ ((c & (1ULL << 38)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+26] ^ ((c & (1ULL << 16)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+14] ^ ((c & (1ULL << 52)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0 ||
      (result &= ~(t[ 0+20] ^ ((c & (1ULL << 34)) != 0 ? sboxes_sliceon : sboxes_sliceoff))) == 0) return result;

  return result;
}

__device__ __host__ __forceinline__ void sboxes_desslice_set(sboxes_bitslice (&t)[64], const sboxes_bitslice (&p)[64]) {
  // L = IP_32..1
	t[32+ 0] = p[ 6];	t[32+ 1] = p[14];	t[32+ 2] = p[22];	t[32+ 3] = p[30];
	t[32+ 4] = p[38];	t[32+ 5] = p[46];	t[32+ 6] = p[54];	t[32+ 7] = p[62];
	t[32+ 8] = p[ 4];	t[32+ 9] = p[12];	t[32+10] = p[20];	t[32+11] = p[28];
	t[32+12] = p[36];	t[32+13] = p[44];	t[32+14] = p[52];	t[32+15] = p[60];
	t[32+16] = p[ 2];	t[32+17] = p[10];	t[32+18] = p[18];	t[32+19] = p[26];
	t[32+20] = p[34];	t[32+21] = p[42];	t[32+22] = p[50];	t[32+23] = p[58];
	t[32+24] = p[ 0];	t[32+25] = p[ 8];	t[32+26] = p[16];	t[32+27] = p[24];
	t[32+28] = p[32];	t[32+29] = p[40];	t[32+30] = p[48];	t[32+31] = p[56];

  // R = IP_64..33
	t[ 0+ 0] = p[ 7];	t[ 0+ 1] = p[15];	t[ 0+ 2] = p[23];	t[ 0+ 3] = p[31];
	t[ 0+ 4] = p[39];	t[ 0+ 5] = p[47];	t[ 0+ 6] = p[55];	t[ 0+ 7] = p[63];
	t[ 0+ 8] = p[ 5];	t[ 0+ 9] = p[13];	t[ 0+10] = p[21];	t[ 0+11] = p[29];
	t[ 0+12] = p[37];	t[ 0+13] = p[45];	t[ 0+14] = p[53];	t[ 0+15] = p[61];
	t[ 0+16] = p[ 3];	t[ 0+17] = p[11];	t[ 0+18] = p[19];	t[ 0+19] = p[27];
	t[ 0+20] = p[35];	t[ 0+21] = p[43];	t[ 0+22] = p[51];	t[ 0+23] = p[59];
	t[ 0+24] = p[ 1];	t[ 0+25] = p[ 9];	t[ 0+26] = p[17];	t[ 0+27] = p[25];
	t[ 0+28] = p[33];	t[ 0+29] = p[41];	t[ 0+30] = p[49];	t[ 0+31] = p[57];
  return;
}

__device__ __host__ __forceinline__ void sboxes_desslice_set64(sboxes_bitslice (&t)[64], const sboxes_block p64) {
  #if __CUDA_ARCH__ >= 500
  //__constant__  
  #endif /* __CUDA_ARCH__ >= 500 */
  static const uint8_t IP_Bit2BitMap[] = {
    /* R00-R15 */ 7, 15, 23, 31, 39, 47, 55, 63,  5, 13, 21, 29, 37, 45, 53, 61, 
    /* R16-R31 */ 3, 11, 19, 27, 35, 43, 51, 59,  1,  9, 17, 25, 33, 41, 49, 57, 
    /* R32-R47 */ 6, 14, 22, 30, 38, 46, 54, 62,  4, 12, 20, 28, 36, 44, 52, 60, 
    /* R48-R63 */ 2, 10, 18, 26, 34, 42, 50, 58,  0,  8, 16, 24, 32, 40, 48, 56,
  };
  int i;
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < sizeof(t)/sizeof(t[0]); ++i) t[i] = (p64 & (1ULL << IP_Bit2BitMap[i])) != 0 ? sboxes_sliceon : sboxes_sliceoff;
  return;
}

__device__ __host__ __forceinline__ void sboxes_desslice_get(sboxes_bitslice (&c)[64], const sboxes_bitslice (&t)[64]) {
	c[ 6] = t[ 0+ 0];	c[14] = t[ 0+ 1];	c[22] = t[ 0+ 2];	c[30] = t[ 0+ 3];
	c[38] = t[ 0+ 4];	c[46] = t[ 0+ 5];	c[54] = t[ 0+ 6];	c[62] = t[ 0+ 7];
	c[ 4] = t[ 0+ 8];	c[12] = t[ 0+ 9];	c[20] = t[ 0+10];	c[28] = t[ 0+11];
	c[36] = t[ 0+12];	c[44] = t[ 0+13];	c[52] = t[ 0+14];	c[60] = t[ 0+15];
	c[ 2] = t[ 0+16];	c[10] = t[ 0+17];	c[18] = t[ 0+18];	c[26] = t[ 0+19];
	c[34] = t[ 0+20];	c[42] = t[ 0+21];	c[50] = t[ 0+22];	c[58] = t[ 0+23];
	c[ 0] = t[ 0+24];	c[ 8] = t[ 0+25];	c[16] = t[ 0+26];	c[24] = t[ 0+27];
	c[32] = t[ 0+28];	c[40] = t[ 0+29];	c[48] = t[ 0+30];	c[56] = t[ 0+31];

	c[ 7] = t[32+ 0];	c[15] = t[32+ 1];	c[23] = t[32+ 2];	c[31] = t[32+ 3];
	c[39] = t[32+ 4];	c[47] = t[32+ 5];	c[55] = t[32+ 6];	c[63] = t[32+ 7];
	c[ 5] = t[32+ 8];	c[13] = t[32+ 9];	c[21] = t[32+10];	c[29] = t[32+11];
	c[37] = t[32+12];	c[45] = t[32+13];	c[53] = t[32+14];	c[61] = t[32+15];
	c[ 3] = t[32+16];	c[11] = t[32+17];	c[19] = t[32+18];	c[27] = t[32+19];
	c[35] = t[32+20];	c[43] = t[32+21];	c[51] = t[32+22];	c[59] = t[32+23];
	c[ 1] = t[32+24];	c[ 9] = t[32+25];	c[17] = t[32+26];	c[25] = t[32+27];
	c[33] = t[32+28];	c[41] = t[32+29];	c[49] = t[32+30];	c[57] = t[32+31];
  return;
}

__device__ __host__ __forceinline__ sboxes_block sboxes_desslice_get64(const sboxes_bitslice (&t)[64]) {
  #if __CUDA_ARCH__ >= 500
  //__constant__  
  #endif /* __CUDA_ARCH__ >= 500 */
  static const uint8_t FP_Bit2BitMap[] = {
     /* R00-R15 */ 6, 14, 22, 30, 38, 46, 54, 62,  4, 12, 20, 28, 36, 44, 52, 60, 
     /* R16-R31 */ 2, 10, 18, 26, 34, 42, 50, 58,  0,  8, 16, 24, 32, 40, 48, 56, 
     /* R32-R47 */ 7, 15, 23, 31, 39, 47, 55, 63,  5, 13, 21, 29, 37, 45, 53, 61, 
     /* R48-R63 */ 3, 11, 19, 27, 35, 43, 51, 59,  1,  9, 17, 25, 33, 41, 49, 57, 
  };
  sboxes_block ret64 = 0;
  int i = sizeof(t)/sizeof(t[0]);

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (--i >= 0) if (t[i] != sboxes_sliceoff) ret64 |= 1ULL << FP_Bit2BitMap[i];
  return ret64;
}

/*
 * Bitslice implementation of DES.
 *
 * Checks that the plaintext bits p[ 0] .. p[63]
 * encrypt to the ciphertext bits c[ 0] .. c[63]
 * given the key bits k[ 0] .. k[55]
 */
__device__ __host__ __forceinline__ sboxes_bitslice sboxes_deseval(
    const sboxes_bitslice (&p)[64], const sboxes_bitslice (&c)[64], const sboxes_bitslice (&k)[56]) {
  sboxes_bitslice t[64], result = sboxes_sliceon;

  sboxes_desslice_set(t, p);
  //sboxes_desslice_enc(t, k);
  //return sboxes_desslice_eval(t, c);
  
  //sboxes_desslice_enc(t, k);
   _r1(&t[32], &t[0], k);   _r2(&t[32], &t[0], k);   _r3(&t[32], &t[0], k);   _r4(&t[32], &t[0], k);
   _r5(&t[32], &t[0], k);   _r6(&t[32], &t[0], k);   _r7(&t[32], &t[0], k);   _r8(&t[32], &t[0], k);
   _r9(&t[32], &t[0], k);  _r10(&t[32], &t[0], k);  _r11(&t[32], &t[0], k);  _r12(&t[32], &t[0], k);
  _r13(&t[32], &t[0], k);  _r14(&t[32], &t[0], k);  // _r15(&t[32], &t[0], k);  _r16(&t[32], &t[0], k); // /

  /*  L[64]..33 */
  _s1(t[31]^k[33], t[ 0]^k[54], t[ 1]^k[12], t[ 2]^k[46], t[ 3]^k[24], t[ 4]^k[27], &t[32+ 8], &t[32+16], &t[32+22], &t[32+30]);
  if ((result &= ~((t[32+ 8]^c[ 5])|(t[32+16]^c[ 3])|(t[32+22]^c[51])|(t[32+30]^c[49]))) == sboxes_sliceoff) return result;

  _s2(t[ 3]^k[13], t[ 4]^k[17], t[ 5]^k[40], t[ 6]^k[34], t[ 7]^k[25], t[ 8]^k[ 5], &t[32+12], &t[32+27], &t[32+ 1], &t[32+17]);
  if ((result &= ~((t[32+12]^c[37])|(t[32+27]^c[25])|(t[32+ 1]^c[15])|(t[32+17]^c[11]))) == sboxes_sliceoff) return result; // t[32+45],60,34,50 <=> C38,26,16,12 ?

  _s3(t[ 7]^k[39], t[ 8]^k[11], t[ 9]^k[19], t[10]^k[20], t[11]^k[ 3], t[12]^k[48], &t[32+23], &t[32+15], &t[32+29], &t[32+ 5]);
  if ((result &= ~((t[32+23]^c[59])|(t[32+15]^c[61])|(t[32+29]^c[41])|(t[32+ 5]^c[47]))) == sboxes_sliceoff) return result;

  _s4(t[11]^k[47], t[12]^k[41], t[13]^k[10], t[14]^k[18], t[15]^k[26], t[16]^k[ 6], &t[32+25], &t[32+19], &t[32+ 9], &t[32+ 0]);
  if ((result &= ~((t[32+25]^c[ 9])|(t[32+19]^c[27])|(t[32+ 9]^c[13])|(t[32+ 0]^c[ 7]))) == sboxes_sliceoff) return result;

  _s5(t[15]^k[22], t[16]^k[44], t[17]^k[ 7], t[18]^k[49], t[19]^k[ 9], t[20]^k[38], &t[32+ 7], &t[32+13], &t[32+24], &t[32+ 2]);
  if ((result &= ~((t[32+ 7]^c[63])|(t[32+13]^c[45])|(t[32+24]^c[ 1])|(t[32+ 2]^c[23]))) == sboxes_sliceoff) return result;

  _s6(t[19]^k[ 0], t[20]^k[15], t[21]^k[37], t[22]^k[50], t[23]^k[21], t[24]^k[16], &t[32+ 3], &t[32+28], &t[32+10], &t[32+18]);
  if ((result &= ~((t[32+ 3]^c[31])|(t[32+28]^c[33])|(t[32+10]^c[21])|(t[32+18]^c[19]))) == sboxes_sliceoff) return result;

  _s7(t[23]^k[43], t[24]^k[23], t[25]^k[ 8], t[26]^k[45], t[27]^k[28], t[28]^k[51], &t[32+31], &t[32+11], &t[32+21], &t[32+ 6]);
  if ((result &= ~((t[32+31]^c[57])|(t[32+11]^c[29])|(t[32+21]^c[43])|(t[32+ 6]^c[55]))) == sboxes_sliceoff) return result;

  _s8(t[27]^k[ 2], t[28]^k[29], t[29]^k[30], t[30]^k[42], t[31]^k[52], t[ 0]^k[14], &t[32+ 4], &t[32+26], &t[32+14], &t[32+20]);
  if ((result &= ~((t[32+ 4]^c[39])|(t[32+26]^c[17])|(t[32+14]^c[53])|(t[32+20]^c[35]))) == sboxes_sliceoff) return result;

  /* t[32]..1 */
  _s1(t[32+31]^k[40], t[32+ 0]^k[ 4], t[32+ 1]^k[19], t[32+ 2]^k[53], t[32+ 3]^k[ 6], t[32+ 4]^k[34], &t[ 8], &t[16], &t[22], &t[30]);
  if ((result &= ~((t[ 8]^c[ 4])|(t[16]^c[ 2])|(t[22]^c[50])|(t[30]^c[48]))) == sboxes_sliceoff) return result;

  _s2(t[32+ 3]^k[20], t[32+ 4]^k[24], t[32+ 5]^k[47], t[32+ 6]^k[41], t[32+ 7]^k[32], t[32+ 8]^k[12], &t[12], &t[27], &t[ 1], &t[17]);
  if ((result &= ~((t[12]^c[36])|(t[27]^c[24])|(t[ 1]^c[14])|(t[17]^c[10]))) == sboxes_sliceoff) return result;

  _s3(t[32+ 7]^k[46], t[32+ 8]^k[18], t[32+ 9]^k[26], t[32+10]^k[27], t[32+11]^k[10], t[32+12]^k[55], &t[23], &t[15], &t[29], &t[ 5]);
  if ((result &= ~((t[23]^c[58])|(t[15]^c[60])|(t[29]^c[40])|(t[ 5]^c[46]))) == sboxes_sliceoff) return result;

  _s4(t[32+11]^k[54], t[32+12]^k[48], t[32+13]^k[17], t[32+14]^k[25], t[32+15]^k[33], t[32+16]^k[13], &t[25], &t[19], &t[ 9], &t[ 0]);
  if ((result &= ~((t[25]^c[ 8])|(t[19]^c[26])|(t[ 9]^c[12])|(t[ 0]^c[ 6]))) == sboxes_sliceoff) return result;

  _s5(t[32+15]^k[29], t[32+16]^k[51], t[32+17]^k[14], t[32+18]^k[ 1], t[32+19]^k[16], t[32+20]^k[45], &t[ 7], &t[13], &t[24], &t[ 2]);
  if ((result &= ~((t[ 7]^c[62])|(t[13]^c[44])|(t[24]^c[ 0])|(t[ 2]^c[22]))) == sboxes_sliceoff) return result;

  _s6(t[32+19]^k[ 7], t[32+20]^k[22], t[32+21]^k[44], t[32+22]^k[ 2], t[32+23]^k[28], t[32+24]^k[23], &t[ 3], &t[28], &t[10], &t[18]);
  if ((result &= ~((t[ 3]^c[30])|(t[28]^c[32])|(t[10]^c[20])|(t[18]^c[18]))) == sboxes_sliceoff) return result;

  _s7(t[32+23]^k[50], t[32+24]^k[30], t[32+25]^k[15], t[32+26]^k[52], t[32+27]^k[35], t[32+28]^k[31], &t[31], &t[11], &t[21], &t[ 6]);
  if ((result &= ~((t[31]^c[56])|(t[11]^c[28])|(t[21]^c[42])|(t[ 6]^c[54]))) == sboxes_sliceoff) return result;

  _s8(t[32+27]^k[ 9], t[32+28]^k[36], t[32+29]^k[37], t[32+30]^k[49], t[32+31]^k[ 0], t[32+ 0]^k[21], &t[ 4], &t[26], &t[14], &t[20]);
  /*if (*/result &= ~((t[ 4]^c[38])|(t[26]^c[16])|(t[14]^c[52])|(t[20]^c[34]))/*) == sboxes_sliceoff) return result*/;

  return result;
}
   
/*
 * Return an odd parity bit for a number.
 */
__device__ __host__ __forceinline__ static int odd_parity(int	n) {
	int	parity = 1;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
	while (n != 0) {
    if ((n & 1) != 0) parity ^= 1;
    n >>= 1;
	}
	return (parity);
}

__device__ __host__ __forceinline__ sboxes_deskey sboxes_clr_parity(const sboxes_deskey key64) {
  return key64 & 0xFEFEFEFEFEFEFEFEULL;
}

__device__ __host__ __forceinline__ sboxes_deskey sboxes_set_parity(const sboxes_deskey key64) {
  sboxes_deskey tmp = key64 & 0xFEFEFEFEFEFEFEFEULL;
  tmp = ((sboxes_deskey)odd_parity((tmp >> 57) & 0x7F)) << 56 |
        ((sboxes_deskey)odd_parity((tmp >> 49) & 0x7F)) << 48 |
        ((sboxes_deskey)odd_parity((tmp >> 41) & 0x7F)) << 40 |
        ((sboxes_deskey)odd_parity((tmp >> 33) & 0x7F)) << 32 |
        ((sboxes_deskey)odd_parity((tmp >> 25) & 0x7F)) << 24 |
        ((sboxes_deskey)odd_parity((tmp >> 17) & 0x7F)) << 16 |
        ((sboxes_deskey)odd_parity((tmp >>  9) & 0x7F)) <<  8 |
        ((sboxes_deskey)odd_parity((tmp >>  1) & 0x7F)) <<  0;
  return tmp;
}

/*
 * The key has been found.
 * Turn the value into something readable, and print it out.
 */
__device__ __host__ __forceinline__ sboxes_deskey sboxes_key_found(const sboxes_bitslice (&key)[56], const sboxes_bitslice	slice) {
	int i;
  int j;
  int kc;
  sboxes_deskey val;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
	for (i = 0; i < 8; i++) {

    kc = 0;

    #if __CUDA_ARCH__ >= 500
    #pragma unroll
    #endif /* __CUDA_ARCH__ >= 500 */
    for (j = 0; j < 7; j++) {
      if ((key[49 - i*7 + j] & slice) != 0) kc |= (1 << j);
    }
    kc = (kc << 1) | odd_parity(kc);
    val = (val << 8) | (kc & 0xFF);
	}
  return val;
}

/*
 * Set the bit slice pattern on the low key bits.
 */
__device__ __host__ __forceinline__ sboxes_deskey sboxes_set_low_keys(sboxes_bitslice	(&key)[56]) {
  sboxes_bitslice *p_key = &key[55];
  sboxes_deskey val = 0;
	int	w;
  int i;
  int j;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (w = 1; w < sboxes_slicebitsize /*bitsizeof(*key)*/; w <<= 1) {
    // w==8 : 00000000111111110000000011111111
    // w==4 : 00001111000011110000111100001111
    // w==2 : 00110011001100110011001100110011
    // w==1 : 01010101010101010101010101010101

    #if __CUDA_ARCH__ >= 500
    #pragma unroll
    #endif /* __CUDA_ARCH__ >= 500 */
    for (i = 0; i < sboxes_slicebitsize /*bitsizeof(*key)*/; i += w*2) {
      // w==8 : 0000000011111111
      // w==4 : 00001111
      // w==2 : 0011
      // w==1 : 01
      #if __CUDA_ARCH__ >= 500
      #pragma unroll
      #endif /* __CUDA_ARCH__ >= 500 */
      for (j = 0; j < w; j++) *p_key = (*p_key << 1);

      #if __CUDA_ARCH__ >= 500
      #pragma unroll
      #endif /* __CUDA_ARCH__ >= 500 */
      for (j = 0; j < w; j++) *p_key = (*p_key << 1) | 1;
    }

    val = (val<<1) | 1;
    if (((p_key-key) % 7) == 0) val <<= 1;
    --p_key;
	}

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while ((p_key-key) >= 0) {
    val = (val << 1) | (*p_key & 1);
    if (((p_key-key) % 7) == 0) val <<= 1;
    --p_key;
  }
  return val;
}


__device__ __host__ __forceinline__ sboxes_deskey sboxes_add_keys(sboxes_bitslice (&key)[56], const sboxes_deskey val56) {
  const size_t max_bitcount = 56 - sboxes_slicebitcount /* bitcountof(*key) */;
  sboxes_deskey   carry = 0;
  sboxes_deskey   ret64 = 0;
  int i;

  if (val56 == 1) return sboxes_inc_keys(key);
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = max_bitcount-1; i >= 0; --i) {
    carry = (carry << 1 ) | (key[i] & 1);
  }

  carry += val56;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < max_bitcount; ++i) {
    if ((i % 7) == 0) ret64 >>= 1;
    key[i] = ~((carry & 1)-1);
    carry >>= 1;
    ret64 = (ret64 >> 1) | ((sboxes_deskey)(key[i] & 1)) << 63;
  }

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (i < 56) {
    if ((i % 7) == 0) ret64 >>= 1;
    ret64 = (ret64 >> 1) | 1ULL << 63;
    ++i;
  }
  return ret64;
}

__device__ __host__ __forceinline__ sboxes_deskey sboxes_inc_keys(sboxes_bitslice (&key)[56]) {
  const size_t max_bitcount = 56 - sboxes_slicebitcount /* bitcountof(*key) */;
  sboxes_deskey   ret64 = 0;
  int i;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < max_bitcount; ++i) {
    if ((i % 7) == 0) ret64 >>= 1;
    key[i] = ~key[i];
    ret64 = (ret64 >> 1) | ((sboxes_deskey)(key[i] & 1)) << 63;
    if (key[i]) break;
  }

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (i < 56) {
    if ((i % 7) == 0) ret64 >>= 1;
    ret64 = (ret64 >> 1) | ((sboxes_deskey)(key[i] & 1)) << 63;
    ++i;
  }

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (i < 56) {
    if ((i % 7) == 0) ret64 >>= 1;
    ret64 = (ret64 >> 1) | 1ULL << 63;
    ++i;
  }
  return ret64;
}

__device__ __host__ __forceinline__ void sboxes_add_keys_fast(sboxes_bitslice (&key)[56], const sboxes_deskey val56) {
  const size_t max_bitcount = 56 - sboxes_slicebitcount /* bitcountof(*key) */;
  sboxes_deskey   carry = 0;
  int i;

  if (val56 == 1) return sboxes_inc_keys_fast(key);
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = max_bitcount-1; i >= 0; --i) {
    carry = (carry << 1 ) | (key[i] & 1);
  }

  carry += val56;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < max_bitcount; ++i) {
    key[i] = ~((carry & 1)-1);
    carry >>= 1;
  }
  return;
}

__device__ __host__ __forceinline__ void sboxes_inc_keys_fast(sboxes_bitslice (&key)[56]) {
  const size_t max_bitcount = 56 - sboxes_slicebitcount /* bitcountof(*key) */;
  int i;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll 56
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < max_bitcount; ++i) if (key[i] = ~key[i]) break;
  return;
}

/*
 * Using the first 56 - bitlength_log2 entries of the key as a counter,
 * iterate to the next value.
 */
__device__ __host__ __forceinline__ sboxes_deskey sboxes_inc_high_keys(sboxes_bitslice *key) {
  int i = 55 - sboxes_slicebitcount /* bitcountof(*key) */;
  sboxes_deskey val = 0;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (i >= 0) {
    key[i] = ~key[i];
    val = (val<<1) | (key[i] & 1);
    if ((i % 7) == 0) val <<= 1;
    if (key[i--] & 1) break; /* =0 -> CARRY */
  }

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  while (i >= 0) {
    val = (val<<1) | (key[i] & 1);
    if ((i % 7) == 0) val <<= 1;
    --i;
  }

//  key = &key[55 - sboxes_slicebitcount];
//  while (*key != 0)  *key-- = 0;
//  key = ~(key = 0);
  return val;
}

__device__ __host__ __forceinline__ void sboxes_set_key64(sboxes_bitslice (&k)[56], const sboxes_deskey k64) {
  sboxes_deskey tmp = k64;
  int i;
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < 56; ++i) {
    if ((i % 7) == 0) tmp >>= 1;
    k[i] = ~((tmp&1)-1);
    tmp >>= 1;
  }
  return;
}

__device__ __host__ __forceinline__ sboxes_deskey sboxes_get_key64(const sboxes_bitslice (&k)[56], const sboxes_bitslice sliceMask) {
  sboxes_deskey key64 = 0;
  int i;
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 55; i >= 0; --i) {
    key64 <<= 1;
    key64 |= ((k[i] & sliceMask) != 0 ? 1:0);
    if ((i % 7) == 0) {
      key64 = (key64 << 1) | ((odd_parity(key64 & 0x7F)) & 1);
    }
  }
  return key64;
}

__device__ __host__ __forceinline__ void sboxes_set_data(sboxes_bitslice (&p)[64], const sboxes_block p64) {
  sboxes_block tmp = p64;
  int i;
  
  tmp = p64;

  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < 64; ++i) {
    p[i] = ~((tmp&1)-1);
    tmp >>= 1;
  }
  return;
}

__device__ __host__ __forceinline__ sboxes_block sboxes_get_data(const sboxes_bitslice (&p)[64]) {
  sboxes_block val64;
  int i;
  
  #if __CUDA_ARCH__ >= 500
  #pragma unroll
  #endif /* __CUDA_ARCH__ >= 500 */
  for (i = 0; i < 64; ++i) val64 = (val64 >> 1) | (p[i] != 0 ? (1ULL<<63) : 0ULL);
  return val64;
}

/*
 * Unroll the bits contained in the plaintext, ciphertext, and key values.
 */
__device__ __host__ __forceinline__ void sboxes_unroll_bits(
	sboxes_bitslice		 *pp,
	sboxes_bitslice		 *pc,
	sboxes_bitslice		 *pk,
  const sboxes_block  p, 
  const sboxes_block  c,
  const sboxes_deskey k
) {
  sboxes_block tmp;
	int i;

  if (pp != NULL) {
    tmp = p;
    #if __CUDA_ARCH__ >= 500
    #pragma unroll
    #endif /* __CUDA_ARCH__ >= 500 */
    for (i = 0; i < 64; ++i) {
      pp[i] = ~((tmp&1)-1); tmp >>= 1;
    }
  }

  if (pc != NULL) {
    tmp = c;
    #if __CUDA_ARCH__ >= 500
    #pragma unroll
    #endif /* __CUDA_ARCH__ >= 500 */
    for (i = 0; i < 64; ++i) {
      pc[i] = ~((tmp&1)-1); tmp >>= 1;
    }
  }
  
  if (pk != NULL) {
    tmp = k;
    for (i = 0; i < 56; ++i) {
      if ((i % 7) == 0) tmp >>= 1;
      pk[i] = ~((tmp&1)-1); tmp >>= 1;
    }
  }
  return;
}
