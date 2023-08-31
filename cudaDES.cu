
//#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif /* WIN32 */
#include "cudaDES.h"

#include "sboxes_deseval.h"
#include "sboxes_deseval.c"

// Regular implementation

const unsigned char* _plain  = (unsigned char *)"\x01\x23\x45\x67\x89\xAB\xCD\xEF";
const unsigned char* _cipher = (unsigned char *)"\x1A\x28\x6F\xAB\x84\x7A\xE5\x20";
const unsigned char* _key    = (unsigned char *)"\x00\x00\x00\x1E\xFE\xFE\xFE\xFE";
const unsigned char* plain =_plain;
const unsigned char* cipher=_cipher;
const unsigned char* key   =_key;
int64_t step_size = 1;
int decrypt = 0;
int isSerial = 0;
int verbose = 0;
int deviceId = -1;
FILE *fpLogFile = NULL;

void des_key_set_parity( unsigned char key[DES_KEY_SIZE] )
{
    int i;

    for( i = 0; i < DES_KEY_SIZE; i++ )
        key[i] = odd_parity_table[key[i] / 2];
}

/*
 * Check the given key's parity, returns 1 on failure, 0 on SUCCESS
 */
int des_key_check_key_parity( const unsigned char key[DES_KEY_SIZE] )
{
    int i;

    for( i = 0; i < DES_KEY_SIZE; i++ )
        if ( key[i] != odd_parity_table[key[i] / 2] )
            return( 1 );

    return( 0 );
}


static void des_setkey( uint32_t SK[32], const unsigned char key[DES_KEY_SIZE] )
{
    int i;
    uint32_t X, Y, T;

    GET_UINT32_BE( X, key, 0 );
    GET_UINT32_BE( Y, key, 4 );

    /*
     * Permuted Choice 1
     */
    T =  ((Y >>  4) ^ X) & 0x0F0F0F0F;  X ^= T; Y ^= (T <<  4);
    T =  ((Y      ) ^ X) & 0x10101010;  X ^= T; Y ^= (T      );

    X =   (LHs[ (X      ) & 0xF] << 3) | (LHs[ (X >>  8) & 0xF ] << 2)
        | (LHs[ (X >> 16) & 0xF] << 1) | (LHs[ (X >> 24) & 0xF ]     )
        | (LHs[ (X >>  5) & 0xF] << 7) | (LHs[ (X >> 13) & 0xF ] << 6)
        | (LHs[ (X >> 21) & 0xF] << 5) | (LHs[ (X >> 29) & 0xF ] << 4);

    Y =   (RHs[ (Y >>  1) & 0xF] << 3) | (RHs[ (Y >>  9) & 0xF ] << 2)
        | (RHs[ (Y >> 17) & 0xF] << 1) | (RHs[ (Y >> 25) & 0xF ]     )
        | (RHs[ (Y >>  4) & 0xF] << 7) | (RHs[ (Y >> 12) & 0xF ] << 6)
        | (RHs[ (Y >> 20) & 0xF] << 5) | (RHs[ (Y >> 28) & 0xF ] << 4);

    X &= 0x0FFFFFFF;
    Y &= 0x0FFFFFFF;

    /*
     * calculate subkeys
     */
    for( i = 0; i < 16; i++ )
    {
        if( i < 2 || i == 8 || i == 15 )
        {
            X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFF;
            Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFF;
        }
        else
        {
            X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFF;
            Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFF;
        }

        *SK++ =   ((X <<  4) & 0x24000000) | ((X << 28) & 0x10000000)
                | ((X << 14) & 0x08000000) | ((X << 18) & 0x02080000)
                | ((X <<  6) & 0x01000000) | ((X <<  9) & 0x00200000)
                | ((X >>  1) & 0x00100000) | ((X << 10) & 0x00040000)
                | ((X <<  2) & 0x00020000) | ((X >> 10) & 0x00010000)
                | ((Y >> 13) & 0x00002000) | ((Y >>  4) & 0x00001000)
                | ((Y <<  6) & 0x00000800) | ((Y >>  1) & 0x00000400)
                | ((Y >> 14) & 0x00000200) | ((Y      ) & 0x00000100)
                | ((Y >>  5) & 0x00000020) | ((Y >> 10) & 0x00000010)
                | ((Y >>  3) & 0x00000008) | ((Y >> 18) & 0x00000004)
                | ((Y >> 26) & 0x00000002) | ((Y >> 24) & 0x00000001);

        *SK++ =   ((X << 15) & 0x20000000) | ((X << 17) & 0x10000000)
                | ((X << 10) & 0x08000000) | ((X << 22) & 0x04000000)
                | ((X >>  2) & 0x02000000) | ((X <<  1) & 0x01000000)
                | ((X << 16) & 0x00200000) | ((X << 11) & 0x00100000)
                | ((X <<  3) & 0x00080000) | ((X >>  6) & 0x00040000)
                | ((X << 15) & 0x00020000) | ((X >>  4) & 0x00010000)
                | ((Y >>  2) & 0x00002000) | ((Y <<  8) & 0x00001000)
                | ((Y >> 14) & 0x00000808) | ((Y >>  9) & 0x00000400)
                | ((Y      ) & 0x00000200) | ((Y <<  7) & 0x00000100)
                | ((Y >>  7) & 0x00000020) | ((Y >>  3) & 0x00000011)
                | ((Y <<  2) & 0x00000004) | ((Y >> 21) & 0x00000002);
    }
}

/*
 * DES key schedule (56-bit, encryption)
 */
int des_setkey_enc( des_context *ctx, const unsigned char key[DES_KEY_SIZE] )
{
    des_setkey( ctx->sk, key );

    return( 0 );
}

/*
 * DES key schedule (56-bit, decryption)
 */
int des_setkey_dec( des_context *ctx, const unsigned char key[DES_KEY_SIZE] )
{
    int i;

    des_setkey( ctx->sk, key );

    for( i = 0; i < 16; i += 2 )
    {
        SWAP( ctx->sk[i    ], ctx->sk[30 - i] );
        SWAP( ctx->sk[i + 1], ctx->sk[31 - i] );
    }

    return( 0 );
}


/*
 * DES-ECB block encryption/decryption
 */
int des_crypt_ecb( des_context *ctx,
                    const unsigned char input[8],
                    unsigned char output[8] )
{
    int i;
    uint32_t X, Y, T, *SK;

    SK = ctx->sk;

    GET_UINT32_BE( X, input, 0 );
    GET_UINT32_BE( Y, input, 4 );

    DES_IP( X, Y );

    for( i = 0; i < 8; i++ )
    {
        DES_ROUND( Y, X );
        DES_ROUND( X, Y );
    }

    DES_FP( Y, X );

    PUT_UINT32_BE( Y, output, 0 );
    PUT_UINT32_BE( X, output, 4 );

    return( 0 );
}

#define des64to56(long_key) ((uint64_t) \
    ( (((uint64_t)(long_key)) >> 1) & (((1ULL<<7)-1) <<  0) ) | \
    ( (((uint64_t)(long_key)) >> 2) & (((1ULL<<7)-1) <<  7) ) | \
    ( (((uint64_t)(long_key)) >> 3) & (((1ULL<<7)-1) << 14) ) | \
    ( (((uint64_t)(long_key)) >> 4) & (((1ULL<<7)-1) << 21) ) | \
    ( (((uint64_t)(long_key)) >> 5) & (((1ULL<<7)-1) << 28) ) | \
    ( (((uint64_t)(long_key)) >> 6) & (((1ULL<<7)-1) << 35) ) | \
    ( (((uint64_t)(long_key)) >> 7) & (((1ULL<<7)-1) << 42) ) | \
    ( (((uint64_t)(long_key)) >> 8) & (((1ULL<<7)-1) << 49) )   \
  )

#define des56to64(short_key) ((uint64_t) \
    ( ((short_key) & (((1ULL<<7)-1) <<  0)) << 1) | \
		( ((short_key) & (((1ULL<<7)-1) <<  7)) << 2) | \
		( ((short_key) & (((1ULL<<7)-1) << 14)) << 3) | \
		( ((short_key) & (((1ULL<<7)-1) << 21)) << 4) | \
		( ((short_key) & (((1ULL<<7)-1) << 28)) << 5) | \
		( ((short_key) & (((1ULL<<7)-1) << 35)) << 6) | \
		( ((short_key) & (((1ULL<<7)-1) << 42)) << 7) | \
		( ((short_key) & (((1ULL<<7)-1) << 49)) << 8)   \
  )

#define uint8to64(key) ((uint64_t)((((((((((((((( \
    (uint64_t)(key)[0]) << 8) | \
    (uint64_t)(key)[1]) << 8) | \
    (uint64_t)(key)[2]) << 8) | \
    (uint64_t)(key)[3]) << 8) | \
    (uint64_t)(key)[4]) << 8) | \
    (uint64_t)(key)[5]) << 8) | \
    (uint64_t)(key)[6]) << 8) | \
    (uint64_t)(key)[7]) \
  )
  
#define uint64to8(buf,key) ( \
    (buf)[7] = ((key) >>  0) & 0xFF, \
    (buf)[6] = ((key) >>  8) & 0xFF, \
    (buf)[5] = ((key) >> 16) & 0xFF, \
    (buf)[4] = ((key) >> 24) & 0xFF, \
    (buf)[3] = ((key) >> 32) & 0xFF, \
    (buf)[2] = ((key) >> 40) & 0xFF, \
    (buf)[1] = ((key) >> 48) & 0xFF, \
    (buf)[0] = ((key) >> 56) & 0xFF, \
    &(buf)[0] \
  )

#define uint64rev8(key) (((((((((((((((( \
    ((key) >>  0]) & 0xFF) << 8) | \
    ((key) >>  8]) & 0xFF) << 8) | \
    ((key) >> 16]) & 0xFF) << 8) | \
    ((key) >> 24]) & 0xFF) << 8) | \
    ((key) >> 32]) & 0xFF) << 8) | \
    ((key) >> 40]) & 0xFF) << 8) | \
    ((key) >> 48]) & 0xFF) << 8) | \
    ((key) >> 56]) & 0xFF) << 0 \
  )

#define bin16oct64(x) ( \
  (((uint64_t)(x))&(1<<0))*(1<<0)+\
  (((uint64_t)(x))&(1<<1))*(1<<2)+\
  (((uint64_t)(x))&(1<<2))*(1<<4)+\
  (((uint64_t)(x))&(1<<3))*(1<<6)+\
  (((uint64_t)(x))&(1<<4))*(1<<8)+\
  (((uint64_t)(x))&(1<<5))*(1<<10)+\
  (((uint64_t)(x))&(1<<6))*(1<<12)+\
  (((uint64_t)(x))&(1<<7))*(1<<14)+\
  (((uint64_t)(x))&(1<<8))*(1<<16)+\
  (((uint64_t)(x))&(1<<9))*(1<<18)+\
  (((uint64_t)(x))&(1<<10))*(1<<20)+\
  (((uint64_t)(x))&(1<<11))*(1<<22)+\
  (((uint64_t)(x))&(1<<12))*(1<<24)+\
  (((uint64_t)(x))&(1<<13))*(1<<26)+\
  (((uint64_t)(x))&(1<<14))*(1<<28)+\
  (((uint64_t)(x))&(1<<15))*(1<<30))

/*
__device__ static inline void des_setkey_cuda(uint32_t SK[32], const uint64_t key)
{
    int i;
    uint32_t X, Y, T;
    //
    // PC1: left and right halves bit-swap
    //
    static const uint32_t LHs[16] =
    {
        0x00000000, 0x00000001, 0x00000100, 0x00000101,
        0x00010000, 0x00010001, 0x00010100, 0x00010101,
        0x01000000, 0x01000001, 0x01000100, 0x01000101,
        0x01010000, 0x01010001, 0x01010100, 0x01010101
    };

    static const uint32_t RHs[16] =
    {
        0x00000000, 0x01000000, 0x00010000, 0x01010000,
        0x00000100, 0x01000100, 0x00010100, 0x01010100,
        0x00000001, 0x01000001, 0x00010001, 0x01010001,
        0x00000101, 0x01000101, 0x00010101, 0x01010101,
    };


    X = key>>32;              // GET_UINT32_BE( X, key, 0 );
    Y = key&0xFFFFFFFF;       // GET_UINT32_BE( Y, key, 4 );

    //
    // Permuted Choice 1
    //
    T =  ((Y >>  4) ^ X) & 0x0F0F0F0F;  X ^= T; Y ^= (T <<  4);
    T =  ((Y      ) ^ X) & 0x10101010;  X ^= T; Y ^= (T      );

    X =   (LHs[ (X      ) & 0xF] << 3) | (LHs[ (X >>  8) & 0xF ] << 2)
        | (LHs[ (X >> 16) & 0xF] << 1) | (LHs[ (X >> 24) & 0xF ]     )
        | (LHs[ (X >>  5) & 0xF] << 7) | (LHs[ (X >> 13) & 0xF ] << 6)
        | (LHs[ (X >> 21) & 0xF] << 5) | (LHs[ (X >> 29) & 0xF ] << 4);

    Y =   (RHs[ (Y >>  1) & 0xF] << 3) | (RHs[ (Y >>  9) & 0xF ] << 2)
        | (RHs[ (Y >> 17) & 0xF] << 1) | (RHs[ (Y >> 25) & 0xF ]     )
        | (RHs[ (Y >>  4) & 0xF] << 7) | (RHs[ (Y >> 12) & 0xF ] << 6)
        | (RHs[ (Y >> 20) & 0xF] << 5) | (RHs[ (Y >> 28) & 0xF ] << 4);

    X &= 0x0FFFFFFF;
    Y &= 0x0FFFFFFF;

    //
    // calculate subkeys
    //
    #pragma unroll
    for( i = 0; i < 16; i++ )
    {
        if( i < 2 || i == 8 || i == 15 )
        {
            X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFF;
            Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFF;
        }
        else
        {
            X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFF;
            Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFF;
        }

        *SK++ =
              ((X <<  4) & 0x24000000) | ((X << 28) & 0x10000000)
            | ((X << 14) & 0x08000000) | ((X << 18) & 0x02080000)
            | ((X <<  6) & 0x01000000) | ((X <<  9) & 0x00200000)
            | ((X >>  1) & 0x00100000) | ((X << 10) & 0x00040000)
            | ((X <<  2) & 0x00020000) | ((X >> 10) & 0x00010000)
            | ((Y >> 13) & 0x00002000) | ((Y >>  4) & 0x00001000)
            | ((Y <<  6) & 0x00000800) | ((Y >>  1) & 0x00000400)
            | ((Y >> 14) & 0x00000200) | ((Y      ) & 0x00000100)
            | ((Y >>  5) & 0x00000020) | ((Y >> 10) & 0x00000010)
            | ((Y >>  3) & 0x00000008) | ((Y >> 18) & 0x00000004)
            | ((Y >> 26) & 0x00000002) | ((Y >> 24) & 0x00000001);

        *SK++ =
              ((X << 15) & 0x20000000) | ((X << 17) & 0x10000000)
            | ((X << 10) & 0x08000000) | ((X << 22) & 0x04000000)
            | ((X >>  2) & 0x02000000) | ((X <<  1) & 0x01000000)
            | ((X << 16) & 0x00200000) | ((X << 11) & 0x00100000)
            | ((X <<  3) & 0x00080000) | ((X >>  6) & 0x00040000)
            | ((X << 15) & 0x00020000) | ((X >>  4) & 0x00010000)
            | ((Y >>  2) & 0x00002000) | ((Y <<  8) & 0x00001000)
            | ((Y >> 14) & 0x00000808) | ((Y >>  9) & 0x00000400)
            | ((Y      ) & 0x00000200) | ((Y <<  7) & 0x00000100)
            | ((Y >>  7) & 0x00000020) | ((Y >>  3) & 0x00000011)
            | ((Y <<  2) & 0x00000004) | ((Y >> 21) & 0x00000002);
    }
}
*/

/*
__device__ static inline uint64_t des_crypt_ecb_cuda64(const uint32_t SK[32], const uint64_t input)
{
    int i;
    uint32_t X, Y, T;
    static const uint32_t SB[8][64] = {
    {
        0x01010400, 0x00000000, 0x00010000, 0x01010404,
        0x01010004, 0x00010404, 0x00000004, 0x00010000,
        0x00000400, 0x01010400, 0x01010404, 0x00000400,
        0x01000404, 0x01010004, 0x01000000, 0x00000004,
        0x00000404, 0x01000400, 0x01000400, 0x00010400,
        0x00010400, 0x01010000, 0x01010000, 0x01000404,
        0x00010004, 0x01000004, 0x01000004, 0x00010004,
        0x00000000, 0x00000404, 0x00010404, 0x01000000,
        0x00010000, 0x01010404, 0x00000004, 0x01010000,
        0x01010400, 0x01000000, 0x01000000, 0x00000400,
        0x01010004, 0x00010000, 0x00010400, 0x01000004,
        0x00000400, 0x00000004, 0x01000404, 0x00010404,
        0x01010404, 0x00010004, 0x01010000, 0x01000404,
        0x01000004, 0x00000404, 0x00010404, 0x01010400,
        0x00000404, 0x01000400, 0x01000400, 0x00000000,
        0x00010004, 0x00010400, 0x00000000, 0x01010004
    },
    {
        0x80108020, 0x80008000, 0x00008000, 0x00108020,
        0x00100000, 0x00000020, 0x80100020, 0x80008020,
        0x80000020, 0x80108020, 0x80108000, 0x80000000,
        0x80008000, 0x00100000, 0x00000020, 0x80100020,
        0x00108000, 0x00100020, 0x80008020, 0x00000000,
        0x80000000, 0x00008000, 0x00108020, 0x80100000,
        0x00100020, 0x80000020, 0x00000000, 0x00108000,
        0x00008020, 0x80108000, 0x80100000, 0x00008020,
        0x00000000, 0x00108020, 0x80100020, 0x00100000,
        0x80008020, 0x80100000, 0x80108000, 0x00008000,
        0x80100000, 0x80008000, 0x00000020, 0x80108020,
        0x00108020, 0x00000020, 0x00008000, 0x80000000,
        0x00008020, 0x80108000, 0x00100000, 0x80000020,
        0x00100020, 0x80008020, 0x80000020, 0x00100020,
        0x00108000, 0x00000000, 0x80008000, 0x00008020,
        0x80000000, 0x80100020, 0x80108020, 0x00108000
    },
    {
        0x00000208, 0x08020200, 0x00000000, 0x08020008,
        0x08000200, 0x00000000, 0x00020208, 0x08000200,
        0x00020008, 0x08000008, 0x08000008, 0x00020000,
        0x08020208, 0x00020008, 0x08020000, 0x00000208,
        0x08000000, 0x00000008, 0x08020200, 0x00000200,
        0x00020200, 0x08020000, 0x08020008, 0x00020208,
        0x08000208, 0x00020200, 0x00020000, 0x08000208,
        0x00000008, 0x08020208, 0x00000200, 0x08000000,
        0x08020200, 0x08000000, 0x00020008, 0x00000208,
        0x00020000, 0x08020200, 0x08000200, 0x00000000,
        0x00000200, 0x00020008, 0x08020208, 0x08000200,
        0x08000008, 0x00000200, 0x00000000, 0x08020008,
        0x08000208, 0x00020000, 0x08000000, 0x08020208,
        0x00000008, 0x00020208, 0x00020200, 0x08000008,
        0x08020000, 0x08000208, 0x00000208, 0x08020000,
        0x00020208, 0x00000008, 0x08020008, 0x00020200
    },
    {
        0x00802001, 0x00002081, 0x00002081, 0x00000080,
        0x00802080, 0x00800081, 0x00800001, 0x00002001,
        0x00000000, 0x00802000, 0x00802000, 0x00802081,
        0x00000081, 0x00000000, 0x00800080, 0x00800001,
        0x00000001, 0x00002000, 0x00800000, 0x00802001,
        0x00000080, 0x00800000, 0x00002001, 0x00002080,
        0x00800081, 0x00000001, 0x00002080, 0x00800080,
        0x00002000, 0x00802080, 0x00802081, 0x00000081,
        0x00800080, 0x00800001, 0x00802000, 0x00802081,
        0x00000081, 0x00000000, 0x00000000, 0x00802000,
        0x00002080, 0x00800080, 0x00800081, 0x00000001,
        0x00802001, 0x00002081, 0x00002081, 0x00000080,
        0x00802081, 0x00000081, 0x00000001, 0x00002000,
        0x00800001, 0x00002001, 0x00802080, 0x00800081,
        0x00002001, 0x00002080, 0x00800000, 0x00802001,
        0x00000080, 0x00800000, 0x00002000, 0x00802080
    },
    {
        0x00000100, 0x02080100, 0x02080000, 0x42000100,
        0x00080000, 0x00000100, 0x40000000, 0x02080000,
        0x40080100, 0x00080000, 0x02000100, 0x40080100,
        0x42000100, 0x42080000, 0x00080100, 0x40000000,
        0x02000000, 0x40080000, 0x40080000, 0x00000000,
        0x40000100, 0x42080100, 0x42080100, 0x02000100,
        0x42080000, 0x40000100, 0x00000000, 0x42000000,
        0x02080100, 0x02000000, 0x42000000, 0x00080100,
        0x00080000, 0x42000100, 0x00000100, 0x02000000,
        0x40000000, 0x02080000, 0x42000100, 0x40080100,
        0x02000100, 0x40000000, 0x42080000, 0x02080100,
        0x40080100, 0x00000100, 0x02000000, 0x42080000,
        0x42080100, 0x00080100, 0x42000000, 0x42080100,
        0x02080000, 0x00000000, 0x40080000, 0x42000000,
        0x00080100, 0x02000100, 0x40000100, 0x00080000,
        0x00000000, 0x40080000, 0x02080100, 0x40000100
    },
    {
        0x20000010, 0x20400000, 0x00004000, 0x20404010,
        0x20400000, 0x00000010, 0x20404010, 0x00400000,
        0x20004000, 0x00404010, 0x00400000, 0x20000010,
        0x00400010, 0x20004000, 0x20000000, 0x00004010,
        0x00000000, 0x00400010, 0x20004010, 0x00004000,
        0x00404000, 0x20004010, 0x00000010, 0x20400010,
        0x20400010, 0x00000000, 0x00404010, 0x20404000,
        0x00004010, 0x00404000, 0x20404000, 0x20000000,
        0x20004000, 0x00000010, 0x20400010, 0x00404000,
        0x20404010, 0x00400000, 0x00004010, 0x20000010,
        0x00400000, 0x20004000, 0x20000000, 0x00004010,
        0x20000010, 0x20404010, 0x00404000, 0x20400000,
        0x00404010, 0x20404000, 0x00000000, 0x20400010,
        0x00000010, 0x00004000, 0x20400000, 0x00404010,
        0x00004000, 0x00400010, 0x20004010, 0x00000000,
        0x20404000, 0x20000000, 0x00400010, 0x20004010
    },
    {
        0x00200000, 0x04200002, 0x04000802, 0x00000000,
        0x00000800, 0x04000802, 0x00200802, 0x04200800,
        0x04200802, 0x00200000, 0x00000000, 0x04000002,
        0x00000002, 0x04000000, 0x04200002, 0x00000802,
        0x04000800, 0x00200802, 0x00200002, 0x04000800,
        0x04000002, 0x04200000, 0x04200800, 0x00200002,
        0x04200000, 0x00000800, 0x00000802, 0x04200802,
        0x00200800, 0x00000002, 0x04000000, 0x00200800,
        0x04000000, 0x00200800, 0x00200000, 0x04000802,
        0x04000802, 0x04200002, 0x04200002, 0x00000002,
        0x00200002, 0x04000000, 0x04000800, 0x00200000,
        0x04200800, 0x00000802, 0x00200802, 0x04200800,
        0x00000802, 0x04000002, 0x04200802, 0x04200000,
        0x00200800, 0x00000000, 0x00000002, 0x04200802,
        0x00000000, 0x00200802, 0x04200000, 0x00000800,
        0x04000002, 0x04000800, 0x00000800, 0x00200002
    },
    {
        0x10001040, 0x00001000, 0x00040000, 0x10041040,
        0x10000000, 0x10001040, 0x00000040, 0x10000000,
        0x00040040, 0x10040000, 0x10041040, 0x00041000,
        0x10041000, 0x00041040, 0x00001000, 0x00000040,
        0x10040000, 0x10000040, 0x10001000, 0x00001040,
        0x00041000, 0x00040040, 0x10040040, 0x10041000,
        0x00001040, 0x00000000, 0x00000000, 0x10040040,
        0x10000040, 0x10001000, 0x00041040, 0x00040000,
        0x00041040, 0x00040000, 0x10041000, 0x00001000,
        0x00000040, 0x10040040, 0x00001000, 0x00041040,
        0x10001000, 0x00000040, 0x10000040, 0x10040000,
        0x10040040, 0x10000000, 0x00040000, 0x10001040,
        0x00000000, 0x10041040, 0x00040040, 0x10000040,
        0x10040000, 0x10001000, 0x10001040, 0x00000000,
        0x10041040, 0x00041000, 0x00041000, 0x00001040,
        0x00001040, 0x00040040, 0x10000000, 0x10041000
    }
    };

    X = input >> 32;      // GET_UINT32_BE( X, input, 0 );
    Y = input&0xFFFFFFFF; // GET_UINT32_BE( Y, input, 4 );

    DES_IP( X, Y );

    #pragma unroll
    for( i = 0; i < 8; i++ )
    {
        DES_ROUND( Y, X );
        DES_ROUND( X, Y );
    }
    
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );
//    DES_ROUND( Y, X );
//    DES_ROUND( X, Y );

    DES_FP( Y, X );
    
    // PUT_UINT32_BE( Y, output, 0 );
    // PUT_UINT32_BE( X, output, 4 );
    return ((uint64_t)Y)<<32 | X;
}
*/

/*
__device__ static inline uint64_t des_crypt_cuda64_std(const uint64_t key, const uint64_t input)
{
  uint32_t SK[32];

  des_setkey_cuda(&SK[0], key);
  return des_crypt_ecb_cuda64(&SK[0], input);
}
*/

__device__ static inline uint64_t des_crypt_cuda64_opt( const uint64_t key, const uint64_t input )
{
  static const uint32_t LHs[16] =
  {
      0x00000000, 0x00000001, 0x00000100, 0x00000101,
      0x00010000, 0x00010001, 0x00010100, 0x00010101,
      0x01000000, 0x01000001, 0x01000100, 0x01000101,
      0x01010000, 0x01010001, 0x01010100, 0x01010101
  };
  static const uint32_t RHs[16] =
  {
      0x00000000, 0x01000000, 0x00010000, 0x01010000,
      0x00000100, 0x01000100, 0x00010100, 0x01010100,
      0x00000001, 0x01000001, 0x00010001, 0x01010001,
      0x00000101, 0x01000101, 0x00010101, 0x01010101,
  };
  static const uint32_t SB[8][64] = { {
      0x01010400, 0x00000000, 0x00010000, 0x01010404,
      0x01010004, 0x00010404, 0x00000004, 0x00010000,
      0x00000400, 0x01010400, 0x01010404, 0x00000400,
      0x01000404, 0x01010004, 0x01000000, 0x00000004,
      0x00000404, 0x01000400, 0x01000400, 0x00010400,
      0x00010400, 0x01010000, 0x01010000, 0x01000404,
      0x00010004, 0x01000004, 0x01000004, 0x00010004,
      0x00000000, 0x00000404, 0x00010404, 0x01000000,
      0x00010000, 0x01010404, 0x00000004, 0x01010000,
      0x01010400, 0x01000000, 0x01000000, 0x00000400,
      0x01010004, 0x00010000, 0x00010400, 0x01000004,
      0x00000400, 0x00000004, 0x01000404, 0x00010404,
      0x01010404, 0x00010004, 0x01010000, 0x01000404,
      0x01000004, 0x00000404, 0x00010404, 0x01010400,
      0x00000404, 0x01000400, 0x01000400, 0x00000000,
      0x00010004, 0x00010400, 0x00000000, 0x01010004
  }, {
      0x80108020, 0x80008000, 0x00008000, 0x00108020,
      0x00100000, 0x00000020, 0x80100020, 0x80008020,
      0x80000020, 0x80108020, 0x80108000, 0x80000000,
      0x80008000, 0x00100000, 0x00000020, 0x80100020,
      0x00108000, 0x00100020, 0x80008020, 0x00000000,
      0x80000000, 0x00008000, 0x00108020, 0x80100000,
      0x00100020, 0x80000020, 0x00000000, 0x00108000,
      0x00008020, 0x80108000, 0x80100000, 0x00008020,
      0x00000000, 0x00108020, 0x80100020, 0x00100000,
      0x80008020, 0x80100000, 0x80108000, 0x00008000,
      0x80100000, 0x80008000, 0x00000020, 0x80108020,
      0x00108020, 0x00000020, 0x00008000, 0x80000000,
      0x00008020, 0x80108000, 0x00100000, 0x80000020,
      0x00100020, 0x80008020, 0x80000020, 0x00100020,
      0x00108000, 0x00000000, 0x80008000, 0x00008020,
      0x80000000, 0x80100020, 0x80108020, 0x00108000
  }, {
      0x00000208, 0x08020200, 0x00000000, 0x08020008,
      0x08000200, 0x00000000, 0x00020208, 0x08000200,
      0x00020008, 0x08000008, 0x08000008, 0x00020000,
      0x08020208, 0x00020008, 0x08020000, 0x00000208,
      0x08000000, 0x00000008, 0x08020200, 0x00000200,
      0x00020200, 0x08020000, 0x08020008, 0x00020208,
      0x08000208, 0x00020200, 0x00020000, 0x08000208,
      0x00000008, 0x08020208, 0x00000200, 0x08000000,
      0x08020200, 0x08000000, 0x00020008, 0x00000208,
      0x00020000, 0x08020200, 0x08000200, 0x00000000,
      0x00000200, 0x00020008, 0x08020208, 0x08000200,
      0x08000008, 0x00000200, 0x00000000, 0x08020008,
      0x08000208, 0x00020000, 0x08000000, 0x08020208,
      0x00000008, 0x00020208, 0x00020200, 0x08000008,
      0x08020000, 0x08000208, 0x00000208, 0x08020000,
      0x00020208, 0x00000008, 0x08020008, 0x00020200
  }, {
      0x00802001, 0x00002081, 0x00002081, 0x00000080,
      0x00802080, 0x00800081, 0x00800001, 0x00002001,
      0x00000000, 0x00802000, 0x00802000, 0x00802081,
      0x00000081, 0x00000000, 0x00800080, 0x00800001,
      0x00000001, 0x00002000, 0x00800000, 0x00802001,
      0x00000080, 0x00800000, 0x00002001, 0x00002080,
      0x00800081, 0x00000001, 0x00002080, 0x00800080,
      0x00002000, 0x00802080, 0x00802081, 0x00000081,
      0x00800080, 0x00800001, 0x00802000, 0x00802081,
      0x00000081, 0x00000000, 0x00000000, 0x00802000,
      0x00002080, 0x00800080, 0x00800081, 0x00000001,
      0x00802001, 0x00002081, 0x00002081, 0x00000080,
      0x00802081, 0x00000081, 0x00000001, 0x00002000,
      0x00800001, 0x00002001, 0x00802080, 0x00800081,
      0x00002001, 0x00002080, 0x00800000, 0x00802001,
      0x00000080, 0x00800000, 0x00002000, 0x00802080
  }, {
      0x00000100, 0x02080100, 0x02080000, 0x42000100,
      0x00080000, 0x00000100, 0x40000000, 0x02080000,
      0x40080100, 0x00080000, 0x02000100, 0x40080100,
      0x42000100, 0x42080000, 0x00080100, 0x40000000,
      0x02000000, 0x40080000, 0x40080000, 0x00000000,
      0x40000100, 0x42080100, 0x42080100, 0x02000100,
      0x42080000, 0x40000100, 0x00000000, 0x42000000,
      0x02080100, 0x02000000, 0x42000000, 0x00080100,
      0x00080000, 0x42000100, 0x00000100, 0x02000000,
      0x40000000, 0x02080000, 0x42000100, 0x40080100,
      0x02000100, 0x40000000, 0x42080000, 0x02080100,
      0x40080100, 0x00000100, 0x02000000, 0x42080000,
      0x42080100, 0x00080100, 0x42000000, 0x42080100,
      0x02080000, 0x00000000, 0x40080000, 0x42000000,
      0x00080100, 0x02000100, 0x40000100, 0x00080000,
      0x00000000, 0x40080000, 0x02080100, 0x40000100
  }, {
      0x20000010, 0x20400000, 0x00004000, 0x20404010,
      0x20400000, 0x00000010, 0x20404010, 0x00400000,
      0x20004000, 0x00404010, 0x00400000, 0x20000010,
      0x00400010, 0x20004000, 0x20000000, 0x00004010,
      0x00000000, 0x00400010, 0x20004010, 0x00004000,
      0x00404000, 0x20004010, 0x00000010, 0x20400010,
      0x20400010, 0x00000000, 0x00404010, 0x20404000,
      0x00004010, 0x00404000, 0x20404000, 0x20000000,
      0x20004000, 0x00000010, 0x20400010, 0x00404000,
      0x20404010, 0x00400000, 0x00004010, 0x20000010,
      0x00400000, 0x20004000, 0x20000000, 0x00004010,
      0x20000010, 0x20404010, 0x00404000, 0x20400000,
      0x00404010, 0x20404000, 0x00000000, 0x20400010,
      0x00000010, 0x00004000, 0x20400000, 0x00404010,
      0x00004000, 0x00400010, 0x20004010, 0x00000000,
      0x20404000, 0x20000000, 0x00400010, 0x20004010
  }, {
      0x00200000, 0x04200002, 0x04000802, 0x00000000,
      0x00000800, 0x04000802, 0x00200802, 0x04200800,
      0x04200802, 0x00200000, 0x00000000, 0x04000002,
      0x00000002, 0x04000000, 0x04200002, 0x00000802,
      0x04000800, 0x00200802, 0x00200002, 0x04000800,
      0x04000002, 0x04200000, 0x04200800, 0x00200002,
      0x04200000, 0x00000800, 0x00000802, 0x04200802,
      0x00200800, 0x00000002, 0x04000000, 0x00200800,
      0x04000000, 0x00200800, 0x00200000, 0x04000802,
      0x04000802, 0x04200002, 0x04200002, 0x00000002,
      0x00200002, 0x04000000, 0x04000800, 0x00200000,
      0x04200800, 0x00000802, 0x00200802, 0x04200800,
      0x00000802, 0x04000002, 0x04200802, 0x04200000,
      0x00200800, 0x00000000, 0x00000002, 0x04200802,
      0x00000000, 0x00200802, 0x04200000, 0x00000800,
      0x04000002, 0x04000800, 0x00000800, 0x00200002
  }, {
      0x10001040, 0x00001000, 0x00040000, 0x10041040,
      0x10000000, 0x10001040, 0x00000040, 0x10000000,
      0x00040040, 0x10040000, 0x10041040, 0x00041000,
      0x10041000, 0x00041040, 0x00001000, 0x00000040,
      0x10040000, 0x10000040, 0x10001000, 0x00001040,
      0x00041000, 0x00040040, 0x10040040, 0x10041000,
      0x00001040, 0x00000000, 0x00000000, 0x10040040,
      0x10000040, 0x10001000, 0x00041040, 0x00040000,
      0x00041040, 0x00040000, 0x10041000, 0x00001000,
      0x00000040, 0x10040040, 0x00001000, 0x00041040,
      0x10001000, 0x00000040, 0x10000040, 0x10040000,
      0x10040040, 0x10000000, 0x00040000, 0x10001040,
      0x00000000, 0x10041040, 0x00040040, 0x10000040,
      0x10040000, 0x10001000, 0x10001040, 0x00000000,
      0x10041040, 0x00041000, 0x00041000, 0x00001040,
      0x00001040, 0x00040040, 0x10000000, 0x10041000
  } };

  //des_setkey_cuda( &SK[0], key );
  uint32_t XK, YK, TK;

  /*
   * PC1: left and right halves bit-swap
   */

  XK = key>>32;
  YK = key&0xFFFFFFFF;

  /*
   * Permuted Choice 1
   */
  TK =  ((YK >>  4) ^ XK) & 0x0F0F0F0F;  XK ^= TK; YK ^= (TK <<  4);
  TK =  ((YK      ) ^ XK) & 0x10101010;  XK ^= TK; YK ^= (TK      );

  XK =  (LHs[ (XK      ) & 0xF] << 3) | (LHs[ (XK >>  8) & 0xF ] << 2)
      | (LHs[ (XK >> 16) & 0xF] << 1) | (LHs[ (XK >> 24) & 0xF ]     )
      | (LHs[ (XK >>  5) & 0xF] << 7) | (LHs[ (XK >> 13) & 0xF ] << 6)
      | (LHs[ (XK >> 21) & 0xF] << 5) | (LHs[ (XK >> 29) & 0xF ] << 4);
  YK =  (RHs[ (YK >>  1) & 0xF] << 3) | (RHs[ (YK >>  9) & 0xF ] << 2)
      | (RHs[ (YK >> 17) & 0xF] << 1) | (RHs[ (YK >> 25) & 0xF ]     )
      | (RHs[ (YK >>  4) & 0xF] << 7) | (RHs[ (YK >> 12) & 0xF ] << 6)
      | (RHs[ (YK >> 20) & 0xF] << 5) | (RHs[ (YK >> 28) & 0xF ] << 4);

  XK &= 0x0FFFFFFF;
  YK &= 0x0FFFFFFF;

  uint32_t SK[32];

  #define DES_SET_SK(__X, __Y, __SK0, __SK1) \
    XK = __X; \
    YK = __Y; \
    __SK0   ((XK <<  4) & 0x24000000) | ((XK << 28) & 0x10000000)  \
          | ((XK << 14) & 0x08000000) | ((XK << 18) & 0x02080000)  \
          | ((XK <<  6) & 0x01000000) | ((XK <<  9) & 0x00200000)  \
          | ((XK >>  1) & 0x00100000) | ((XK << 10) & 0x00040000)  \
          | ((XK <<  2) & 0x00020000) | ((XK >> 10) & 0x00010000)  \
          | ((YK >> 13) & 0x00002000) | ((YK >>  4) & 0x00001000)  \
          | ((YK <<  6) & 0x00000800) | ((YK >>  1) & 0x00000400)  \
          | ((YK >> 14) & 0x00000200) | ((YK      ) & 0x00000100)  \
          | ((YK >>  5) & 0x00000020) | ((YK >> 10) & 0x00000010)  \
          | ((YK >>  3) & 0x00000008) | ((YK >> 18) & 0x00000004)  \
          | ((YK >> 26) & 0x00000002) | ((YK >> 24) & 0x00000001); \
    __SK1   ((XK << 15) & 0x20000000) | ((XK << 17) & 0x10000000)  \
          | ((XK << 10) & 0x08000000) | ((XK << 22) & 0x04000000)  \
          | ((XK >>  2) & 0x02000000) | ((XK <<  1) & 0x01000000)  \
          | ((XK << 16) & 0x00200000) | ((XK << 11) & 0x00100000)  \
          | ((XK <<  3) & 0x00080000) | ((XK >>  6) & 0x00040000)  \
          | ((XK << 15) & 0x00020000) | ((XK >>  4) & 0x00010000)  \
          | ((YK >>  2) & 0x00002000) | ((YK <<  8) & 0x00001000)  \
          | ((YK >> 14) & 0x00000808) | ((YK >>  9) & 0x00000400)  \
          | ((YK      ) & 0x00000200) | ((YK <<  7) & 0x00000100)  \
          | ((YK >>  7) & 0x00000020) | ((YK >>  3) & 0x00000011)  \
          | ((YK <<  2) & 0x00000004) | ((YK >> 21) & 0x00000002);

  DES_SET_SK(((XK <<  1) | (XK >> 27)) & 0x0FFFFFFF, ((YK <<  1) | (YK >> 27)) & 0x0FFFFFFF, SK[ 0] =, SK[ 1] =);
  DES_SET_SK(((XK <<  1) | (XK >> 27)) & 0x0FFFFFFF, ((YK <<  1) | (YK >> 27)) & 0x0FFFFFFF, SK[ 2] =, SK[ 3] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[ 4] =, SK[ 5] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[ 6] =, SK[ 7] =);

  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[ 8] =, SK[ 9] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[10] =, SK[11] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[12] =, SK[13] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[14] =, SK[15] =);

  DES_SET_SK(((XK <<  1) | (XK >> 27)) & 0x0FFFFFFF, ((YK <<  1) | (YK >> 27)) & 0x0FFFFFFF, SK[16] =, SK[17] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[18] =, SK[19] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[20] =, SK[21] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[22] =, SK[23] =);

  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[24] =, SK[25] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[26] =, SK[27] =);
  DES_SET_SK(((XK <<  2) | (XK >> 26)) & 0x0FFFFFFF, ((YK <<  2) | (YK >> 26)) & 0x0FFFFFFF, SK[28] =, SK[29] =);
  DES_SET_SK(((XK <<  1) | (XK >> 27)) & 0x0FFFFFFF, ((YK <<  1) | (YK >> 27)) & 0x0FFFFFFF, SK[30] =, SK[31] =);


  //return des_crypt_ecb_cuda64( &SK[0], input );
  //uint32_t &X=XK, &Y=YK, &T=TK;
  uint32_t X, Y, T;

  X = input >> 32;      // GET_UINT32_BE( X, input, 0 );
  Y = input&0xFFFFFFFF; // GET_UINT32_BE( Y, input, 4 );

  DES_IP( X, Y );

  #define DES_ROUND_SK(__SK0,X,__SK1,Y)       \
      T = __SK0 ^ X;                          \
      Y ^= SB[7][ (T      ) & 0x3F ] ^        \
           SB[5][ (T >>  8) & 0x3F ] ^        \
           SB[3][ (T >> 16) & 0x3F ] ^        \
           SB[1][ (T >> 24) & 0x3F ];         \
      T = __SK1 ^ ((X << 28) | (X >> 4));     \
      Y ^= SB[6][ (T      ) & 0x3F ] ^        \
           SB[4][ (T >>  8) & 0x3F ] ^        \
           SB[2][ (T >> 16) & 0x3F ] ^        \
           SB[0][ (T >> 24) & 0x3F ];

  DES_ROUND_SK( SK[ 0], Y, SK[ 1], X );
  DES_ROUND_SK( SK[ 2], X, SK[ 3], Y );
  DES_ROUND_SK( SK[ 4], Y, SK[ 5], X );
  DES_ROUND_SK( SK[ 6], X, SK[ 7], Y );

  DES_ROUND_SK( SK[ 8], Y, SK[ 9], X );
  DES_ROUND_SK( SK[10], X, SK[11], Y );
  DES_ROUND_SK( SK[12], Y, SK[13], X );
  DES_ROUND_SK( SK[14], X, SK[15], Y );

  DES_ROUND_SK( SK[16], Y, SK[17], X );
  DES_ROUND_SK( SK[18], X, SK[19], Y );
  DES_ROUND_SK( SK[20], Y, SK[21], X );
  DES_ROUND_SK( SK[22], X, SK[23], Y );

  DES_ROUND_SK( SK[24], Y, SK[25], X );
  DES_ROUND_SK( SK[26], X, SK[27], Y );
  DES_ROUND_SK( SK[28], Y, SK[29], X );
  DES_ROUND_SK( SK[30], X, SK[31], Y );

  DES_FP( Y, X );

  #undef DES_ROUND_SK
  #undef DES_SET_SK
  
  return ((uint64_t)Y)<<32 | X;
}

/*
__device__ static inline int des_setkey_enc_cuda( des_context *ctx, const unsigned char key[DES_KEY_SIZE] )
{
    des_setkey_cuda( ctx->sk, uint8to64(key) );
    return( 0 );
}
 
__device__ static inline int des_crypt_ecb_cuda( des_context *ctx,
        const unsigned char input[8],
        unsigned char output[8] )
{
  uint64_t output64;
  
  output64 = des_crypt_ecb_cuda64( &ctx->sk[0], uint8to64(input));
  uint64to8(output, output64);
  return 0;
} 

__device__ static inline void newKey_cuda(unsigned char* key, int inc)
{
  *(uint64_t *)key = des56to64(des64to56(*(uint64_t *)(&key[0])) + inc);
}

__device__ static inline int equals_cuda(const unsigned char* a, const unsigned char* b)
{
    return (*(uint64_t*)a == *(uint64_t*)b);
}

__device__ static inline void displayData_cuda(const unsigned char* data, int size)
{
    for (int i = 0; i<size; ++i){
        printf("%c %02x\t",data[i]/*>=0x20&&data[i]<0x80?data[i]:'.',data[i]);
    }
    printf("\n");
}
*/

//__global__ void DESkernel(volatile int* keyfound, unsigned char* key, const unsigned char* plain, const unsigned char* cipher, int size, volatile uint32_t *d_state_ptr)
__global__ void DESkernel(volatile int* keyfound, uint64_t* key, const uint64_t plain, const uint64_t cipher, const uint64_t stepsize, volatile uint32_t *d_state_ptr)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int inc = blockDim.x * gridDim.x; //#threads * #blocks

    volatile uint64_t &state_thread_shared_idx  = *(uint64_t *)(&d_state_ptr[0]);
    volatile uint64_t &state_thread_shared_key  = *(uint64_t *)(&d_state_ptr[2]);
    volatile uint64_t &state_thread_shared_cnt  = *(uint64_t *)(&d_state_ptr[4]);
    volatile uint32_t &state_thread_start_count =                d_state_ptr[6];
    volatile uint32_t &state_thread_stop_count  =                d_state_ptr[7];

    atomicAdd((uint32_t *)&state_thread_start_count, 1);
/*
    volatile uint32_t * const &state_thread_running_map =               &d_state_ptr[4];
    const size_t    state_thread_running_map_bitsize = sizeof(state_thread_running_map[0])*8;
    const uint32_t  state_thread_running_map_bitmask = 1UL<<(tid%state_thread_running_map_bitsize);
    const size_t    state_thread_running_map_index   = tid/state_thread_running_map_bitsize;
    atomicOr ((uint32_t *)&state_thread_running_map[state_thread_running_map_index], state_thread_running_map_bitmask);
*/    

//    *keyfound = 0;
    
/*    printf("plain kernel\n");
    displayData_cuda(plain, size);
    printf("key kernel\n");
    displayData_cuda(key, size);
    printf("cipher kernel\n");
    displayData_cuda(cipher, size);
*/    

/*
     des_context my_ctx;
     unsigned char buf[8];
     unsigned char my_key[8];
     memcpy(my_key,key,size);
*/

    //int inc_bits = 0;
    //for (int i = inc; (i >>= 1) != 0; ++inc_bits);
    const int inc_bits = 32-__clz(inc-1);
    const uint64_t count = (1ULL<<56)/inc;
    //int count_bits = 0;
    //for (uint64_t i = count; (i >>= 1) != 0; ++count_bits);
    const int count_bits = 64-__clzll(count-1);
    //int idx_bits = 0;
    //const int count_bits = 56-inc_bits;
    //const uint64_t count = 1ULL<<count_bits;

    //initalize offset for threads
    //const int printTid = (tid & ~(3|(inc/2))) == 0 || (tid & ~(3|(inc/2))) == ((inc-1) & ~(3|(inc/2)));
    const int printTid = (tid | 1) == 1 || (tid | 1) == (inc-1);
    //if (printTid) printf("tid=%0*X/%d, inc=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (unsigned long long) inc, (int) inc_bits, (unsigned long long)uint8to64(my_key));
    //newKey_cuda(my_key, tid);
    //if (printTid) printf("tid=%0*X/%d, cnt=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (unsigned long long) count, (int) count_bits, (unsigned long long)uint8to64(my_key));
     
    uint64_t idx = 0ULL;
    int my_found = 0;
//    const uint64_t plain64 = uint8to64(plain);
//    const uint64_t cipher64 = uint8to64(cipher);
//    const uint64_t key64 = uint8to64(key);
    const uint64_t &plain64 = plain;
    const uint64_t &cipher64 = cipher;
          uint64_t &key64 = *key;

    //uint64_t key56 = des64to56(key64);
    //key56 = des56to64(key56);
    //key56 = uint8to64((uint8_t *)&key56);
    //key56 = des64to56(key56);
    //key56 += tid;
    //key56 = des56to64(key56);
    //key56 = uint8to64((uint8_t *)&key56);
    //key56 = des64to56(key56);
    //if ((tid & ~(3|(inc/2))) == 0 || (tid & ~(3|(inc/2))) == ((inc-1) & ~(3|(inc/2)))) printf("tid=%0*X/%d, cnt=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (unsigned long long) count, (int) count_bits, des56to64(key56));
    
    //if (printTid) printf("tid=%0*X/%d, inc=%0*llX/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (inc_bits+4)/4, (unsigned long long) inc, (int) inc_bits, key64);
    uint64_t key56 = des64to56(key64) + count*tid;
    const uint64_t key56max = key56 + count;
    if (printTid) printf("tid=%0*X/%d, inc=%0*llX/%02d, cnt=%0*llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (inc_bits+4)/4, (unsigned long long) inc, inc_bits, (count_bits+4)/4, (unsigned long long) count, (int) count_bits, des56to64(key56));

#if 0
    do {
        state_thread_shared_idx = idx;

//        idx_bits = 0; for (uint64_t i = idx; (i >>= 1) != 0; ++idx_bits);
//        if (printTid && (idx < 4*inc || idx >= ((1ULL<<56) -4*inc))) printf("tid=%0*X/%d, idx=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (unsigned long long) idx, (int)idx_bits, des56to64(key56));

/*
        uint64to8(my_key, des56to64(key56));

        des_setkey_enc_cuda ( &my_ctx, my_key);
        des_crypt_ecb_cuda( &my_ctx, plain, buf );
      // printf("tid:%d my cipher:%c %02x   %c %02x   %c %02x   %c %02x  \n",tid,buf[0],buf[0],buf[1],buf[1],buf[2],buf[2],buf[3],buf[3]);

        if (equals_cuda(buf, cipher)) {
            *keyfound = my_found = 1;
            break;
        }

        //newKey_cuda(my_key, inc);
        key56 = des56to64(key56);
        key56 = uint8to64((uint8_t *)&key56);
        key56 = des64to56(key56);
        key56 += inc;
        key56 = des56to64(key56);
        key56 = uint8to64((uint8_t *)&key56);
        key56 = des64to56(key56);
        if (printTid) printf("tid=%0*X/%d, idx2=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (unsigned long long) idx, (int)idx_bits, des56to64(key56));
*/

        //des_setkey_cuda(&my_ctx.sk[0], des56to64(key56));
        //uint64_t enc64 = des_crypt_ecb_cuda64( &my_ctx, plain64 );
        //uint64to8(&buf[0], enc64);
        //des_crypt_ecb_cuda( &my_ctx, plain, buf );
        //const uint64_t enc64 = des_crypt_ecb_cuda64( &my_ctx.sk[0], plain64 );
        //uint64to8(&buf[0], enc64);

        if (des_crypt_cuda64(des56to64(key56), plain64) == cipher64) {
          *keyfound = my_found = 1;
          break;
        }

        //key56 = des64to56(uint8to64(my_key));
        key56 += 1;

    } while (!(*keyfound) && (idx += inc) < (1ULL<<56));
#endif /* 0 */

    const uint64_t count_mask = atomicOr((unsigned long long *)&state_thread_shared_key, des56to64(count*tid)) & ~(des56to64(count)-1);
    //const uint64_t count_mask = state_thread_shared_key & ~(des56to64(count)-1);
    //if (printTid && (idx < 4 || idx >= count-4)) printf("tid=%0*X/%d, inc=%0*llX/%02d, idx=%0*llX/%02d, count_mask=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (inc_bits+4)/4, (unsigned long long) inc, inc_bits, (count_bits+4)/4, (unsigned long long) idx, count_bits, count_mask);
    do {
        const uint64_t key64 = des56to64(key56);
//        if ((idx & ((1ULL<<16)-1)) == 0) {
//            atomicMax((unsigned long long *)&state_thread_shared_idx, idx);
//            atomicMax((unsigned long long *)&state_thread_shared_key, key64 | count_mask);
//        }
        state_thread_shared_idx = idx;
        atomicMax((unsigned long long *)&state_thread_shared_key, key64 | count_mask);
        
//        if (printTid && (idx < 4 || idx >= count-4)) printf("tid=%0*X/%d, inc=%0*llX/%02d, idx=%0*llX/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (inc_bits+4)/4, (unsigned long long) inc, inc_bits, (count_bits+4)/4, (unsigned long long) idx, count_bits, des56to64(key56));
        const uint64_t enc64 = des_crypt_cuda64_opt(key64, plain64);
        atomicAdd((unsigned long long *)&state_thread_shared_cnt, 1);
        if (my_found = (enc64 == cipher64)) *keyfound = 1;
    } while (!*keyfound && ((idx += stepsize), (key56 += stepsize) < key56max /* && idx < count */));
//    uint64to8(&my_key[0], des56to64(key56));

    if (printTid || my_found) {
      if (my_found) {
        *keyfound = 1;
        key64 = des56to64(key56);
//        printf("!!! KEY FOUND (tid %d, loops %llu=$%llX) !!!\n",tid, idx/inc, idx/inc);
        printf("\n!!! KEY FOUND (tid %d, loops %llu=$%llX) !!!\n",tid, idx, idx);
//        printf("tid:%d key:%c %02X   %c %02X   %c %02X   %c %02X   %c %02X   %c %02X   %c %02X   %c %02X   \n",tid,my_key[0],my_key[0],my_key[1],my_key[1],my_key[2],my_key[2],my_key[3],my_key[3],my_key[4],my_key[4],my_key[5],my_key[5],my_key[6],my_key[6],my_key[7],my_key[7]);
//        memcpy(key, my_key, size);
//        uint64to8(key, des56to64(key56));
      }
//      const uint64_t idx = ((key56 - des64to56(key64) - ((1ULL<<56)/inc)*tid)) * inc;
//      idx_bits = 0; for (uint64_t i = idx; (i >>= 1) != 0; ++idx_bits);
//      printf("tid=%0*X/%d, idx=%016llx/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, idx, idx_bits, des56to64(key56) /*uint8to64(my_found ? key : my_key) */);
      const int idx_bits = 64-__clzll(idx);
      printf("tid=%0*X/%d, idx=%0*llX/%02d, my_key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (count_bits+3)/4, idx, idx_bits, des56to64(key56) /*uint8to64(my_found ? key : my_key) */);
    }
/*
    atomicAnd((uint32_t *)&state_thread_running_map[state_thread_running_map_index], ~state_thread_running_map_bitmask);
*/
    atomicAdd((uint32_t *)&state_thread_stop_count, 1);
}

__global__ void DESkernelBitSplice(volatile int* keyfound, uint64_t* key, const uint64_t plain, const uint64_t cipher, const int64_t step_size, volatile uint32_t *d_state_ptr)
{
  //#define printf(...)
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int inc = blockDim.x * gridDim.x; //#threads * #blocks

  volatile uint64_t &state_thread_shared_idx  = *(uint64_t *)(&d_state_ptr[0]);
  volatile uint64_t &state_thread_shared_key  = *(uint64_t *)(&d_state_ptr[2]);
  volatile uint64_t &state_thread_shared_cnt  = *(uint64_t *)(&d_state_ptr[4]);
  volatile uint32_t &state_thread_start_count =                d_state_ptr[6];
  volatile uint32_t &state_thread_stop_count  =                d_state_ptr[7];

  atomicAdd((uint32_t *)&state_thread_start_count, 1);

  __shared__ sboxes_bitslice  plainslice  [64];
  __shared__ sboxes_bitslice  cipherslice [64];

  const int       key_bits    = 56;
  const uint64_t  key_size    = 1ULL<<key_bits;
  const uint64_t  key_offs    = 0;
  const uint64_t  key_mask    = key_size-1;

  const int       slice_size  = bitsizeof(sboxes_bitslice);
  const int       slice_bits  = 32-__clz(slice_size-1);
  const uint64_t  slice_offs  = key_size / slice_size;
  const uint64_t  slice_mask  = slice_offs * (slice_size-1);

  const uint32_t  inc_size    = inc;
  const int       inc_bits    = 32-__clz(inc-1);
  const uint64_t  inc_offs    = slice_offs / inc_size;
  const uint64_t  inc_mask    = inc_offs * (inc_size-1);

  const uint64_t  count_size  = inc_offs;
  const int       count_bits  = 64-__clzll(count_size-1);
  const uint64_t  count_offs  = inc_offs / count_size;
  const uint64_t  count_mask  = count_offs * (count_size-1); //= ((1ULL<<56)/slice_size/inc) * ~(inc-1); // = (~(slice_size-1)) << (slice_bits+inc_bits);
  
  const int       printTid    = ((tid & ~(inc/2|inc/4))| 1|2) == (1|2) || ((tid | (inc/2/*|inc/4*/)) & ~(1/*|2*/)) == ((inc-1) & ~(1/*|2*/));
  const uint64_t  up_mask64   = des56to64(slice_mask|inc_mask);

  sboxes_bitslice   keyslice[56];
  sboxes_bitslice   resultslice;
  sboxes_deskey     key64 = *key;
  sboxes_deskey     key56 = des64to56(key64);
  uint64_t          idx = 0ULL;
  int64_t           step = step_size >= 0 ? step_size : -step_size;
  
  #define printstate(tid, __idx, __key64)                                           \
    printf("tid=$%0*X/%d "                                                            \
           "slice=$%0*llX/%02d inc=$%0*llX/%02d idx=$%0*llX/%02d key"                 \
           "=%016llX"                                                                 \
           "=%016llo:%016llo"                                                         \
           "=%0*llo/%d:%0*llo/%d:%0*llo/%d"                                           \
           "\n",                                                                      \
                 (inc_bits  +3)/4, (unsigned )tid, (int)inc_bits,                   \
                 (slice_bits+4)/4, (unsigned long long)slice_size, (int)slice_bits,   \
                 (inc_bits  +4)/4, (unsigned long long)inc_size  , (int)inc_bits  ,   \
                 (count_bits+4)/4, (unsigned long long)__idx     , (int)count_bits,   \
             (unsigned long long)__key64                                              \
             , (unsigned long long)bin16oct64(__key64>>48)                            \
             , (unsigned long long)bin16oct64(__key64>>32)                            \
                , (slice_bits*8                                  )/7,   (unsigned long long) bin16oct64((__key64&des56to64(slice_mask))>>(64-((slice_bits                    )*8/7)   )), (int)(slice_bits*8                                 )/7   \
                , (inc_bits  *8+(slice_bits*8)                 %7)/7,   (unsigned long long) bin16oct64((__key64&des56to64(inc_mask  ))>>(64-((inc_bits+slice_bits           )*8/7)   )), (int)(inc_bits*8  +(slice_bits*8)%7                )/7   \
                , (count_bits*8+(inc_bits  *8+(slice_bits*8)%7)%7)/7-32,(unsigned long long) bin16oct64((__key64&des56to64(count_mask))>>(64-((slice_bits+inc_bits+count_bits)*8/7)+32)), (int)(count_bits*8+(inc_bits*8+(slice_bits*8)%7)%7)/7-32 \
    )

  sboxes_set_data(plainslice, plain);
  sboxes_set_data(cipherslice, cipher);
  sboxes_set_key64(keyslice, *key);
  
  key64 = sboxes_set_low_keys(keyslice);                  //key56 = des64to56(key64);   if (printTid) printstate(tid, idx, key64);
  //key64 = sboxes_get_key64   (keyslice, sboxes_sliceon);  key56 = des64to56(key64);   if (printTid) printstate(tid, idx, key64);
  key64 = sboxes_add_keys(keyslice, count_size*tid);  //key56 = des64to56(key64);   if (printTid) printstate(tid, idx, key64);
  //key64 = sboxes_get_key64   (keyslice, sboxes_sliceon);  key56 = des64to56(key64);   if (printTid) printstate(tid, idx, key64);
  key56 = des64to56(key64);

  if (tid == 0) {
      printstate(inc_size, count_size, up_mask64);
      printf("[BitSplice:%dbit/%d]\n"
             "key_size=%016llX, key_bits=%02d, key_offs56=%016llX, key_mask56=%016llX, key_offs64=%016llX, key_mask64=%016llX\n"
             "slc_size=%016llX, slc_bits=%02d, slc_offs56=%016llX, slc_mask56=%016llX, slc_offs64=%016llX, slc_mask64=%016llX\n"
             "inc_size=%016llX, inc_bits=%02d, inc_offs56=%016llX, inc_mask56=%016llX, inc_offs64=%016llX, inc_mask64=%016llX\n"
             "cnt_size=%016llX, cnt_bits=%02d, cnt_offs56=%016llX, cnt_mask56=%016llX, cnt_offs64=%016llX, cnt_mask64=%016llX\n", 
             (int)sboxes_slicebitsize, (int)sboxes_slicebitcount,
             (unsigned long long)key_size  , key_bits  , (unsigned long long)key_offs  , (unsigned long long)key_mask,    (unsigned long long)des56to64(key_offs  ), (unsigned long long)des56to64(key_mask  ), 
             (unsigned long long)slice_size, slice_bits, (unsigned long long)slice_offs, (unsigned long long)slice_mask,  (unsigned long long)des56to64(slice_offs), (unsigned long long)des56to64(slice_mask), 
             (unsigned long long)inc_size  , inc_bits  , (unsigned long long)inc_offs  , (unsigned long long)inc_mask,    (unsigned long long)des56to64(inc_offs  ), (unsigned long long)des56to64(inc_mask  ), 
             (unsigned long long)count_size, count_bits, (unsigned long long)count_offs, (unsigned long long)count_mask,  (unsigned long long)des56to64(count_offs), (unsigned long long)des56to64(count_mask) );
      /*printf("up_mask64=%016llX\n", up_mask64);*/
      /*printf("tid=%0*X/%d, slc=%014llX|%0*X/%02d, cnt=%014llX|%0*llX/%02d\n", 
          (inc_bits+3)/4, tid, inc_bits,
          slice_mask, (slice_bits+4)/4, slice_size, slice_bits,
          count_mask, (count_bits+3)/4, count_size, count_bits);*/
  }

  atomicAnd((unsigned long long *)&state_thread_shared_key, ~(count_mask|slice_mask));
  atomicOr ((unsigned long long *)&state_thread_shared_key, tid*count_size);

  //if (printTid) printstate(tid, idx, key64);

  __syncthreads();
  state_thread_shared_idx = 0;

  #pragma unroll 1024
  do {
      state_thread_shared_idx = idx;
      if ((idx & ((1UL<<16)-1)) == 0) {
        atomicMax((unsigned long long *)&state_thread_shared_key, (state_thread_shared_key & (~count_mask)) | (key56));
        key64 = des56to64(key56);
        if (printTid && (idx < 4 || idx >= count_size-4)) printstate(tid, idx, key64);
      }

      if (step_size > 0) resultslice = sboxes_deseval(plainslice, cipherslice, keyslice);
      else if (step_size == 0) break;
      else {
        sboxes_bitslice t[64];
        sboxes_desslice_set(t, cipherslice);
        sboxes_desslice_dec(t, keyslice);
        resultslice = sboxes_desslice_eval(t, plainslice);
      }
      atomicAdd((unsigned long long *)&state_thread_shared_cnt, slice_size);

      /*
        if (resultslice != 0) break;
        if (step_size == 1)   key64 = sboxes_inc_keys(keyslice);
        else                  key64 = sboxes_add_keys(keyslice, step_size);
        key56 += step_size; // key56 = des56to64(key64);
        if (printTid && (idx < 4 || idx >= count_size-4)) printf("tid=%0*X/%d, slc=%02llX/%02d, inc=%0*llX/%02d, idx=%0*llX/%02d, key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, (slice_bits+4)/4, (unsigned long long)slice, slice_bits, (inc_bits+4)/4, (unsigned long long) inc, inc_bits, (count_bits+4)/4, (unsigned long long) idx, count_bits, key64);
        */
  } //while (!*keyfound && (idx += step_size) < count_size);
  while (resultslice == 0 && !*keyfound && (sboxes_add_keys_fast(keyslice, 1*step), key56 += step, (idx += step) < count_size));
  
  state_thread_shared_idx = 0;
  if (resultslice != 0 || step_size == 0) {
    *keyfound = 1;
    *key = sboxes_key_found(keyslice, resultslice);
    printstate(tid, idx, des56to64(key56));
    const int idx_bits = 64-__clzll(idx);
    printf("tid=%0*X/%d, idx=%0*llX/%02d, key=%016llX\n", (inc_bits+3)/4, tid, inc_bits, 
                         (unsigned long long)(count_bits+3)/4, idx, idx_bits, (unsigned long long)key64);
    printf("\n!!! KEY FOUND (tid %d, loops %llu=$%llX @ %016lX) WITH %s!!!\n", tid, idx, idx, key64, step_size > 0 ? "ENCRYPTION" : "DECRYPTION");
  }

  atomicAdd((uint32_t *)&state_thread_stop_count, 1);
  return;
  #ifdef printf
  #undef printf
  #endif /* printf */
}

#define printf(fmt, ...) \
  do { \
    if (!fpLogFile) fpLogFile = fopen("cudaDES.log", "at+"); \
    if (fpLogFile) { \
      fprintf(fpLogFile, (fmt), ## __VA_ARGS__); \
      if (strchr(fmt, '\r') != NULL || strchr(fmt, '\n') != NULL) fclose(fpLogFile), fpLogFile = NULL; \
    } \
    fprintf(stdout, (fmt), ## __VA_ARGS__); \
  } while (0)

//void cudaFunction(unsigned char* key, const unsigned char* plain, const unsigned char* cipher, int size)
uint64_t cudaFunction(uint64_t key, const uint64_t plain, const uint64_t cipher, const int64_t step_size = 1)
{   
//    unsigned char* startkey = (unsigned char*)malloc(sizeof(unsigned char)*size);
//    memcpy(startkey,key,size);

    float elapsedTime;
//    size_t real_size;

//    real_size = size * sizeof(unsigned char);
//    unsigned char* d_key;
//    unsigned char* d_plain;
//    unsigned char* d_cipher;
    uint64_t * d_key;
    //uint64_t * d_plain;
    //uint64_t * d_cipher;
    int* d_keyfound;
    int  keyfound = 0;

    if (deviceId >= 0) {
      cudaSetDevice(deviceId);
      checkCUDAError("cudaSetDevice");
    }

    //malloc device memory
//    cudaMalloc(&d_key, real_size);
    cudaMalloc(&d_key, sizeof(*d_key));
    checkCUDAError("cudaMalloc d_key");
//    cudaMalloc(&d_plain, real_size);
//    cudaMalloc(&d_plain, sizeof(*d_plain));
//    checkCUDAError("cudaMalloc d_plain");
//    cudaMalloc(&d_cipher, real_size);
//    cudaMalloc(&d_cipher, sizeof(*d_cipher));
//    checkCUDAError("cudaMalloc d_cipher");
    cudaMalloc(&d_keyfound, sizeof(*d_keyfound));
    checkCUDAError("cudaMalloc d_keyfound");

    //copy to device
//    cudaMemcpy(d_key, key, real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, &key, sizeof(key), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy to device key");
//    cudaMemcpy(d_plain, plain, real_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_plain, &plain, sizeof(plain), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy to device plain");
//    cudaMemcpy(d_cipher, cipher, real_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_cipher, &cipher, sizeof(cipher), cudaMemcpyHostToDevice);
//    checkCUDAError("cudaMemcpy to device cipher");
    cudaMemcpy(d_keyfound, &keyfound, sizeof(*d_keyfound), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy to device keyfound");
 
     //invoke kernel
    int numberBlocks = 24;
    int numberThreads = 64;
    //numberBlocks  = 32768;    // <=> 2^15
    //numberBlocks  = 16384;    // <=> 2^14
    //numberBlocks  =  4096;    // <=> 2^12
    //numberBlocks  = 1024;     // <=> 2^10
    //numberBlocks = 512;
    //numberBlocks = 64;
    //numberBlocks = 32;
    //numberThreads = 512;
    //numberThreads = 512;      // <=> 2^9
    //numberThreads = 256;      // <=> 2^9
    //numberThreads = 64;       // <=> 2^6
    //numberThreads = 32;

    if (true) {
        // The following code sample configures an occupancy-based kernel launch of DESkernel according to the user input.
        int blockSize;      // The launch configurator returned block size
        int minGridSize;    // The minimum grid size needed to achieve the
                            // maximum occupancy for a full device
                            // launch
        int gridSize;       // The actual grid size needed, based on input
                            // size

        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            (void*)DESkernelBitSplice,
            0/*dynamicSMemSize*/,
            0/*blockSizeLimit*/);

        // Round up according to array size
        gridSize = (/*arrayCount*/ (minGridSize*blockSize) + blockSize - 1) / blockSize;
        
        //DESkernel<<<gridSize, blockSize>>>(array, arrayCount);
        //cudaDeviceSynchronize();

        // If interested, the occupancy can be calculated with
        // cudaOccupancyMaxActiveBlocksPerMultiprocessor

        ////////////////////////////////////////
        
        // The following code sample calculates the occupancy of DESkernel. 
        // It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.
        int numBlocks;        // Occupancy in terms of active blocks

        // These variables are used to convert occupancy to warps
        int device;
        cudaDeviceProp prop;
        int activeWarps;
        int maxWarps;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            DESkernelBitSplice,
            blockSize,
            0);

        activeWarps = numBlocks * blockSize / prop.warpSize;
        maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

        ////////////////////////////////////////
        
        printf("Occupancy: %.3lf (gridSize=%d,blockSize=%d,numBlocks=%d)\nwarpSize=%d,regsPerBlock=%d,threadsSM=%d,regsPerSM=%d,SMCount=%d\n", (double)activeWarps / maxWarps * 100, gridSize, blockSize, numBlocks, prop.warpSize, prop.regsPerBlock, prop.maxThreadsPerMultiProcessor, prop.regsPerMultiprocessor, prop.multiProcessorCount);
        
        numberBlocks  = minGridSize;
        numberThreads = blockSize*numBlocks;
        //numberThreads = prop.warpSize*numBlocks*2;
    }    

    int numberBlocks_bits  = 0;
    int numberThreads_bits = 0;
    for (int i = numberBlocks; (i & 1) == 0 && (i >>= 1) != 0; ++numberBlocks_bits);
    for (int i = numberThreads; (i & 1) == 0 && (i >>= 1) != 0; ++numberThreads_bits);

    printf("Blocks:%d(%d*2^%d) * Threads:%d(%d*2^%d) = $%08X(%d*2^%d) Loops:$%llX\n", 
      numberBlocks, (int)(numberBlocks/(1UL<<numberBlocks_bits)), numberBlocks_bits, 
      numberThreads, (int)(numberThreads/(1UL<<numberThreads_bits)), numberThreads_bits, 
      numberBlocks*numberThreads, (int)(numberBlocks*numberThreads/(1UL<<(numberBlocks_bits+numberThreads_bits))), numberBlocks_bits+numberThreads_bits,
      (1ULL<<56)/(numberBlocks*numberThreads));
    
    while ((numberBlocks/(1UL<<numberBlocks_bits)) > 1) {
      if (((numberBlocks/(1UL<<numberBlocks_bits))) & 1) numberBlocks += (1UL<<numberBlocks_bits);
      ++numberBlocks_bits;
    }
    while ((numberThreads/(1UL<<numberThreads_bits)) > 1) {
      if (((numberThreads/(1UL<<numberThreads_bits))) & 1) numberThreads -= (1UL<<numberThreads_bits);
      ++numberThreads_bits;
    }
    
    //numberThreads >>= 1;
    //numberBlocks <<= 3;
    
    printf("Blocks:%d(%d*2^%d) * Threads:%d(%d*2^%d) = $%08X(%d*2^%d) Loops:$%llX\n", 
      numberBlocks, (int)(numberBlocks/(1UL<<numberBlocks_bits)), numberBlocks_bits, 
      numberThreads, (int)(numberThreads/(1UL<<numberThreads_bits)), numberThreads_bits, 
      numberBlocks*numberThreads, (int)(numberBlocks*numberThreads/(1UL<<(numberBlocks_bits+numberThreads_bits))), numberBlocks_bits+numberThreads_bits,
      (1ULL<<56)/(numberBlocks*numberThreads));

    cudaFree(0);
    checkCUDAError("cudaFree(0)");

    // --- state ---
    uint32_t* state_ptr;
    const size_t state_count = 8; // + ((1L<<(numberBlocks_bits+numberThreads_bits)) + (sizeof(*state_ptr)*8) - 1)/(sizeof(*state_ptr)*8);
    const size_t state_size = sizeof(*state_ptr)*state_count;
//    printf("state_count = %u, state_size = %u\n", (unsigned)state_count, (unsigned)state_size);
    cudaHostAlloc(&state_ptr, state_size, cudaHostAllocPortable);
    checkCUDAError("cudaHostAlloc state_ptr");
    memset(state_ptr, 0, state_size);

    uint32_t* d_state_ptr;
    cudaMalloc(&d_state_ptr, state_size);
    checkCUDAError("cudaMalloc d_state_ptr");
    cudaMemcpy(d_state_ptr, state_ptr, state_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpyDefault state_ptr to device");

    // --- state_stream ---
    cudaStream_t state_stream;
    cudaStreamCreateWithFlags(&state_stream, cudaStreamNonBlocking);
    checkCUDAError("cudaStreamCreateWithFlags state_stream, cudaStreamNonBlocking");
    //cudaMemcpyAsync(d_state_ptr, state_ptr, state_size, cudaMemcpyHostToDevice, state_stream);
    //checkCUDAError("cudaMemcpyAsync state_array to device");
    //cudaMemcpyAsync(state_ptr, d_state_ptr, state_size, cudaMemcpyDeviceToHost, state_stream);
    //checkCUDAError("cudaMemcpyAsync from device state_array");
    cudaStreamSynchronize(state_stream);
    checkCUDAError("cudaStreamSynchronize state_stream");    

    // --- kernel_stream ---
    cudaStream_t kernel_stream;
    cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking);
    checkCUDAError("cudaStreamCreateWithFlags kernel_stream, cudaStreamNonBlocking");
    cudaStreamSynchronize(kernel_stream);
    checkCUDAError("cudaStreamSynchronize kernel_stream");    


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    checkCUDAError("cudaEventCreate start");
    cudaEventCreate(&stop);
    checkCUDAError("cudaEventCreate stop");

    putenv((char *)"CUDA_LAUNCH_BLOCKING=0");

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    checkCUDAError("cudaDeviceSetCacheConfig cudaFuncCachePreferL1");

    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 80);
    //checkCUDAError("cudaDeviceSetLimit cudaLimitPrintfFifoSize");

    cudaEventRecord(start, kernel_stream);
    checkCUDAError("cudaEventRecord start");

    //nt sharedSize = 3*real_size+sizeof(des_context);
    //DESkernel << <numberBlocks, numberThreads>> >(d_keyfound, d_key, plain, cipher, d_state_ptr);
    DESkernelBitSplice << <numberBlocks, numberThreads, 0, kernel_stream>> >(d_keyfound, d_key, plain, cipher, step_size, d_state_ptr);
    checkCUDAError("cudakernel call");

    printf("\nWaiting for kernel completion...\n"); fflush(stdout);
    cudaError e;

    cudaEventRecord(stop, kernel_stream);
    checkCUDAError("cudaEventRecord stop");

#ifdef WIN32
#define sleep(s) _sleep(1000*(s))
#endif /* WIN32 */

#define state_print(__time, __state_ptr, __state_count, __width, __error) do {                                          \
  const int seconds = (unsigned long)(__time)%60;                                                                       \
  const int minutes = (unsigned long)((__time)/60)%60;                                                                  \
  const int hours   = (unsigned long)((__time)/3600);                                                                   \
  printf("\r[%06lus]", (unsigned long)(__time));                                                                        \
  printf("\r[%02d:%02d:%02d]", hours, minutes, seconds);                                                                \
  printf(" Strt:%04lX",  (unsigned long)(__state_ptr)[6+0]);                                                            \
  printf(" Stop:%04lX",  (unsigned long)(__state_ptr)[6+1]);                                                            \
  const int bit_count = 1;                                                                                              \
  const int join_count = (((__state_count)-6) + ((__width)*bit_count-1)) / ((__width)*bit_count);                       \
  uint64_t slc = (1ULL<<56);                                                                                            \
  slc /= bitsizeof(sboxes_bitslice);                                                                                    \
  int slc_bits = 0; for (uint64_t i = slc-1; i != 0; ++slc_bits, i >>= 1);                                              \
  const uint64_t max = slc/(numberBlocks*numberThreads);                                                                \
  int max_bits = 0; for (uint64_t i = max-1; i != 0; ++max_bits, i >>= 1);                                              \
  int max64_bits = 0; for (uint64_t i = des56to64(max)-1; i != 0; ++max64_bits, i >>= 1);                               \
  const uint64_t idx = *((uint64_t *)&(__state_ptr)[0]);                                                                \
  int idx_bits = 0; for (uint64_t i = idx-1; i != 0; ++idx_bits, i >>= 1);                                              \
  const uint64_t cnt = *((uint64_t *)&(__state_ptr)[4]);                                                                \
  int cnt_bits = 0; for (uint64_t i = cnt-1; i != 0; ++cnt_bits, i >>= 1);                                              \
  const uint64_t key = des56to64(*((uint64_t *)&(__state_ptr)[2]));                                                     \
  int key_bits = 0; for (uint64_t i = key; i != 0; ++key_bits, i >>= 1);                                                \
  printf(" Idx:%0*llX/%02d", (max_bits+3)/4, (unsigned long long)idx, idx_bits);                                        \
  printf(" Key:%0*lX:%0*llX/%02d+%02d", (64-max64_bits+3)/4, (unsigned long)(key >> ((max64_bits/4)*4)),                \
      (max64_bits+3)/4, (unsigned long long)(key & ((1ULL << max64_bits)-1)), (int)64-max64_bits, max64_bits);          \
  printf(" Cnt:%0*llX/%02d", (56+3)/4, (unsigned long long)cnt, cnt_bits);                                              \
  for (int i = 0; i < ((__state_count)-8); i += join_count) {                                                           \
    if (i == 0) printf(" ");                                                                                            \
    int val = 0; for (int k = 0; k < bit_count; ++k) {                                                                  \
      val <<= 1; for (int j = 0; j < join_count; ++j) if ((__state_ptr)[8+i+join_count*k+j]) { val |= 1; break; }       \
    }                                                                                                                   \
    printf("%1X", val);                                                                                                 \
  }                                                                                                                     \
  /*printf(" "); const int bits_count = ((max_bits)+(__width-1))/__width;                                               \
  for (int i = 0; i < max_bits; i += bits_count) printf("%c", (idx_bits) >= i ? '1' : '0');                             \
  printf("*%d", bits_count);                                                                                          */\
  printf(" %.5f%%", (float)(max*(__state_ptr)[6+1] + idx*((__state_ptr)[6+0]-(__state_ptr)[6+1])) * 100.0/slc);         \
  printf(" (%.3f MKeys/s" " ETA +%.0fs" " [%d:%02d:%02d:%02d]" ")",                                                     \
            ((float)cnt+0.5)/(__time)/(1000.0f*1000.0f),                                                                \
            ((float)((1ULL<<56)/((cnt+0.5f)/(__time)))),                                                                \
      (int)(((float)((1ULL<<56)/((cnt+0.5f)/(__time))) - (__time))/3600)/24,                                            \
      (int)(((float)((1ULL<<56)/((cnt+0.5f)/(__time))) - (__time))/3600)%24,                                            \
      (int)(((float)((1ULL<<56)/((cnt+0.5f)/(__time))) - (__time))/  60)%60,                                            \
      (int)(((float)((1ULL<<56)/((cnt+0.5f)/(__time))) - (__time))     )%60);                                           \
  if (__error != -1) printf(" (%d)", __error); printf("\r");                                                            \
} while (0)

/*
      
*/
//    uint32_t lastStrts = ~0L, lastStops = ~0L;
    time_t startTime = time(NULL);

    printf("\n");
    state_print(time(NULL)-startTime, state_ptr, state_count, 20, -1 /*e*/);
    uint32_t lastStrts = state_ptr[6+0], lastStops = state_ptr[6+1];
    printf("\n");
    fflush(stdout);
    e = cudaEventQuery(stop);
    sleep(1);
      
    while (true && (e = cudaEventQuery(stop)) == cudaErrorNotReady) {
      checkCUDAError("cudaEventQuery to query stop event");

      cudaEvent_t memcpydone;
      cudaEventCreate(&memcpydone);
      checkCUDAError("cudaEventCreate memcpydone");

      // without no state can be displayed (!!)
      printf("\r  \r");
      fflush(stdout);
      sleep(1);

      cudaMemcpyAsync(state_ptr, d_state_ptr, state_size, cudaMemcpyDeviceToHost, state_stream);
      checkCUDAError("cudaMemcpyAsync from device state_array");

      printf("\r.  \r");
      fflush(stdout);
      sleep(1);
  
      cudaEventRecord(memcpydone, state_stream);
      checkCUDAError("cudaEventRecord memcpydone");

      printf("\r.. \r");
      fflush(stdout);
      sleep(1);
    
      cudaEventSynchronize(memcpydone);
      checkCUDAError("cudaEventSynchronize memcpydone");

      printf("\r ..\r");
      fflush(stdout);
      sleep(1);

      //cudaStreamSynchronize(state_stream);
      //checkCUDAError("cudaStreamSynchronize state_stream");

      printf("\r  .\r");
      fflush(stdout);
      
      cudaEventDestroy(memcpydone);
      checkCUDAError("cudaEventDestroy memcpydone");

      printf("\r   \r");
      if (lastStrts != state_ptr[6+0] || lastStops != state_ptr[6+1]) printf("\n");
      state_print(time(NULL)-startTime, state_ptr, state_count, 20, -1 /*e*/);
      lastStrts = state_ptr[6+0]; lastStops = state_ptr[6+1];
      printf("\n");
      fflush(stdout);
    }
    printf("done (e=%d).\n", e);

     //stop recorder and print time
    cudaEventSynchronize(stop);
    checkCUDAError("cudaEventSynchronize stop");

    cudaStreamSynchronize(state_stream);
    checkCUDAError("cudaStreamSynchronize state_stream");    

    cudaStreamSynchronize(kernel_stream);
    checkCUDAError("cudaStreamSynchronize state_stream");    
    
    //copy back to host
//    cudaMemcpy(key, d_key, real_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&key, d_key, sizeof(*d_key), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy from device key");

    cudaMemcpy(&keyfound, d_keyfound, sizeof(*d_keyfound), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy from device keyfound");

    cudaMemcpy(state_ptr, d_state_ptr, state_size, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpyAsync from device state_array");

    cudaDeviceSynchronize();
    checkCUDAError("cudaDeviceSynchronize stop");

    state_print(time(NULL)-startTime, state_ptr, state_count, 20, -1 /*e*/);
    printf("\n");

    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("result is : ");
    if (/*equals(startkey,key)*/ !keyfound){
        printf("Key was not found\n");
    } else {
//        displayData(key, size);
        displayData(key);
    }
    printf("Elapsed time is: %.03fs ($%llX keys, %.3f Mkeys/s : max. %.0fs [%d:%02d:%02d:%02d]\n",
      elapsedTime/1000.0, 
      (unsigned long long)*((uint64_t *)&(state_ptr)[4]),
      (float)*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0)/(1000*1000),
      (float)((1ULL<<56)/(*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0))+0.5),
      (int)(((1ULL<<56)/(*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0))+1)/3600)/24,
      (int)(((1ULL<<56)/(*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0))+1)/3600)%24,
      (int)(((1ULL<<56)/(*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0))+1)/  60)%60,
      (int)(((1ULL<<56)/(*((uint64_t *)&(state_ptr)[4])/(elapsedTime/1000.0))+1)     )%60
    );
      
    //
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Free device memory
    cudaFreeHost(state_ptr);
    cudaStreamDestroy(state_stream);
    cudaFree(d_state_ptr);
    cudaFree(d_key);
    //cudaFree(d_plain);
    //cudaFree(d_cipher);
    cudaFree(d_keyfound);

    return keyfound ? key : 0ULL;
}

void newKey(unsigned char* key)
{
    ++*(uint64_t *)key;
}

int equals(const unsigned char* a, const unsigned char* b)
{
    return (*(uint64_t*)a == *(uint64_t*)b);
}

void displayData(const unsigned char* data, int size)
{
    for (int i = 0; i<size; ++i){
        printf("%02X", data[i]);
    }
    printf("\t");
    for (int i = 0; i<size; ++i){
        printf("%c",isprint(data[i])?data[i]:'.');
    }
    printf("\n");
}

void displayData(const uint64_t data)
{
    uint8_t buf[8];
    uint64to8(&buf[0], data);
    displayData(&buf[0], sizeof(buf));
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

unsigned char* convert(char *s)
{
    unsigned char* val = (unsigned char*) malloc(strlen(s)/2);
    /* WARNING: no sanitization or error-checking whatsoever */
    for(int count = 0; count < sizeof(val)/sizeof(val[0]); count++) {
        sscanf(s, "%2hhx", &val[count]);
        s += 2 * sizeof(char);
    }
    return val;
}

__global__ static void DESencKernel(sboxes_block &cipher, const sboxes_block plain, const sboxes_deskey key) {
  sboxes_bitslice plainslice[64];
  sboxes_bitslice cipherslice[64];
  sboxes_bitslice keyslice[56];
  sboxes_bitslice temp[64];
  
  sboxes_set_data(plainslice, plain);
  sboxes_desslice_set(temp, plainslice);
  sboxes_desslice_set64(temp, plain); 

  sboxes_set_key64(keyslice, key);
  sboxes_desslice_enc(temp, keyslice);

  sboxes_desslice_get(cipherslice, temp);
  cipher = sboxes_get_data(cipherslice);

  cipher = sboxes_desslice_get64(temp);
  return;
}

__global__ static void DESdecKernel(sboxes_block &plain, const sboxes_block cipher, const sboxes_deskey key) {
  sboxes_bitslice plainslice[64];
  sboxes_bitslice cipherslice[64];
  sboxes_bitslice keyslice[56];
  sboxes_bitslice temp[64];

  sboxes_set_data(cipherslice, cipher);
  sboxes_desslice_set(temp, cipherslice);
  sboxes_desslice_set64(temp, cipher);

  sboxes_set_key64(keyslice, key);
  sboxes_desslice_dec(temp, keyslice);

  sboxes_desslice_get(plainslice, temp);
  plain = sboxes_get_data(plainslice);

  plain = sboxes_desslice_get64(temp);
  return;
}

__global__ static void TDESencKernel(sboxes_block &cipher, const sboxes_block plain, const sboxes_deskey key1h, const sboxes_deskey key2l) {
  sboxes_bitslice plainslice[64];
  sboxes_bitslice cipherslice[64];
  sboxes_bitslice keyslice1h[56];
  sboxes_bitslice keyslice2l[56];
  sboxes_bitslice temp[64];
  
  sboxes_set_data(plainslice, plain);
  sboxes_desslice_set(temp, plainslice);
  sboxes_set_key64(keyslice1h, key1h);
  sboxes_set_key64(keyslice2l, key2l);
  sboxes_desslice_tenc(temp, keyslice1h, keyslice2l);

  sboxes_desslice_get(cipherslice, temp);
  cipher = sboxes_get_data(cipherslice);
  return;
}

__global__ static void TDESdecKernel(sboxes_block &plain, const sboxes_block cipher, const sboxes_deskey key1h, const sboxes_deskey key2l) {
  sboxes_bitslice plainslice[64];
  sboxes_bitslice cipherslice[64];
  sboxes_bitslice keyslice1h[56];
  sboxes_bitslice keyslice2l[56];
  sboxes_bitslice temp[64];
  
  sboxes_set_data(cipherslice, cipher);
  sboxes_desslice_set(temp, cipherslice);
  sboxes_set_key64(keyslice1h, key1h);
  sboxes_set_key64(keyslice2l, key2l);
  sboxes_desslice_tdec(temp, keyslice1h, keyslice2l);
 
  sboxes_desslice_get(plainslice, temp);
  plain = sboxes_get_data(plainslice);
  return;
}

static sboxes_block cudaDESenc(const sboxes_block plain, const sboxes_deskey key) {
  sboxes_block cipher;
  sboxes_block *d_cipher;

  cudaMalloc(&d_cipher, sizeof(*d_cipher));
  checkCUDAError("cudaMalloc d_cipher");

  DESencKernel << <1, 1>> >(*d_cipher, plain, key);
  checkCUDAError("cudakernel DESencKernel call");

  cudaMemcpy(&cipher, d_cipher, sizeof(*d_cipher), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from device cipher");

  cudaDeviceSynchronize();
  checkCUDAError("cudaDeviceSynchronize");

  cudaFree(d_cipher);
  return cipher;
}

static sboxes_block cudaDESdec(const sboxes_block cipher, const sboxes_deskey key) {
  sboxes_block plain;
  sboxes_block *d_plain;

  cudaMalloc(&d_plain, sizeof(*d_plain));
  checkCUDAError("cudaMalloc d_plain");

  DESdecKernel << <1, 1>> >(*d_plain, cipher, key);
  checkCUDAError("cudakernel DESdecKernel call");

  cudaMemcpy(&plain, d_plain, sizeof(*d_plain), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from device plain");

  cudaDeviceSynchronize();
  checkCUDAError("cudaDeviceSynchronize");

  cudaFree(d_plain);
  return plain;
}

static sboxes_block cudaTDESenc(const sboxes_block plain, const sboxes_deskey key1h, const sboxes_deskey key2l) {
  sboxes_block cipher;
  sboxes_block *d_cipher;

  cudaMalloc(&d_cipher, sizeof(*d_cipher));
  checkCUDAError("cudaMalloc d_cipher");

  TDESencKernel << <1, 1>> >(*d_cipher, plain, key1h, key2l);
  checkCUDAError("cudakernel TDESencKernel call");

  cudaMemcpy(&cipher, d_cipher, sizeof(*d_cipher), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from device cipher");

  cudaDeviceSynchronize();
  checkCUDAError("cudaDeviceSynchronize");

  cudaFree(d_cipher);
  return cipher;
}

static sboxes_block cudaTDESdec(const sboxes_block cipher, const sboxes_deskey key1h, const sboxes_deskey key2l) {
  sboxes_block plain;
  sboxes_block *d_plain;

  cudaMalloc(&d_plain, sizeof(*d_plain));
  checkCUDAError("cudaMalloc d_plain");

  TDESdecKernel << <1, 1>> >(*d_plain, cipher, key1h, key2l);
  checkCUDAError("cudakernel TDESdecKernel call");

  cudaMemcpy(&plain, d_plain, sizeof(*d_plain), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from device plain");

  cudaDeviceSynchronize();
  checkCUDAError("cudaDeviceSynchronize");

  cudaFree(d_plain);
  return plain;
}

////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <vector>
#include <algorithm>

static size_t     knownPlainCipherCount   = 0;
static uint64_t * knownCipher2PlainArray  = NULL;
static uint64_t * knownPlain2CipherArray  = NULL;

static bool readKnownFile(const char *filename, const char *cachefile) {
  FILE *fp;
  uint64_t ecw = 0, ccw = 0;
  char inbuf[1024];
  char *str;
  int32_t i, count = 0;
  
  if ((fp = fopen(cachefile, "rb")) != NULL) {
    fseek(fp, 0, SEEK_END);
    count = ftell(fp);
    knownPlainCipherCount = count / (sizeof(uint64_t)*2) / 2;
    printf("opened cache file '%s' (size=%.3fkb)...\n", cachefile, (float)count/1024.0);
    fflush(stdout);
    knownCipher2PlainArray = (uint64_t *)calloc(knownPlainCipherCount, sizeof(uint64_t)*2);
    knownPlain2CipherArray = (uint64_t *)calloc(knownPlainCipherCount, sizeof(uint64_t)*2);
    fseek(fp, 0, SEEK_SET);
    count  = fread(&knownPlain2CipherArray[0], sizeof(uint64_t)*2, knownPlainCipherCount, fp);
    count += fread(&knownCipher2PlainArray[0], sizeof(uint64_t)*2, knownPlainCipherCount, fp);
    printf("#entries=%lu read (%.3fkb)\n", (unsigned long)knownPlainCipherCount, (float)count*sizeof(uint64_t)*2/1024.0);
    fclose(fp);   
    printf("closed cache file '%s'\n", cachefile);
  } else if ((fp = fopen(filename, "rt")) != NULL) {
    std::map<uint64_t, uint64_t> knownCipher2PlainMap;
    std::map<uint64_t, uint64_t> knownPlain2CipherMap;

    printf("opened cw file '%s'\n", filename);
    knownCipher2PlainMap.clear();
    knownPlain2CipherMap.clear();
    
    fseek(fp, 0, SEEK_END);
    const size_t filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    count = 0;
    while (!feof(fp) && (str = fgets(&inbuf[0], sizeof(inbuf)/sizeof(inbuf[0])-1, fp)) != NULL) {
      if (str != NULL) if (str = strchr(&str[0], ',')) ++str;
      if (str != NULL) if (str = strchr(&str[0], ',')) ++str;
      ecw = ccw = 0;
      if (str != NULL) {
//        printf("buf=%s\n", str);
        for (ecw = 0, i = 0; i < 16; ++i, ++str) ecw <<= 4, ecw |= (toupper(*str)>='A'&&toupper(*str)<='F')?(toupper(*str)-'A'+10):(*str-'0');
        ++str;
        for (ccw = 0, i = 0; i < 16; ++i, ++str) ccw <<= 4, ccw |= (toupper(*str)>='A'&&toupper(*str)<='F')?(toupper(*str)-'A'+10):(*str-'0');
        ++str;
        if (ccw == 0) 
        for (ccw = 0, i = 0; i < 16; ++i, ++str) ccw <<= 4, ccw |= (toupper(*str)>='A'&&toupper(*str)<='F')?(toupper(*str)-'A'+10):(*str-'0');
      }
      if (ecw != 0 && ccw != 0) {
        ++count;
        knownCipher2PlainMap[ecw] = ccw;
        knownPlain2CipherMap[ccw] = ecw;
        printf("ECW=%016llX CCW=%016llX (#%d %.2f%%)\r", (unsigned long long)ecw, (unsigned long long)ccw, count, (float)ftell(fp) * 100.0/filesize);
      }
    } while (!feof(fp));
    fclose(fp);
    printf("\nclosed %s\n", filename);
    printf("cipher2plain #entries=%u\n", (unsigned int)knownCipher2PlainMap.size());
    printf("plain2cipher #entries=%u\n", (unsigned int)knownPlain2CipherMap.size());

    std::map<uint64_t, uint64_t>::iterator knownCipher2PlainMapIt;
    std::vector<std::pair<uint64_t, uint64_t>> knownCipher2PlainVector;
    std::vector<std::pair<uint64_t, uint64_t>> knownPlain2CipherVector;
    
    knownPlain2CipherVector.resize(count);
    knownCipher2PlainVector.resize(count);
    for (knownCipher2PlainMapIt = knownCipher2PlainMap.begin(), i = 0; i < knownCipher2PlainVector.size() && knownCipher2PlainMapIt != knownCipher2PlainMap.end(); ++knownCipher2PlainMapIt, ++i) {
      knownPlain2CipherVector[i] = std::pair<uint64_t, uint64_t>(knownCipher2PlainMapIt->first, knownCipher2PlainMapIt->second);
      knownCipher2PlainVector[i] = std::pair<uint64_t, uint64_t>(knownCipher2PlainMapIt->second, knownCipher2PlainMapIt->first);
    }
    sort(knownCipher2PlainVector.begin(), knownCipher2PlainVector.end());
    sort(knownPlain2CipherVector.begin(), knownPlain2CipherVector.end());
    knownPlainCipherCount = count;
    knownCipher2PlainArray = (uint64_t *)calloc(knownPlainCipherCount, sizeof(uint64_t)*2);
    knownPlain2CipherArray = (uint64_t *)calloc(knownPlainCipherCount, sizeof(uint64_t)*2);
    for (i = 0; i < count; ++i) {
      knownCipher2PlainArray[i*2+0] = knownCipher2PlainVector[i].first;
      knownCipher2PlainArray[i*2+1] = knownCipher2PlainVector[i].second;
      knownPlain2CipherArray[i*2+0] = knownPlain2CipherVector[i].first;
      knownPlain2CipherArray[i*2+1] = knownPlain2CipherVector[i].second;
    }
    if ((fp = fopen(cachefile, "wb")) != NULL) {
      printf("opened cache file '%s'\n", cachefile);
      count  = fwrite(&knownPlain2CipherArray[0], sizeof(uint64_t)*2, knownPlainCipherCount, fp);
      count += fwrite(&knownCipher2PlainArray[0], sizeof(uint64_t)*2, knownPlainCipherCount, fp);
      printf("#entries=%lu written (%.3fkb)\n", (unsigned long)knownPlainCipherCount, (float)count*sizeof(uint64_t)*2/1024);
      fclose(fp);
      printf("closed cache file '%s'\n", cachefile);
    }
    knownPlain2CipherVector.clear();
    knownCipher2PlainVector.clear();
    knownPlain2CipherMap.clear();
    knownCipher2PlainMap.clear();
  }
  return knownPlainCipherCount;
}

void parseArgs(int argc, char** argv)
{
//    char c;
//    char* cipherIn;
//    char* keyIn;
#if 0
    int optionIndex = 0;
    struct option longOption[]=
    {
        {"plaintext",1,NULL,'p'},
        {"ciphertext",1,NULL,'c'},
        {"startkey",1,NULL,'k'},
        {"serial",1,NULL,'s'},
        {"verbose",1,NULL,'v'},
        {0,0,0,0}
    };
    if (argc < 6) 
    {
        printf("Wrong number of arguments\n");
        exit(1);
    }
    while((c=getopt_long(argc,argv,"p:c:k:sv",longOption,&optionIndex))!=-1)
    {
        switch(c)
        {
            case 'p':
                plain = (unsigned char*)strdup(optarg);
                break;
            case 'c':
                cipherIn = strdup(optarg);
                cipher = convert(cipherIn);
                break;
            case 'k':
                keyIn = strdup(optarg);
                key = convert(keyIn);
                break;
            case 's':
                isSerial = 1;
                break;
            case 'v':
                verbose = 1;
                break;
            default:
                printf("Bad argument %c\n",c);
                exit(1);
        }
    }    
#else

    uint64_t delta_short = 0;
    uint64_t delta_abs = 0;
    int delta_bits_short = 0;
    int delta_bits = 0;
    int step_size_bits = 1;
    int c = 1;
    int vz = 1;

    if (argc > c && argv[c][0] == '#' && isdigit(argv[c][1])) deviceId = atoi(&argv[c++][1]);
    if (argc > c && strlen(argv[c]) == 16) key = convert(argv[c++]);
        if (argc > c && strlen(argv[c]) <= 3 && (atoi(argv[c]) != 0 || argv[c][0] == '+' || argv[c][0] == '-')) {
      vz = argv[c][0] == '-' ? -1 : +1;
      delta_short = 1ULL<<(delta_bits_short = atoi(&argv[c++][vz < 0 ? 1 : 0]));
      delta_abs = 
        ( (delta_short & (((1ULL<<7)-1) <<  0)) << 1) |
        ( (delta_short & (((1ULL<<7)-1) <<  7)) << 2) |
        ( (delta_short & (((1ULL<<7)-1) << 14)) << 3) |
        ( (delta_short & (((1ULL<<7)-1) << 21)) << 4) |
        ( (delta_short & (((1ULL<<7)-1) << 28)) << 5) |
        ( (delta_short & (((1ULL<<7)-1) << 35)) << 6) |
        ( (delta_short & (((1ULL<<7)-1) << 42)) << 7) |
        ( (delta_short & (((1ULL<<7)-1) << 49)) << 8) ;
      if (delta_abs != 0) for (uint64_t i = delta_abs; (i & 1) == 0; i >>= 1) ++delta_bits;
//      if (delta_bits_short < 0) delta = -delta_abs, delta_bits = -delta_bits;
//      if (vz < 0) delta_bits = -delta_bits;
    }
    if (argc > c && argv[c][0] == '/') decrypt = 1, argv[c][1] ? *++argv[c] : ++c;
    if (argc > c && strlen(argv[c]) <= 3 && argv[c][0] == '*') step_size = 1ULL<<(step_size_bits = atoi(&argv[c++][1]));
    if (argc > (c+1) && strlen(argv[c+0]) == 16 &&strlen(argv[c+1]) == 16) {
      plain  = convert(argv[c++]);
      cipher = convert(argv[c++]);
    }

    if (argc != c || (argc > c && (strcmp(argv[c], "/?") == 0 || strcmp(argv[c], "-h") == 0))) {
        printf("Wrong number of arguments (%d/%d)\n", argc, c);
        printf("Usage: %s [#deviceId] [[startkey-hex] [+n|-n : key-delta-2^n] [/ : decrypt] [*n : step-in-2^n] [<64-bit-plaintext-hex> <64-bit-ciphertext-hex>]\n", argv[0]);
        exit(1);
    }


    printf("plaintext : %02X%02X%02X%02X%02X%02X%02X%02X\n", plain[0], plain[1], plain[2], plain[3], plain[4], plain[5], plain[6], plain[7]);
    printf("ciphertext: %02X%02X%02X%02X%02X%02X%02X%02X\n", cipher[0], cipher[1], cipher[2], cipher[3], cipher[4], cipher[5], cipher[6], cipher[7]);
    printf("key       : %02X%02X%02X%02X%02X%02X%02X%02X\n", key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
    printf("delta56   :%c%016llX/%c%d\n",                       vz < 0 ? '-' : '+', (unsigned long long)delta_short, vz < 0 ? '-' : '+', delta_bits_short);
    printf("delta64   :%c%016llX/%c%d\n",                       vz < 0 ? '-' : '+', (unsigned long long)delta_abs,   vz < 0 ? '-' : '+', delta_bits);
/*
    printf("delta_key :%c%02X%02X%02X%02X%02X%02X%02X%02X\n", delta_bits       < 0 ? '-' : '+',
      ((const uint8_t *)&delta_abs)[0],
      ((const uint8_t *)&delta_abs)[1],
      ((const uint8_t *)&delta_abs)[2],
      ((const uint8_t *)&delta_abs)[3],
      ((const uint8_t *)&delta_abs)[4],
      ((const uint8_t *)&delta_abs)[5],
      ((const uint8_t *)&delta_abs)[6],
      ((const uint8_t *)&delta_abs)[7]);
*/
//    printf("deltakey64:%c%016llX\n", vz < 0 ? '-' : '+', (unsigned long long)delta_abs);
//    (*(uint64_t *)&key[0]) += delta;
    uint64_t newkey64 = des56to64(des64to56(uint8to64(&key[0])) + vz*des64to56(delta_abs));
    static uint8_t newkey[17];
    uint64to8(&newkey[0], newkey64);
    key = newkey;
    printf("startkey  : %02X%02X%02X%02X%02X%02X%02X%02X\n", key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
    printf("stepsize56: %016llX/%d\n", (long long)           step_size ,  step_size_bits);
    printf("stepsize64: %016llX/%d\n", (long long) des56to64(step_size), (step_size_bits*8+7)/7);
    printf("decrypt   : %d\n", (int)decrypt);
    
    

    isSerial = 0;
    verbose = 1;
#endif
}
/*
 * Main routine
 */
int main( int argc, char** argv )
{
  des_context my_ctx;
  unsigned char buf[8];
  uint64_t foundkey;

  cipher = _cipher;

  /*    static unsigned char my_keys[8] =
  {
      0x60, 0x65, 0x79, 0x69, 0x65, 0x79, 0x6B, 0x65
  };
  static const unsigned char my_keys[24] =
  {
      0x6B, 0x65, 0x79, 0x6B, 0x65, 0x79, 0x6B, 0x65,
      0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01,
      0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23
  };

  static const unsigned char my_plain[3][8] =
  {
      { 0x70, 0x6C, 0x61, 0x69, 0x6E, 0x31, 0x32, 0x33 },
      { 0x70, 0x6C, 0x61, 0x69, 0x6E, 0x34, 0x35, 0x36 },
      { 0x70, 0x6C, 0x61, 0x69, 0x6E, 0x37, 0x38, 0x39 }
  }; 

  static unsigned char my_cipher[8] =
  {
      0x1B, 0xCD, 0xB8, 0x89, 0x88, 0xE2, 0x02, 0x7F
  };   
  */
  printf("\n");


  int rc = 0;
  
  if (true) {
    readKnownFile("mscDES.csv", "mscDES.bin");
  }
  
  if (true) {
    sboxes_deskey K3_H  = 0x84D5EA9EE5752730ULL;
    sboxes_deskey K3_L  = 0x93642BD887669FCFULL;
    sboxes_deskey EK2_H = 0x2021222324252627ULL;
    sboxes_deskey EK2_L = 0x28292A2B2C2D2E2FULL;
    sboxes_deskey EK1_H = 0x1011121314151617ULL;
    sboxes_deskey EK1_L = 0x18191A1B1C1D1E1FULL;
    sboxes_deskey ECW1  = 0x3A77C880A42AF2BBULL;
    sboxes_deskey PCW1  = 0xBCFBB26913BABE8BULL;
    sboxes_deskey ECW2  = 0xB3A95B27DC867E38ULL;
    sboxes_deskey PCW2  = 0x68E1DA5B24AD861FULL;
    
    printf("[MSC] testK3       > %016llX%016llX\n", (unsigned long long)K3_H, (unsigned long long)K3_L);
    printf("[MSC] testEK2      > %016llX%016llX\n", (unsigned long long)EK2_H, (unsigned long long)EK2_L);
    sboxes_deskey K2_H = cudaDESdec(cudaDESenc(cudaDESdec(EK2_H, K3_H), K3_L), K3_H);
    sboxes_deskey K2_L = cudaDESdec(cudaDESenc(cudaDESdec(EK2_L, K3_H), K3_L), K3_H);
    printf("[MSC] testK2       > %016llX%016llX\n", (unsigned long long)K2_H, (unsigned long long)K2_L);
    sboxes_deskey K2_3des_H = cudaTDESdec(EK2_H, K3_H, K3_L);
    sboxes_deskey K2_3des_L = cudaTDESdec(EK2_L, K3_H, K3_L);
    printf("[MSC] testK2_3ses  > %016llX%016llX [%s]\n", (unsigned long long)K2_3des_H, (unsigned long long)K2_3des_L, K2_H == K2_3des_H && K2_L == K2_3des_L ? ("OK") : (rc = 1, "FAIL"));
    
    printf("[MSC] testEK1      > %016llX%016llX\n", (unsigned long long)EK1_H, (unsigned long long)EK1_L);
    sboxes_deskey K1_H = cudaDESdec(cudaDESenc(cudaDESdec(EK1_H, K2_H), K2_L), K2_H);
    sboxes_deskey K1_L = cudaDESdec(cudaDESenc(cudaDESdec(EK1_L, K2_H), K2_L), K2_H);
    printf("[MSC] testK1       > %016llX%016llX\n", (unsigned long long)K1_H, (unsigned long long)K1_L);
    sboxes_deskey K1_3des_H = cudaTDESdec(EK1_H, K2_H, K2_L);
    sboxes_deskey K1_3des_L = cudaTDESdec(EK1_L, K2_H, K2_L);
    printf("[MSC] testK1_3des  > %016llX%016llX [%s]\n", (unsigned long long)K1_3des_H, (unsigned long long)K1_3des_L, K1_H == K1_3des_H && K1_L == K1_3des_L ? ("OK") : (rc = 1, "FAIL"));

    printf("[MSC] testECW1     > %016llX\n", (unsigned long long)ECW1);
    sboxes_deskey CW1 = cudaDESdec(cudaDESenc(cudaDESdec(ECW1, K1_H), K1_L), K1_H);
    printf("[MSC] tessCW1      > %016llX [%s]\n", (unsigned long long)CW1, CW1 == PCW1 ? "OK" : "FAIL");
    sboxes_deskey CW1_3des = cudaTDESdec(ECW1, K1_H, K1_L);
    printf("[MSC] tessCW1_3des > %016llX [%s]\n", (unsigned long long)CW1, CW1_3des == PCW1 ? ("OK") : (rc = 1, "FAIL"));
    printf("[MSC] tessPCW1     > %016llX\n", (unsigned long long)PCW1);

    printf("[MSC] testECW2     > %016llX\n", (unsigned long long)ECW2);
    sboxes_deskey CW2 = cudaDESdec(cudaDESenc(cudaDESdec(ECW2, K1_H), K1_L), K1_H);
    printf("[MSC] tessCW2      > %016llX [%s]\n", (unsigned long long)CW2, CW2 == PCW2 ? "OK" : "FAIL");
    sboxes_deskey CW2_3des = cudaTDESdec(ECW2, K1_H, K1_L);
    printf("[MSC] tessCW2_3des > %016llX [%s]\n", (unsigned long long)CW2, CW2_3des == PCW2 ? ("OK") : (rc = 1, "FAIL"));
    printf("[MSC] tessPCW2     > %016llX\n", (unsigned long long)PCW2);
    
    if (CW1 == PCW1 && CW2 == PCW2) {
      for (int32_t i = 0; i < 16; i += 4) {
        uint64_t XOR_Parity_H = (i>=8)?(1ULL<<((i-8)*8+0)):0;
        uint64_t XOR_Parity_L = (i<=7)?(1ULL<<((i-0)*8+0)):0;
        printf("[MSC] testXOR_%02d   > %016llX%016llX\n", i, (unsigned long long)XOR_Parity_H, (unsigned long long)XOR_Parity_L);

        K1_H = sboxes_deskey(K1_H ^ XOR_Parity_H);
        K1_L = sboxes_deskey(K1_L ^ XOR_Parity_L);
        printf("[MSC] testK1__%02d   > %016llX%016llX\n", i, (unsigned long long)K1_H, (unsigned long long)K1_L);
        
        sboxes_deskey CW1 = cudaDESdec(cudaDESenc(cudaDESdec(ECW1, K1_H), K1_L), K1_H);
        printf("[MSC] testCW1_%02d   > %016llX [%s]\n", i, (unsigned long long)CW1, CW1 == PCW1 ? "OK" : "FAIL");
        if (CW1 != PCW1) break;

        sboxes_deskey CW2 = cudaDESdec(cudaDESenc(cudaDESdec(ECW2, K1_H), K1_L), K1_H);
        printf("[MSC] testCW2_%02d   > %016llX [%s]\n", i, (unsigned long long)CW2, CW2 == PCW2 ? "OK" : "FAIL");
        if (CW2 != PCW2) break;
      }
    }
    if (CW1 != PCW1 || CW2 != PCW2) rc = 1;
    printf("\n");
  }

  parseArgs(argc,argv);

  if (true) {
      sboxes_bitslice plainslice[64];
      sboxes_bitslice cipherslice[64];
      sboxes_bitslice keyslice[56];

      sboxes_set_data(plainslice, uint8to64(&plain[0]));
      printf("plainval  : %016llX\n", (unsigned long long)sboxes_get_data(plainslice));

      sboxes_set_data(cipherslice, uint8to64(&cipher[0]));
      printf("cipherval : %016llX\n", (unsigned long long)sboxes_get_data(cipherslice));

      sboxes_set_key64(keyslice, uint8to64(&_key[0]));
      printf("keyval    : %016llX\n", (unsigned long long)sboxes_get_key64(keyslice, sboxes_sliceon));
      
  /*
      sboxes_bitslice temp[64];
      sboxes_bitslice newcipherslice[64];
      sboxes_desslice_set(temp, plainslice);
      sboxes_desslice_enc(temp, keyslice);
      sboxes_desslice_get(newcipherslice, temp);
      printf("newcipher : %016llX\n", (unsigned long long)sboxes_get_data(newcipherslice));
      // sboxes_get_data(newcipherslice)

      sboxes_bitslice newplainslice[64];
      sboxes_desslice_set(temp, cipherslice);
      sboxes_desslice_dec(temp, keyslice);
      sboxes_desslice_get(newplainslice, temp);
      printf("newplain  : %016llX\n", (unsigned long long)sboxes_get_data(newplainslice));
      // sboxes_get_data(newplainslice) 
  */

      sboxes_block newcipher = cudaDESenc(uint8to64(&plain[0]), uint8to64(&_key[0]));
      printf("newcipher : %016llX\n", (unsigned long long)newcipher);

      sboxes_block newplain = cudaDESdec(uint8to64(&cipher[0]), uint8to64(&_key[0]));
      printf("newplain  : %016llX\n", (unsigned long long)newplain);
      
      if (uint8to64(&plain[0]) != newplain ||
          uint8to64(&cipher[0]) != newcipher) {
        //for (int i = 0; i < 64; ++i) printf("newcipherslice[%d] = %016llX\n", i, (unsigned long long)newcipherslice[i]);
        rc = 1;
      }
      printf("\n");
  }
  if (rc) {
    printf("ERROR\n");
    return rc;
  }


  if (verbose) {
      printf("start key : ");
      displayData(key, 8);
      printf("plain     : ");
      displayData(plain, 8);
      printf("cipher    : ");
      displayData(cipher, 8);
  }
  printf("\n");

  //
  if(isSerial == 0)
  {
      printf("Running the CUDA implementation (step_size=%lld)\n", (long long)step_size);
      memcpy(buf, key, 8);
      key = buf;
      //        cudaFunction(key,plain,cipher,8);
      foundkey = cudaFunction(uint8to64(&key[0]),uint8to64(&plain[0]),uint8to64(&cipher[0]),(decrypt ? -1:+1)*step_size);
      if (foundkey != 0ULL) fprintf(stderr, "\nOK (KEY $%016llX FOUND)\n", (unsigned long long)foundkey); else fprintf(stderr, "\nERROR (KEY NOT FOUND)\n");
      return (foundkey == 0ULL);
        //
  }
  else
  {
      printf("Running the serial implementation\n");
      cudaEvent_t start, stop;
      float elapsedTime;
      int size = 8;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start,0);
      

      printf("=====START======\n");
      
      
      int keyfound = 0;
      long i = 0;

      printf("plain cpu : ");
      displayData(plain, size);
      printf("strtkeycpu: ");
      displayData(key, size);
      unsigned char my_key[8];
      memcpy(my_key,key,size);
      unsigned char found_key[8];
      memcpy(found_key, key, size);

      while(i<500000000 && !(keyfound))
      {
  /*            if ( i % 100000 == 0){
              printf("loop %i!!! found: %i my key:%c %02x   %c %02x   %c %02x   %c %02x   \n",i,keyfound,my_key[0],my_key[0],my_key[1],my_key[1],my_key[2],my_key[2],my_key[3],my_key[3]);
          }
  */
          des_setkey_enc ( &my_ctx, my_key);


          des_crypt_ecb ( &my_ctx, plain, buf );

          if (equals(buf, cipher))
          {
              printf("!!! KEY FOUND (loop %li)!!!\n",i);
              keyfound = 1;
              memcpy(found_key, my_key, size);
              break;
          }

          newKey(my_key);
          ++i;

      }

      printf("=====END========\n");

      printf("\n");

      //stop recorder and print time
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime,start,stop);
      printf("result is : ");
      if (equals(key,found_key)){
          printf("Key was not found\n");
      } else {
          displayData(my_key, size);
      }
      printf("Elapsed time is: %f\n",elapsedTime);
      //
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

  }
  return ( 0 );
}
