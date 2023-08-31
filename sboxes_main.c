#pragma once
/*
 * Test the automatically-generated DES implementation.
 *
 * Key should be - 5e d9 20 4f ec e0 b9 67
 *
 * Written by Matthew Kwan - April 1997.
 */

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/time.h>
#else /* WIN32 */
#include <time.h>
#endif /* WIN32 */

#include "sboxes_deseval.h"

/*
 * Local variables.
 */
static int	bitlength = bitsizeof(sboxes_bitslice);
static int	bitlength_log2 = 5; //bitlength==16?4:bitlength==32?5:bitlength==64?6:bitlength==128?7:bitlength==256:8; //bitcountof(sboxes_bitslice);

/*
 * Set the bit length variables.
 */
static void set_bitlength(void) {
	sboxes_bitslice	x = 0;
  --x;

	bitlength = 0;
	for (x = 0, --x; x != 0; x >>= 1)
	    bitlength++;

	printf ("%d-bit machine\n", bitlength);

	if (bitlength == 64)
	    bitlength_log2 = 6;
	else if (bitlength == 32)
	    bitlength_log2 = 5;
	else {
	    fprintf (stderr, "Cannot handle %d-bit machines\n", bitlength);
	    exit (1);
	}
}

/*
 * Test the DES key evaluation routine for speed.
 */
static void test_speed(const sboxes_bitslice	*p,
                       const sboxes_bitslice	*c) {
	int		i, t;
	double		td;
	sboxes_bitslice	key[56];
	//struct timeval	start_tv, end_tv;
  time_t start_t, end_t;
  sboxes_deskey val;
	const int	n = (1UL<<24);

	for (i=0; i<56; i++)
	    key[i] = 0;

	val = sboxes_set_low_keys(key);
  printf("%08lX:%08lX\n", (unsigned long)(val>>32), (unsigned long)(val));

	/* Do a dummy run to get the function loaded into memory */
	sboxes_deseval(p, c, key);

	/* Begin the actual run */
	//gettimeofday (&start_tv, NULL);
  start_t = time(NULL);

	for (i=0; i<n; i++) {
    sboxes_bitslice	res = sboxes_deseval (p, c, key);

    if (res != 0) {
      val = sboxes_key_found(key, res);
      printf(" found:%08lX:%08lX\n", (unsigned long)(val>>32), (unsigned long)val);
    }

    val = sboxes_inc_high_keys(key);
    printf("%08lX:%08lX\r", (unsigned long)(val>>32), (unsigned long)(val));
	}

	// gettimeofday (&end_tv, NULL);
  end_t = time(NULL);

  t = (end_t - start_t) * 1000000;
	//t = (end_tv.tv_sec - start_tv.tv_sec) * 1000000 + (end_tv.tv_usec - start_tv.tv_usec);
  td = (double) t / 1000000.0;

	printf ("Searched %.1lf keys per second\n", (double) td /*n*bitlength/td*/);
}

/*
 * Iterate through all 2^56 DES keys.
 */
static void keysearch(const sboxes_bitslice	*p,
                      const sboxes_bitslice	*c,
                      const sboxes_bitslice	*k) {
	int		i;
	sboxes_bitslice	key[56];
  sboxes_deskey val;
  
	for (i=0; i<56; i++)
	    key[i] = k[i];

	val = sboxes_set_low_keys(key);
  printf("%08lX:%08lX\n", (unsigned long)(val>>32), (unsigned long)(val));

	for (;;) {
    sboxes_bitslice	res = sboxes_deseval (p, c, key);

    if (res != 0) {
      val = sboxes_key_found(key, res);
//      printf("%08lX %08lX\r", key[0], key[50]);
      printf(" found:%08lX:%08lX\n", (unsigned long)(val>>32), (unsigned long)(val));
    }

    val = sboxes_inc_high_keys(key);
    //val = sboxes_add_keys(key, 1);
    printf("%08lX:%08lX\r", (unsigned long)(val>>32), (unsigned long)(val));
	}
}

/*
 * Unroll the bits contained in the plaintext, ciphertext, and key values.
 */
static void unroll_bits(sboxes_bitslice		  *p,
                        sboxes_bitslice		  *c,
                        sboxes_bitslice		  *k,
                        const unsigned char	*ivc,
                        const unsigned char	*pc,
                        const unsigned char	*cc,
                        const unsigned char	*kc) {
	int			i;
	unsigned char		ptext[8];

	for (i=0; i<8; i++)
	    ptext[i] = ivc[i] ^ pc[i];

	for (i=0; i<64; i++)
	    if ((ptext[i/8] & (128 >> (i % 8))) != 0)
		p[63 - i] = ~0UL;
	    else
		p[63 - i] = 0;

	for (i=0; i<64; i++)
	    if ((cc[i/8] & (128 >> (i % 8))) != 0)
		c[63 - i] = ~0UL;
	    else
		c[63 - i] = 0;

	if (kc != NULL) {
	    for (i=0; i<56; i++)
		if ((kc[i/7] & (128 >> (i % 7))) != 0)
		    k[55 - i] = ~0UL;
		else
		    k[55 - i] = 0;
	} else {
	    for (i=0; i<56; i++)
		k[i] = 0;
	}
/*
  sboxes_unroll_bits(
      p, c, k, 
      sboxes_char2key(&ivc[0]) ^ sboxes_char2key(&pc[0]), 
      sboxes_char2key(&cc[0]), 
      sboxes_char2key(&kc[0]));
*/
}

/*
 * Set up the sample plaintext, ciphertext and key values.
 */
static void build_samples (sboxes_bitslice	*p,
                           sboxes_bitslice	*c,
                           sboxes_bitslice	*k,
                           int		practice_flag) {
	unsigned char	iv_p[8]
			= {0xa2, 0x18, 0x5a, 0xbf, 0x45, 0x96, 0x60, 0xbf};
	unsigned char	pt_p[8]
			= {0x54, 0x68, 0x65, 0x20, 0x75, 0x6e, 0x6b, 0x6e};
	unsigned char	ct_p[8]
			= {0x3e, 0xa7, 0x86, 0xf9, 0x1d, 0x76, 0xbb, 0xd3};
	unsigned char	key_p[8]
			= {0x5e, 0xd9, 0x20, 0x4f, 0xec, 0xe0, 0xb9, 0x67};

	unsigned char	iv_s[8]
			= {0x99, 0xe9, 0x7c, 0xbf, 0x4f, 0x7a, 0x6e, 0x8f};
	unsigned char	pt_s[8]
			= {0x54, 0x68, 0x65, 0x20, 0x75, 0x6e, 0x6b, 0x6e};
	unsigned char	ct_s[8]
			= {0x79, 0x45, 0x81, 0xc0, 0xa0, 0x6e, 0x40, 0xa2};

	if (practice_flag)
    sboxes_unroll_bits(p, c, k, sboxes_char2key(&iv_p[0]) ^ sboxes_char2key(&pt_p[0]), sboxes_char2key(&ct_p[0]), sboxes_char2key(&key_p[0]));
	else
    sboxes_unroll_bits(p, c, k, sboxes_char2key(&iv_s[0]) ^ sboxes_char2key(&pt_s[0]), sboxes_char2key(&ct_s[0]), sboxes_sliceoff            );
/*
	if (practice_flag)
	    unroll_bits (p, c, k, iv_p, pt_p, ct_p, key_p);
	else
	    unroll_bits (p, c, k, iv_s, pt_s, ct_s, NULL);
*/
}

/*
 * Entry point.
 */
void main(int argc, char *argv[]) {
	int		i;
	int		speed_flag = 0;
	int		practice_flag = 0;
	sboxes_bitslice	p[64], c[64], k[56];

	set_bitlength ();

	for (i=1; i<argc; i++) {
	    if (argv[i][0] != '-')
		continue;

	    if (argv[i][1] == 'S')
		speed_flag = 1;
	    else if (argv[i][1] == 'P')
		practice_flag = 1;
	}

	build_samples (p, c, k, practice_flag);

	if (speed_flag)
	    test_speed (p, c);
	else
	    keysearch (p, c, k);

	exit (0);
}
