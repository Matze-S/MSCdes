
	Bitslice DES key search

	Written by Matthew Kwan - mkwan@cs.mu.oz.au


What you have here is a bitslice implementation of DES with some test
code. The code actually carries out a search for the original RSA
challenge, but that's fairly pointless now. Think of it as legacy code.

It works best on 64-bit RISC machines, but can usually do a pretty good
job on other machines if you get the compile options right.

In its current state, you can probably get a few percent speedup by
hardwiring the plaintext and ciphertext values, using Gray code to step
through key values, and maybe by pre-computing the first round every 256
iterations. Also, if you are using a 64-bit machine, you can check all
64 combinations of the six bits entering the same S-box in round 1, and
thus pre-compute its output. But like I said, it's only a few percent.


Usage: a.out [-S] [-P]

-S	Carry out a speed test. Otherwise begin an exhaustive search.

-P	Use the RSA practice DES value. Otherwise use the RSA secret
	key challenge values.


mkwan
