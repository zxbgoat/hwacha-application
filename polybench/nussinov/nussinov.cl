/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). Integer data, as 4.2.1. */
typedef int DATA_TYPE;
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)
/* nussinov: RNA folding score table. table[i][j] depends only on entries with a smaller j - i, so
 * the host walks the diagonals d = j - i = 1 .. n-1 and one work-item handles every (i, i + d);
 * each entry performs the same sequence of max operations as the 4.2.1 loop nest. */
__kernel void nussinov_kernel(__global char *seq, __global DATA_TYPE *table, int d, int n)
{
	int i = get_global_id(0);
	int j = i + d;
	if (j < n)
	{
		int k;
		DATA_TYPE t = table[i*n + j];
		if (j-1 >= 0)
			t = max_score(t, table[i*n + (j-1)]);
		if (i+1 < n)
			t = max_score(t, table[(i+1)*n + j]);
		if (j-1 >= 0 && i+1 < n)
		{
			if (i < j-1)
				t = max_score(t, table[(i+1)*n + (j-1)] + match(seq[i], seq[j]));
			else
				t = max_score(t, table[(i+1)*n + (j-1)]);
		}
		for (k = i+1; k < j; k++)
			t = max_score(t, table[i*n + k] + table[(k+1)*n + j]);
		table[i*n + j] = t;
	}
}
