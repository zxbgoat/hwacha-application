/* DeepBench convolution (conv_bench: cudnnConvolutionForward / BackwardData / BackwardFilter on the
 * problem set (w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride), NCHW, cross-correlation as
 * cuDNN). Written for hwacha-cc, direct (im2col-free) form: one work-item per output element of
 * each pass, the reduction loop inside.
 *   x: n x c x h x w   f: k x c x r x s   y: n x k x oh x ow,  oh = (h + 2 pad_h - r) / hstride + 1 */
typedef float DATA_TYPE;
/* forward: y[n][k][oy][ox] = sum_{c,fy,fx} x[n][c][oy*hs - pad_h + fy][ox*ws - pad_w + fx] * f[k][c][fy][fx] */
__kernel void conv_fwd(__global DATA_TYPE *x, __global DATA_TYPE *f, __global DATA_TYPE *y, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int hs, int ws, int OH, int OW)
{
	int p = get_global_id(0);     /* ox + oy * OW */
	int q = get_global_id(1);     /* k + n * K */
	if ((p < OH*OW) && (q < N*K))
	{
		int ox = p % OW, oy = p / OW, k = q % K, n = q / K;
		DATA_TYPE acc = 0;
		int c, fy, fx;
		for (c = 0; c < C; c++)
			for (fy = 0; fy < R; fy++)
			{
				int iy = oy*hs - pad_h + fy;
				if (iy < 0 || iy >= H) continue;
				for (fx = 0; fx < S; fx++)
				{
					int ix = ox*ws - pad_w + fx;
					if (ix < 0 || ix >= W) continue;
					acc += x[((n*C + c)*H + iy)*W + ix] * f[((k*C + c)*R + fy)*S + fx];
				}
			}
		y[((n*K + k)*OH + oy)*OW + ox] = acc;
	}
}
/* backward data: dx[n][c][iy][ix] = sum_{k,fy,fx: oy*hs - pad_h + fy == iy, ...} dy[n][k][oy][ox] * f[k][c][fy][fx] */
__kernel void conv_bwd_data(__global DATA_TYPE *dy, __global DATA_TYPE *f, __global DATA_TYPE *dx, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int hs, int ws, int OH, int OW)
{
	int p = get_global_id(0);     /* ix + iy * W */
	int q = get_global_id(1);     /* c + n * C */
	if ((p < H*W) && (q < N*C))
	{
		int ix = p % W, iy = p / W, c = q % C, n = q / C;
		DATA_TYPE acc = 0;
		int k, fy, fx;
		for (k = 0; k < K; k++)
			for (fy = 0; fy < R; fy++)
			{
				int ty = iy + pad_h - fy;
				if (ty < 0 || ty % hs != 0) continue;
				int oy = ty / hs;
				if (oy >= OH) continue;
				for (fx = 0; fx < S; fx++)
				{
					int tx = ix + pad_w - fx;
					if (tx < 0 || tx % ws != 0) continue;
					int ox = tx / ws;
					if (ox >= OW) continue;
					acc += dy[((n*K + k)*OH + oy)*OW + ox] * f[((k*C + c)*R + fy)*S + fx];
				}
			}
		dx[((n*C + c)*H + iy)*W + ix] = acc;
	}
}
/* backward filter: df[k][c][fy][fx] = sum_{n,oy,ox} dy[n][k][oy][ox] * x[n][c][oy*hs - pad_h + fy][ox*ws - pad_w + fx] */
__kernel void conv_bwd_filter(__global DATA_TYPE *dy, __global DATA_TYPE *x, __global DATA_TYPE *df, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int hs, int ws, int OH, int OW)
{
	int p = get_global_id(0);     /* fx + fy * S */
	int q = get_global_id(1);     /* c + k * C */
	if ((p < R*S) && (q < K*C))
	{
		int fx = p % S, fy = p / S, c = q % C, k = q / C;
		DATA_TYPE acc = 0;
		int n, oy, ox;
		for (n = 0; n < N; n++)
			for (oy = 0; oy < OH; oy++)
			{
				int iy = oy*hs - pad_h + fy;
				if (iy < 0 || iy >= H) continue;
				for (ox = 0; ox < OW; ox++)
				{
					int ix = ox*ws - pad_w + fx;
					if (ix < 0 || ix >= W) continue;
					acc += dy[((n*K + k)*OH + oy)*OW + ox] * x[((n*C + c)*H + iy)*W + ix];
				}
			}
		df[((k*C + c)*R + fy)*S + fx] = acc;
	}
}
