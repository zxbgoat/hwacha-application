/* DeepBench LSTM (rnn_bench "lstm": cuDNN CUDNN_LSTM, one layer, unidirectional, CUDNN_SKIP_INPUT
 * so every gate receives x_t directly instead of W x_t):
 *   i = sigm(x + R_i h + b_i)   f = sigm(x + R_f h + b_f)   o = sigm(x + R_o h + b_o)   g = tanh(x + R_g h + b_g)
 *   c_t = f * c_{t-1} + i * g    h_t = o * tanh(c_t)
 * R is 4*hidden x hidden (gate order i, f, o, g as cuDNN), b 4*hidden. One launch per time step,
 * one work-item per (batch, unit) computing its four gate dot products; y / cs hold h_t / c_t for
 * every t. sigm / tanh via exp (hwacha-cc's exp expansion). Written for hwacha-cc. */
typedef float DATA_TYPE;
static inline DATA_TYPE sigm(DATA_TYPE z) { return 1.0f / (1.0f + exp(-z)); }
static inline DATA_TYPE tanh_(DATA_TYPE z) { return 2.0f / (1.0f + exp(-2.0f * z)) - 1.0f; }
__kernel void lstm_step(__global DATA_TYPE *x, __global DATA_TYPE *R, __global DATA_TYPE *b, __global DATA_TYPE *h0, __global DATA_TYPE *c0, __global DATA_TYPE *y, __global DATA_TYPE *cs, int t, int batch, int hidden)
{
	int j = get_global_id(0);
	int n = get_global_id(1);
	if ((j < hidden) && (n < batch))
	{
		__global DATA_TYPE *hprev = t == 0 ? h0 + n*hidden : y + ((t-1)*batch + n)*hidden;
		DATA_TYPE cprev = t == 0 ? c0[n*hidden + j] : cs[((t-1)*batch + n)*hidden + j];
		DATA_TYPE xv = x[(t*batch + n)*hidden + j];
		DATA_TYPE zi = xv + b[0*hidden + j], zf = xv + b[1*hidden + j], zo = xv + b[2*hidden + j], zg = xv + b[3*hidden + j];
		int i;
		for (i = 0; i < hidden; i++)
		{
			DATA_TYPE h = hprev[i];
			zi += R[(0*hidden + j)*hidden + i] * h;
			zf += R[(1*hidden + j)*hidden + i] * h;
			zo += R[(2*hidden + j)*hidden + i] * h;
			zg += R[(3*hidden + j)*hidden + i] * h;
		}
		DATA_TYPE ig = sigm(zi), fg = sigm(zf), og = sigm(zo), gg = tanh_(zg);
		DATA_TYPE c = fg * cprev + ig * gg;
		cs[(t*batch + n)*hidden + j] = c;
		y[(t*batch + n)*hidden + j] = og * tanh_(c);
	}
}
