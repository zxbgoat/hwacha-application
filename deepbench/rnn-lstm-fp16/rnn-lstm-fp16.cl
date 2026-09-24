/* DeepBench LSTM in half precision (rnn_bench "lstm" with the "half" precision: CUDNN_DATA_HALF x, R,
 * b, h, c with float compute). cuDNN CUDNN_LSTM, one layer, unidirectional, CUDNN_SKIP_INPUT
 * so every gate receives x_t directly instead of W x_t):
 *   i = sigm(x + R_i h + b_i)   f = sigm(x + R_f h + b_f)   o = sigm(x + R_o h + b_o)   g = tanh(x + R_g h + b_g)
 *   c_t = f * c_{t-1} + i * g    h_t = o * tanh(c_t)
 * R is 4*hidden x hidden (gate order i, f, o, g as cuDNN), b 4*hidden. One launch per time step,
 * one work-item per (batch, unit) computing its four gate dot products; y / cs hold h_t / c_t for
 * every t as halves (h_{t-1} / c_{t-1} are read back rounded, as cuDNN does with half state).
 * Gate math in float (vlxh + vfcvt.s.h on load, vfcvt.h.s on store). Written for hwacha-cc. */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef float DATA_TYPE;
static inline DATA_TYPE sigm(DATA_TYPE z) { return 1.0f / (1.0f + exp(-z)); }
static inline DATA_TYPE tanh_(DATA_TYPE z) { return 2.0f / (1.0f + exp(-2.0f * z)) - 1.0f; }
__kernel void lstm_step(__global half *x, __global half *R, __global half *b, __global half *h0, __global half *c0, __global half *y, __global half *cs, int t, int batch, int hidden)
{
	int j = get_global_id(0);
	int n = get_global_id(1);
	if ((j < hidden) && (n < batch))
	{
		__global half *hprev = t == 0 ? h0 + n*hidden : y + ((t-1)*batch + n)*hidden;
		DATA_TYPE cprev = t == 0 ? (float)c0[n*hidden + j] : (float)cs[((t-1)*batch + n)*hidden + j];
		DATA_TYPE xv = (float)x[(t*batch + n)*hidden + j];
		DATA_TYPE zi = xv + (float)b[0*hidden + j], zf = xv + (float)b[1*hidden + j], zo = xv + (float)b[2*hidden + j], zg = xv + (float)b[3*hidden + j];
		int i;
		for (i = 0; i < hidden; i++)
		{
			DATA_TYPE h = (float)hprev[i];
			zi += (float)R[(0*hidden + j)*hidden + i] * h;
			zf += (float)R[(1*hidden + j)*hidden + i] * h;
			zo += (float)R[(2*hidden + j)*hidden + i] * h;
			zg += (float)R[(3*hidden + j)*hidden + i] * h;
		}
		DATA_TYPE ig = sigm(zi), fg = sigm(zf), og = sigm(zo), gg = tanh_(zg);
		DATA_TYPE c = fg * cprev + ig * gg;
		cs[(t*batch + n)*hidden + j] = (half)c;
		y[(t*batch + n)*hidden + j] = (half)(og * tanh_(c));
	}
}
