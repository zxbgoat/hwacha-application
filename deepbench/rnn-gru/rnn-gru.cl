/* DeepBench GRU (rnn_bench "gru": cuDNN CUDNN_GRU, one layer, unidirectional, CUDNN_SKIP_INPUT so
 * every gate receives x_t directly instead of W x_t), cuDNN's formulation:
 *   r = sigm(x + R_r h + b_Wr + b_Rr)   z = sigm(x + R_z h + b_Wz + b_Rz)
 *   h' = tanh(x + r * (R_h h + b_Rh) + b_Wh)   h_t = (1 - z) * h' + z * h_{t-1}
 * R is 3*hidden x hidden (gate order r, z, h as cuDNN), bW / bR 3*hidden. One launch per time
 * step, one work-item per (batch, unit). Written for hwacha-cc. */
typedef float DATA_TYPE;
static inline DATA_TYPE sigm(DATA_TYPE v) { return 1.0f / (1.0f + exp(-v)); }
static inline DATA_TYPE tanh_(DATA_TYPE v) { return 2.0f / (1.0f + exp(-2.0f * v)) - 1.0f; }
__kernel void gru_step(__global DATA_TYPE *x, __global DATA_TYPE *R, __global DATA_TYPE *bW, __global DATA_TYPE *bR, __global DATA_TYPE *h0, __global DATA_TYPE *y, int t, int batch, int hidden)
{
	int j = get_global_id(0);
	int n = get_global_id(1);
	if ((j < hidden) && (n < batch))
	{
		__global DATA_TYPE *hprev = t == 0 ? h0 + n*hidden : y + ((t-1)*batch + n)*hidden;
		DATA_TYPE xv = x[(t*batch + n)*hidden + j];
		DATA_TYPE zr = xv + bW[0*hidden + j] + bR[0*hidden + j], zz = xv + bW[1*hidden + j] + bR[1*hidden + j], zh = bR[2*hidden + j];
		int i;
		for (i = 0; i < hidden; i++)
		{
			DATA_TYPE h = hprev[i];
			zr += R[(0*hidden + j)*hidden + i] * h;
			zz += R[(1*hidden + j)*hidden + i] * h;
			zh += R[(2*hidden + j)*hidden + i] * h;
		}
		DATA_TYPE r = sigm(zr), z = sigm(zz);
		DATA_TYPE hn = tanh_(xv + r * zh + bW[2*hidden + j]);
		y[(t*batch + n)*hidden + j] = (1.0f - z) * hn + z * hprev[j];
	}
}
