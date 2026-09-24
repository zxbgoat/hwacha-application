/* DeepBench vanilla RNN in half precision (rnn_bench "vanilla" with the "half" precision: CUDNN_DATA_HALF
 * x, R, b, h with float compute). cuDNN CUDNN_RNN_RELU, one layer, unidirectional,
 * CUDNN_SKIP_INPUT so the input is fed straight to the cell without an input weight matrix):
 *   h_t = ReLU(x_t + R h_{t-1} + b)      x_t, h_t: batch x hidden, R: hidden x hidden
 * One launch per time step, one work-item per (batch, unit); y holds h_t for every t (the work-items
 * read h_{t-1} from y[t-1], so no in-place hazard) as halves: the state is quantised every step, as
 * cuDNN's half state. Math in float (vlxh + vfcvt.s.h on load, vfcvt.h.s on store). Written for hwacha-cc. */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef float DATA_TYPE;
__kernel void rnn_relu_step(__global half *x, __global half *R, __global half *b, __global half *h0, __global half *y, int t, int batch, int hidden)
{
	int j = get_global_id(0);   /* hidden unit */
	int n = get_global_id(1);   /* batch row */
	if ((j < hidden) && (n < batch))
	{
		__global half *hprev = t == 0 ? h0 + n*hidden : y + ((t-1)*batch + n)*hidden;
		DATA_TYPE acc = (float)x[(t*batch + n)*hidden + j] + (float)b[j];
		int i;
		for (i = 0; i < hidden; i++)
			acc += (float)R[j*hidden + i] * (float)hprev[i];
		y[(t*batch + n)*hidden + j] = (half)(acc > 0 ? acc : 0);
	}
}
