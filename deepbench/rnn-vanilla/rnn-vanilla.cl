/* DeepBench vanilla RNN (rnn_bench "vanilla": cuDNN CUDNN_RNN_RELU, one layer, unidirectional,
 * CUDNN_SKIP_INPUT so the input is fed straight to the cell without an input weight matrix):
 *   h_t = ReLU(x_t + R h_{t-1} + b)      x_t, h_t: batch x hidden, R: hidden x hidden
 * One launch per time step, one work-item per (batch, unit); y holds h_t for every t (the work-items
 * read h_{t-1} from y[t-1], so no in-place hazard). Written for hwacha-cc. */
typedef float DATA_TYPE;
__kernel void rnn_relu_step(__global DATA_TYPE *x, __global DATA_TYPE *R, __global DATA_TYPE *b, __global DATA_TYPE *h0, __global DATA_TYPE *y, int t, int batch, int hidden)
{
	int j = get_global_id(0);   /* hidden unit */
	int n = get_global_id(1);   /* batch row */
	if ((j < hidden) && (n < batch))
	{
		__global DATA_TYPE *hprev = t == 0 ? h0 + n*hidden : y + ((t-1)*batch + n)*hidden;
		DATA_TYPE acc = x[(t*batch + n)*hidden + j] + b[j];
		int i;
		for (i = 0; i < hidden; i++)
			acc += R[j*hidden + i] * hprev[i];
		y[(t*batch + n)*hidden + j] = acc > 0 ? acc : 0;
	}
}
