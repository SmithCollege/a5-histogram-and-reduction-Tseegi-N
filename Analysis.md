**Reduction**: Reduction in GPU optimize around 100000. Up until that point, CPU reduction seems to be doing a good job. Even though GPU reduction with and without divergence are awfully close to eachother, the difference becomes more noticable with bigger numbers as more threads are being efficiently used.

**Histogram**: In terms of optimization, GPU basic is better than GPU with strides, which is better than CPU histograms. GPU with strides are more computationally heavy but input arrays are not coalesced.
