// Warp-level reduction primitives for optimized atomic operations
//
// These functions reduce contention on atomic operations by having
// each warp perform a parallel reduction, then only lane 0 performs
// a single atomic add to global memory.

__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__
double warp_reduce_sum_double(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
