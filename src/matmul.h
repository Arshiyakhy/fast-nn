#pragma once

// Multiplies A (m rows, k columns) by B (k rows, n columns)
// and writes the result into C (m rows, n columns).
// All three grids are flat lists of floats, stored row by row.
void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n);

// Same math as matmul_naive, but with the loops in a cache friendly order.
void matmul_loop_order(const float* A, const float* B, float* C, int m, int k, int n);

// Same math, but works on small square tiles so the data stays in the CPU's fast cache.
// block is the tile size (for example 32).
void matmul_tiled(const float* A, const float* B, float* C, int m, int k, int n, int block);

// Same loop order, with the inner loop done 8 numbers at a time using AVX2.
// Needs a CPU with AVX2 and FMA.
void matmul_avx2(const float* A, const float* B, float* C, int m, int k, int n);
