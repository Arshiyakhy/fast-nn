#pragma once

// Multiplies A (m rows, k columns) by B (k rows, n columns)
// and writes the result into C (m rows, n columns).
// All three grids are flat lists of floats, stored row by row.
void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n);
