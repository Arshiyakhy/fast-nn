#include "matmul.h"

#include <immintrin.h>

#include <algorithm>

void matmul_avx2(const float* A, const float* B, float* C, int m, int k, int n) {
    std::fill(C, C + m * n, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            const float a = A[i * k + p];
            const __m256 va = _mm256_set1_ps(a);
            const float* b_row = B + p * n;
            float* c_row = C + i * n;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vb = _mm256_loadu_ps(b_row + j);
                __m256 vc = _mm256_loadu_ps(c_row + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(c_row + j, vc);
            }
            for (; j < n; j++) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}
