#include <algorithm>
#include "matmul.h"

void matmul_naive(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matmul_loop_order(const float* A, const float* B, float* C, int m, int k, int n) {
    std::fill(C, C + m * n, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            const float a = A[i * k + p];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[p * n + j];
            }
        }
    }
}

void matmul_tiled(const float* A, const float* B, float* C, int m, int k, int n, int block) {
    block = std::max(block, 1);
    std::fill(C, C + m * n, 0.0f);
    for (int ii = 0; ii < m; ii += block) {
        const int i_end = std::min(ii + block, m);
        for (int pp = 0; pp < k; pp += block) {
            const int p_end = std::min(pp + block, k);
            for (int jj = 0; jj < n; jj += block) {
                const int j_end = std::min(jj + block, n);
                for (int i = ii; i < i_end; i++) {
                    for (int p = pp; p < p_end; p++) {
                        const float a = A[i * k + p];
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
}
