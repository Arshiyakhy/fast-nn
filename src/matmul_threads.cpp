#include "matmul.h"

#include <algorithm>
#include <thread>
#include <vector>

void matmul_threads(const float* A, const float* B, float* C, int m, int k, int n, int num_threads) {
    num_threads = std::max(1, std::min(num_threads, m));
    const int base = m / num_threads;
    const int extra = m % num_threads;  // the first 'extra' workers get one more row

    std::vector<std::thread> workers;
    int row = 0;
    for (int t = 0; t < num_threads; t++) {
        const int rows = base + (t < extra ? 1 : 0);
        const float* A_part = A + row * k;
        float* C_part = C + row * n;
        if (t == num_threads - 1) {
            matmul_avx2(A_part, B, C_part, rows, k, n);  // last block runs on this thread
        } else {
            workers.emplace_back(matmul_avx2, A_part, B, C_part, rows, k, n);
        }
        row += rows;
    }
    for (auto& w : workers) {
        w.join();
    }
}
