#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include "matmul.h"

static void BM_MatmulNaive(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    for (auto _ : state) {
        matmul_naive(A.data(), B.data(), C.data(), n, n, n);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 2LL * n * n * n);
}
BENCHMARK(BM_MatmulNaive)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);
