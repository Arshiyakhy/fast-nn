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

static void BM_MatmulLoopOrder(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    for (auto _ : state) {
        matmul_loop_order(A.data(), B.data(), C.data(), n, n, n);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 2LL * n * n * n);
}
BENCHMARK(BM_MatmulLoopOrder)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);

static void BM_MatmulTiled(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int block = static_cast<int>(state.range(1));
    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    for (auto _ : state) {
        matmul_tiled(A.data(), B.data(), C.data(), n, n, n, block);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 2LL * n * n * n);
}
BENCHMARK(BM_MatmulTiled)
    ->Args({256, 32})->Args({512, 32})->Args({1024, 32})
    ->Args({1024, 8})->Args({1024, 16})->Args({1024, 64})->Args({1024, 128})
    ->Unit(benchmark::kMillisecond);

static void BM_MatmulAvx2(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    for (auto _ : state) {
        matmul_avx2(A.data(), B.data(), C.data(), n, n, n);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 2LL * n * n * n);
}
BENCHMARK(BM_MatmulAvx2)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);

static void BM_MatmulThreads(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int threads = static_cast<int>(state.range(1));
    std::vector<float> A(n * n), B(n * n), C(n * n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    for (auto _ : state) {
        matmul_threads(A.data(), B.data(), C.data(), n, n, n, threads);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 2LL * n * n * n);
}
BENCHMARK(BM_MatmulThreads)
    ->Args({1024, 1})->Args({1024, 2})->Args({1024, 4})->Args({1024, 6})->Args({1024, 8})
    ->Args({512, 8})->Args({256, 8})
    ->UseRealTime()->Unit(benchmark::kMillisecond);
