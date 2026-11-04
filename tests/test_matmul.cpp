#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "matmul.h"

TEST(MatmulNaive, TwoByTwo) {
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(4, 0.0f);
    matmul_naive(A.data(), B.data(), C.data(), 2, 2, 2);
    EXPECT_FLOAT_EQ(C[0], 19);
    EXPECT_FLOAT_EQ(C[1], 22);
    EXPECT_FLOAT_EQ(C[2], 43);
    EXPECT_FLOAT_EQ(C[3], 50);
}

TEST(MatmulNaive, TwoByThreeTimesThreeByTwo) {
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    std::vector<float> C(4, 0.0f);
    matmul_naive(A.data(), B.data(), C.data(), 2, 3, 2);
    EXPECT_FLOAT_EQ(C[0], 58);
    EXPECT_FLOAT_EQ(C[1], 64);
    EXPECT_FLOAT_EQ(C[2], 139);
    EXPECT_FLOAT_EQ(C[3], 154);
}

TEST(MatmulNaive, IdentityLeavesMatrixUnchanged) {
    std::vector<float> I = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> B = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> C(9, 0.0f);
    matmul_naive(I.data(), B.data(), C.data(), 3, 3, 3);
    for (int x = 0; x < 9; x++) {
        EXPECT_FLOAT_EQ(C[x], B[x]);
    }
}

TEST(MatmulNaive, RowTimesColumn) {
    std::vector<float> A = {1, 2, 3};
    std::vector<float> B = {4, 5, 6};
    std::vector<float> C(1, 0.0f);
    matmul_naive(A.data(), B.data(), C.data(), 1, 3, 1);
    EXPECT_FLOAT_EQ(C[0], 32);
}

TEST(MatmulNaive, ColumnTimesRow) {
    std::vector<float> A = {1, 2, 3};
    std::vector<float> B = {4, 5};
    std::vector<float> C(6, 0.0f);
    matmul_naive(A.data(), B.data(), C.data(), 3, 1, 2);
    float expected[6] = {4, 5, 8, 10, 12, 15};
    for (int x = 0; x < 6; x++) {
        EXPECT_FLOAT_EQ(C[x], expected[x]);
    }
}

static void check_matches_naive(int m, int k, int n) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(m * k), B(k * n), expected(m * n), actual(m * n, 99.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);
    matmul_naive(A.data(), B.data(), expected.data(), m, k, n);
    matmul_loop_order(A.data(), B.data(), actual.data(), m, k, n);
    for (int x = 0; x < m * n; x++) {
        EXPECT_NEAR(actual[x], expected[x], 1e-4f);
    }
}

TEST(MatmulLoopOrder, MatchesNaiveOnManyShapes) {
    check_matches_naive(2, 2, 2);
    check_matches_naive(2, 3, 2);
    check_matches_naive(37, 53, 29);
    check_matches_naive(64, 64, 64);
    check_matches_naive(100, 1, 7);
}

static void check_tiled_matches_naive(int m, int k, int n, int block) {
    std::mt19937 gen(321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(m * k), B(k * n), expected(m * n), actual(m * n, 99.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);
    matmul_naive(A.data(), B.data(), expected.data(), m, k, n);
    matmul_tiled(A.data(), B.data(), actual.data(), m, k, n, block);
    for (int x = 0; x < m * n; x++) {
        EXPECT_NEAR(actual[x], expected[x], 1e-4f);
    }
}

TEST(MatmulTiled, MatchesNaiveOnManyShapesAndBlocks) {
    check_tiled_matches_naive(2, 2, 2, 4);
    check_tiled_matches_naive(2, 3, 2, 1);
    check_tiled_matches_naive(37, 53, 29, 8);
    check_tiled_matches_naive(64, 64, 64, 16);
    check_tiled_matches_naive(100, 1, 7, 32);
    check_tiled_matches_naive(70, 70, 70, 32);
}
