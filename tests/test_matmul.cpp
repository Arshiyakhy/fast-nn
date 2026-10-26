#include <gtest/gtest.h>
#include <vector>
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
