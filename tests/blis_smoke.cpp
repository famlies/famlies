// Verifies that BLIS was fetched, built, and linked: exercises the native typed
// API (the BLAS/CBLAS layers are disabled) against a hand-computed product.

#include <famlies/famlies.hpp>

#include <cmath>
#include <cstdio>

int main()
{
    std::printf("BLIS version: %s\n", famlies::blis_version());

    // Row-major 2x2: A = [1 2; 3 4], B = [5 6; 7 8], C = A*B = [19 22; 43 50].
    const double a[4] = {1.0, 2.0, 3.0, 4.0};
    const double b[4] = {5.0, 6.0, 7.0, 8.0};
    double c[4] = {0.0, 0.0, 0.0, 0.0};

    const double alpha = 1.0;
    const double beta = 0.0;

    bli_dgemm(BLIS_NO_TRANSPOSE,
              BLIS_NO_TRANSPOSE,
              2,
              2,
              2,
              &alpha,
              a,
              2,
              1, // row stride 2, column stride 1
              b,
              2,
              1,
              &beta,
              c,
              2,
              1);

    const double expected[4] = {19.0, 22.0, 43.0, 50.0};
    for (int i = 0; i < 4; ++i)
    {
        if (std::fabs(c[i] - expected[i]) > 1.0e-12)
        {
            std::printf("mismatch at %d: got %f, expected %f\n",
                        i,
                        c[i],
                        expected[i]);
            return 1;
        }
    }

    std::printf("gemm result matches: [%g %g; %g %g]\n", c[0], c[1], c[2], c[3]);
    return 0;
}
