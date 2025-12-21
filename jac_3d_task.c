#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define SMALL_N       128
#define MEDIUM_N      256
#define LARGE_N       512
#define EXTRALARGE_N  1024

#define SMALL_TSTEPS      20
#define MEDIUM_TSTEPS     40
#define LARGE_TSTEPS      20
#define EXTRALARGE_TSTEPS  10

static const float maxeps = 0.1e-7f;

static void init(int n, float (*A)[n][n], float (*B)[n][n])
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                if (i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1)
                    A[i][j][k] = 0.0f;
                else
                    A[i][j][k] = 4.0f + (float)i + (float)j + (float)k;

                B[i][j][k] = 0.0f;
            }
}

static void verify(int n, const float (*A)[n][n])
{
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                s += (double)A[i][j][k] * (double)(i+1) * (double)(j+1) * (double)(k+1)
                     / (double)(n * n * n);

    printf("S = %.10f\n", s);
}

static int choose_block_i(int n, int threads)
{
    int inner = n - 4; // обл i=2..n-3 имеет длину n-4
    int target_tasks = 8 * threads;
    int bi = (inner + target_tasks - 1) / target_tasks; 
    if (bi < 2) bi = 2;
    return bi;
}

static double kernel_jacobi_3d_task(int n, int itmax)
{
    float (*A)[n][n] = (float (*)[n][n])malloc((size_t)n * n * n * sizeof(float));
    float (*B)[n][n] = (float (*)[n][n])malloc((size_t)n * n * n * sizeof(float));
    if (!A || !B) {
        fprintf(stderr, "Memory allocation failed for n=%d\n", n);
        free(A); free(B);
        return -1.0;
    }

    init(n, A, B);

    const float inv_12 = 1.0f / 12.0f;
    float eps = 0.0f;

    double t0 = omp_get_wtime();

    // 1 parallel-регион
    #pragma omp parallel default(none) shared(A,B,n,itmax,inv_12,eps)
    {
        int threads = omp_get_num_threads();
        int bi = choose_block_i(n, threads);

        for (int it = 1; it <= itmax; ++it) {

            #pragma omp single
            eps = 0.0f;

            // задачи по блокам i 
            #pragma omp single
            {
                for (int i0 = 2; i0 <= n - 3; i0 += bi) {
                    int i1 = i0 + bi - 1;
                    if (i1 > n - 3) i1 = n - 3;

                    #pragma omp task firstprivate(i0,i1) shared(A,B,n,inv_12)
                    {
                        for (int i = i0; i <= i1; ++i)
                            for (int j = 2; j <= n - 3; ++j)
                                for (int k = 2; k <= n - 3; ++k) {
                                    float sum =
                                        A[i-1][j][k] + A[i+1][j][k] +
                                        A[i][j-1][k] + A[i][j+1][k] +
                                        A[i][j][k-1] + A[i][j][k+1] +
                                        A[i-2][j][k] + A[i+2][j][k] +
                                        A[i][j-2][k] + A[i][j+2][k] +
                                        A[i][j][k-2] + A[i][j][k+2];

                                    B[i][j][k] = sum * inv_12;
                                }
                    }
                }
                #pragma omp taskwait
            }

            // задачи по блокам i 
            #pragma omp single
            {
                for (int i0 = 2; i0 <= n - 3; i0 += bi) {
                    int i1 = i0 + bi - 1;
                    if (i1 > n - 3) i1 = n - 3;

                    #pragma omp task firstprivate(i0,i1) shared(A,B,n,eps)
                    {
                        float local_eps = 0.0f;

                        for (int i = i0; i <= i1; ++i)
                            for (int j = 2; j <= n - 3; ++j)
                                for (int k = 2; k <= n - 3; ++k) {
                                    float e = fabsf(A[i][j][k] - B[i][j][k]);
                                    A[i][j][k] = B[i][j][k];
                                    local_eps = Max(local_eps, e);
                                }

                        // одна критсекция на задач
                        #pragma omp critical
                        {
                            eps = Max(eps, local_eps);
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    double t1 = omp_get_wtime();

    printf("Final eps = %.10e\n", eps);
    printf("TOTAL_TIME_SEC: %.6f\n", (t1 - t0));
    verify(n, A);

    free(A);
    free(B);
    return (t1 - t0);
}

int main(int argc, char **argv)
{
    int dataset = (argc > 1) ? atoi(argv[1]) : 1;

    int n = SMALL_N, tsteps = SMALL_TSTEPS;
    const char *dataset_name = "SMALL";

    switch (dataset) {
        case 1: n = SMALL_N;      tsteps = SMALL_TSTEPS;      dataset_name = "SMALL";      break;
        case 2: n = MEDIUM_N;     tsteps = MEDIUM_TSTEPS;     dataset_name = "MEDIUM";     break;
        case 3: n = LARGE_N;      tsteps = LARGE_TSTEPS;      dataset_name = "LARGE";      break;
        case 4: n = EXTRALARGE_N; tsteps = EXTRALARGE_TSTEPS; dataset_name = "EXTRALARGE"; break;
        default: break;
    }

    printf("OpenMP TASK Version\n");
    printf("Dataset: %s (n=%d, tsteps=%d)\n", dataset_name, n, tsteps);
    printf("OMP max threads = %d\n", omp_get_max_threads());

    double time = kernel_jacobi_3d_task(n, tsteps);
    (void)time;

    return 0;
}

