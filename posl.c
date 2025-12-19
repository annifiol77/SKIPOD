#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define SMALL_N       128
#define MEDIUM_N      256
#define LARGE_N       512
#define EXTRALARGE_N  1024

// Итерации (стартовые, быстрые)
#define SMALL_TSTEPS      20
#define MEDIUM_TSTEPS     20
#define LARGE_TSTEPS      15
#define EXTRALARGE_TSTEPS  6 // больше  слишком долго
static const float maxeps = 0.1e-7f;

// init: k самый внутренний — кэш-френдли
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

// relax: stencil радиуса 2 => считаем только 2..n-3
static void relax(int n, const float (*A)[n][n], float (*B)[n][n])
{
    const float inv_12 = 1.0f / 12.0f;

    for (int i = 2; i <= n - 3; ++i)
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

// resid: ВАЖНО — обновляем A и eps только там, где B реально вычислен (2..n-3)
static float resid(int n, float (*A)[n][n], const float (*B)[n][n])
{
    float local_eps = 0.0f;

    for (int i = 2; i <= n - 3; ++i)
        for (int j = 2; j <= n - 3; ++j)
            for (int k = 2; k <= n - 3; ++k) {
                float e = fabsf(A[i][j][k] - B[i][j][k]);
                A[i][j][k] = B[i][j][k];
                local_eps = Max(local_eps, e);
            }

    return local_eps;
}

static void verify(int n, const float (*A)[n][n])
{
    double s = 0.0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                s += (double)A[i][j][k] * (double)(i+1) * (double)(j+1) * (double)(k+1)
                     / (double)(n * n * n);
            }

    printf("S = %.10f\n", s);
}

/*
  usage:
    ./posl <dataset_id> [mode]
  dataset_id:
    1 -> SMALL
    2 -> MEDIUM
    3 -> LARGE
    4 -> EXTRALARGE
  mode:
    fixed  -> ровно itmax итераций (для сравнения -O*)
    conv   -> остановка по eps < maxeps
*/
int main(int argc, char **argv)
{
    int dataset = (argc > 1) ? atoi(argv[1]) : 1;
    const char *mode = (argc > 2) ? argv[2] : "fixed";

    int n = SMALL_N;
    int itmax = SMALL_TSTEPS;
    const char *dataset_name = "SMALL";

    switch (dataset) {
        case 1: n = SMALL_N;      itmax = SMALL_TSTEPS;      dataset_name = "SMALL";      break;
        case 2: n = MEDIUM_N;     itmax = MEDIUM_TSTEPS;     dataset_name = "MEDIUM";     break;
        case 3: n = LARGE_N;      itmax = LARGE_TSTEPS;      dataset_name = "LARGE";      break;
        case 4: n = EXTRALARGE_N; itmax = EXTRALARGE_TSTEPS; dataset_name = "EXTRALARGE"; break;
        default: n = SMALL_N;     itmax = SMALL_TSTEPS;      dataset_name = "SMALL";      break;
    }

    float (*A)[n][n] = (float (*)[n][n])malloc((size_t)n * n * n * sizeof(float));
    float (*B)[n][n] = (float (*)[n][n])malloc((size_t)n * n * n * sizeof(float));

    if (!A || !B) {
        fprintf(stderr, "Memory allocation failed for n=%d\n", n);
        free(A);
        free(B);
        return 1;
    }

    init(n, A, B);

    double t0 = omp_get_wtime();
    float eps = 0.0f;

    if (mode[0] == 'c') { // conv
        for (int it = 1; it <= itmax; ++it) {
            relax(n, A, B);
            eps = resid(n, A, B);
            if (eps < maxeps) break;
        }
    } else { // fixed
        for (int it = 1; it <= itmax; ++it) {
            relax(n, A, B);
            eps = resid(n, A, B);
        }
    }

    double t1 = omp_get_wtime();

    printf("Dataset: %s (n=%d, itmax=%d, mode=%s)\n", dataset_name, n, itmax, mode);
    printf("Final eps = %.10e\n", eps);

    // строка для парсинга скриптами
    printf("TOTAL_TIME_SEC: %.6f\n", (t1 - t0));

    verify(n, A);

    free(A);
    free(B);
    return 0;
}

