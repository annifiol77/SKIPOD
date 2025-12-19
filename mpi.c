#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define Max(a,b) ((a)>(b)?(a):(b))

// --- Датасеты (как ты использовала в posl/for/task) ---
#define SMALL_N       128
#define MEDIUM_N      256
#define LARGE_N       512
#define EXTRALARGE_N  1024

// --- Число итераций для fixed-режима (можешь синхронизировать с posl.c) ---
#define SMALL_TSTEPS       20
#define MEDIUM_TSTEPS      20
#define LARGE_TSTEPS       15
#define EXTRALARGE_TSTEPS  10

static const float maxeps = 0.1e-7f;

// Удобная печать только на rank 0
static void p0(int rank, const char *fmt, ...)
{
    if (rank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

// Инициализация: каждый MPI-процесс заполняет только свой диапазон i
// Глобальные индексы i_global = start_i .. end_i
// Локальная раскладка: local i = 0..local_n-1 храним в A[H+i] где H=2 (halo)
static void init_local(int n, int start_i, int local_n,
                       float *A, float *B)
{
    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n; // j*k
    // Обнуляем всё, включая halo
    memset(A, 0, (size_t)(local_n + 2*H) * plane * sizeof(float));
    memset(B, 0, (size_t)(local_n + 2*H) * plane * sizeof(float));

    for (int li = 0; li < local_n; ++li) {
        int i = start_i + li;
        float *Ap = A + (size_t)(H + li) * plane;

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1)
                    Ap[(size_t)j * n + k] = 0.0f;
                else
                    Ap[(size_t)j * n + k] = 4.0f + (float)i + (float)j + (float)k;
            }
        }
    }
}

// Обмен halo толщиной 2 с соседями по i (левый/правый)
// Отправляем/получаем 2 плоскости j*k
static void exchange_halo(int n, int rank, int size,
                          int local_n, float *A)
{
    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n;
    const int left  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Указатели на 2 "реальных" первых/последних слоёв
    float *send_left  = A + (size_t)H * plane;                     // слой i_local=0 (в A[H])
    float *send_right = A + (size_t)(H + local_n - 2) * plane;     // слой i_local=local_n-2 (первый из последних двух)

    // Указатели на halo области (2 слоя слева и 2 справа)
    float *recv_left  = A + (size_t)0 * plane;                     // A[0], A[1]
    float *recv_right = A + (size_t)(H + local_n) * plane;         // A[H+local_n], A[H+local_n+1]

    // Обмен делаем через Sendrecv: безопасно и просто
    // 1) Обмен "двух первых слоёв" -> левому, получаем "две правые halo" от правого
    MPI_Sendrecv(send_left,  (int)(2 * plane), MPI_FLOAT, left,  100,
                 recv_right, (int)(2 * plane), MPI_FLOAT, right, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 2) Обмен "двух последних слоёв" -> правому, получаем "две левые halo" от левого
    MPI_Sendrecv(send_right, (int)(2 * plane), MPI_FLOAT, right, 200,
                 recv_left,  (int)(2 * plane), MPI_FLOAT, left,  200,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Jacobi-relax: считаем B по A (используем halo)
// Важно: li от 0..local_n-1 соответствует глобальному i=start_i+li
static void relax_local(int n, int start_i, int local_n,
                        const float *A, float *B)
{
    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n;
    const float inv_12 = 1.0f / 12.0f;

    for (int li = 0; li < local_n; ++li) {
        int i = start_i + li;
        // Глобальные границы по i не обновляем
        if (i < 2 || i > n - 3) continue;

        const float *Am = A + (size_t)(H + li) * plane; // центр
        float       *Bp = B + (size_t)(H + li) * plane;

        for (int j = 2; j <= n - 3; ++j) {
            for (int k = 2; k <= n - 3; ++k) {
                // Индекс в плоскости
                size_t idx = (size_t)j * n + k;

                // Соседи по j/k в той же плоскости
                float s =
                    Am[idx - 1] + Am[idx + 1] +
                    Am[idx - n] + Am[idx + n] +
                    Am[idx - 2] + Am[idx + 2] +
                    Am[idx - 2 * n] + Am[idx + 2 * n];

                // Соседи по i: берем из соседних плоскостей (через halo)
                const float *A_im1 = A + (size_t)(H + li - 1) * plane;
                const float *A_ip1 = A + (size_t)(H + li + 1) * plane;
                const float *A_im2 = A + (size_t)(H + li - 2) * plane;
                const float *A_ip2 = A + (size_t)(H + li + 2) * plane;

                s += A_im1[idx] + A_ip1[idx] + A_im2[idx] + A_ip2[idx];

                Bp[idx] = s * inv_12;
            }
        }
    }
}

// resid: A = B, eps = max|A-B| на локальном блоке (без глобальных границ)
static float resid_local(int n, int start_i, int local_n,
                         float *A, const float *B)
{
    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n;
    float local_eps = 0.0f;

    for (int li = 0; li < local_n; ++li) {
        int i = start_i + li;
        if (i == 0 || i == n - 1) continue;

        float *Ap = A + (size_t)(H + li) * plane;
        const float *Bp = B + (size_t)(H + li) * plane;

        for (int j = 1; j <= n - 2; ++j) {
            for (int k = 1; k <= n - 2; ++k) {
                size_t idx = (size_t)j * n + k;
                float e = fabsf(Ap[idx] - Bp[idx]);
                Ap[idx] = Bp[idx];
                local_eps = Max(local_eps, e);
            }
        }
    }
    return local_eps;
}

// verify: считаем сумму по локальному блоку и делаем Reduce(SUM)
static double verify_local(int n, int start_i, int local_n,
                           const float *A)
{
    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n;
    double s = 0.0;
    const double denom = (double)n * (double)n * (double)n;

    for (int li = 0; li < local_n; ++li) {
        int i = start_i + li;
        const float *Ap = A + (size_t)(H + li) * plane;
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                s += (double)Ap[(size_t)j * n + k] *
                     (double)(i + 1) * (double)(j + 1) * (double)(k + 1) / denom;
            }
        }
    }
    return s;
}

// Разбиение i по процессам: блочное, с “хвостом” на первые ranks
static void split_1d(int n, int rank, int size, int *start_i, int *local_n)
{
    int base = n / size;
    int rem  = n % size;

    *local_n = base + (rank < rem ? 1 : 0);
    *start_i = rank * base + (rank < rem ? rank : rem);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dataset = 1;
    const char *mode = "fixed";
    if (argc > 1) dataset = atoi(argv[1]);
    if (argc > 2) mode = argv[2];

    int n = SMALL_N, itmax = SMALL_TSTEPS;
    const char *dataset_name = "SMALL";

    switch (dataset) {
        case 1: n = SMALL_N;      itmax = SMALL_TSTEPS;      dataset_name = "SMALL";      break;
        case 2: n = MEDIUM_N;     itmax = MEDIUM_TSTEPS;     dataset_name = "MEDIUM";     break;
        case 3: n = LARGE_N;      itmax = LARGE_TSTEPS;      dataset_name = "LARGE";      break;
        case 4: n = EXTRALARGE_N; itmax = EXTRALARGE_TSTEPS; dataset_name = "EXTRALARGE"; break;
        default: n = SMALL_N;     itmax = SMALL_TSTEPS;      dataset_name = "SMALL";      break;
    }

    int start_i, local_n;
    split_1d(n, rank, size, &start_i, &local_n);

    const int H = 2;
    const size_t plane = (size_t)n * (size_t)n;
    const size_t total_planes = (size_t)(local_n + 2*H);

    float *A = (float*)malloc(total_planes * plane * sizeof(float));
    float *B = (float*)malloc(total_planes * plane * sizeof(float));
    if (!A || !B) {
        if (rank == 0) fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    init_local(n, start_i, local_n, A, B);

    if (rank == 0) {
        printf("MPI Version (1D i-decomposition, halo=2)\n");
        printf("Dataset: %s (n=%d, itmax=%d, mode=%s)\n", dataset_name, n, itmax, mode);
        printf("MPI size = %d\n", size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    float eps = 0.0f;
    for (int it = 1; it <= itmax; ++it) {
        // 1) Обновить halo для A
        exchange_halo(n, rank, size, local_n, A);

        // 2) Посчитать B из A
        relax_local(n, start_i, local_n, A, B);

        // 3) eps и копирование A=B (локально)
        float local_eps = resid_local(n, start_i, local_n, A, B);

        // 4) глобальный eps = max по всем процессам
        MPI_Allreduce(&local_eps, &eps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0 && (it == 1 || it == itmax || (it % 5 == 0))) {
            printf("it=%4d   eps=%e\n", it, (double)eps);
        }

        // В режиме не-fixed можно было бы сделать ранний выход по eps
        // Но для честных замеров лучше fixed.
        if (strcmp(mode, "fixed") != 0) {
            if (eps < maxeps) break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Final eps = %e\n", (double)eps);
        printf("TOTAL_TIME_SEC: %.6f\n", t1 - t0);
    }

    double local_s = verify_local(n, start_i, local_n, A);
    double global_s = 0.0;
    MPI_Reduce(&local_s, &global_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("S = %.10f\n", global_s);
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}

