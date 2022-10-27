#include "osqp/osqp.h"
#include <math.h>

c_int to_sparse(double* M, size_t M_rows, size_t M_cols, c_float** _P_x, c_int** _P_i, c_int** _P_p, char ut) {

    c_float* P_x = (c_float*)malloc(M_cols * M_rows * sizeof (c_float));
    c_int* P_i = (c_int*)malloc(M_cols * M_rows * sizeof (c_int));
    c_int* P_p = (c_int*)malloc((M_cols + 1) * sizeof (c_int));

    size_t pxi = 0;

    if (!ut) {
        for (size_t col = 0; col < M_cols; ++col) {
            P_p[col] = pxi;
            for (size_t row = 0; row < M_rows; ++row) {
                c_float value = M[M_cols * row + col];
                if (fabs(value) > 0.0001) {
                    P_x[pxi] = value;
                    P_i[pxi] = row;
                    ++pxi;
                }
            }
        }
    } else {
        for (size_t col = 0; col < M_cols; ++col) {
            P_p[col] = pxi;
            for (size_t row = 0; row <= col; ++row) {
                c_float value = M[M_cols * row + col];
                if (fabs(value) > 0.0001) {
                    P_x[pxi] = value;
                    P_i[pxi] = row;
                    ++pxi;
                }
            }
        }
    }

    P_p[M_cols] = pxi;

    *_P_x = (c_float*)realloc(P_x, pxi * sizeof (c_float));
    *_P_i = (c_int*)realloc(P_i, pxi * sizeof (c_int));
    *_P_p = P_p;

    return pxi;

}

int qp_solve(double* solution, double* H, double* _f, double* A, double* lb, double* ub, size_t n_vars, size_t n_constraints) {

    c_float *P_x;
    c_int *P_i, *P_p;
    c_int P_nnz  = to_sparse(H, n_vars, n_vars, &P_x, &P_i, &P_p, TRUE);

    c_float *A_x;
    c_int *A_i, *A_p;
    c_int A_nnz  = to_sparse(A, n_constraints, n_vars, &A_x, &A_i, &A_p, FALSE);

    c_float* q = _f;
    c_float* l = lb;
    c_float* u = ub;

    c_int exitflag = 0;

    (OSQPData *)c_malloc(sizeof(OSQPData));

    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings  *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData      *data     = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate data
    if (data) {
        data->n = n_vars;
        data->m = n_constraints;
        data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
        data->q = q;
        data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
        data->l = l;
        data->u = u;
    }

    // Define solver settings as default
    if (settings) {
        osqp_set_default_settings(settings);
        settings->verbose = FALSE;
    }


    // Setup workspace
    exitflag = osqp_setup(&work, data, settings);

    // Solve Problem
    exitflag = osqp_solve(work);

    memcpy(solution, work->solution->x, n_vars * sizeof (double));

    // Clean workspace

    free(P_x);
    free(P_p);
    free(P_i);

    free(A_x);
    free(A_p);
    free(A_i);


    osqp_cleanup(work);
    if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    if (settings)  c_free(settings);

    return 1;
}