/*
 * _gaussfit.c -- Bounded Levenberg-Marquardt Gaussian fitting for phspectra.
 *
 * Moves the model function + optimization loop into C so the entire fit
 * runs with zero Python callbacks.  Supports up to MAX_PARAMS parameters
 * (12 Gaussians = 36 params).
 *
 * License: MIT (same as phspectra)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <string.h>

/* Maximum number of parameters (32 components * 3 params each). */
#define MAX_PARAMS 96
/* Maximum data points we stack-allocate for.  Heap fallback above this. */
#define STACK_N 1024

/* -----------------------------------------------------------------------
 * 1. Multi-Gaussian model evaluation
 * ----------------------------------------------------------------------- */

static void
eval_multi_gaussian(const double *x, int n, const double *params, int m,
                    double *out)
{
    int i, k;
    double a, mu, sig, z;

    memset(out, 0, (size_t)n * sizeof(double));
    for (k = 0; k < m; k += 3) {
        a   = params[k];
        mu  = params[k + 1];
        sig = params[k + 2];
        if (sig <= 0.0) sig = 1e-30;
        for (i = 0; i < n; i++) {
            z = (x[i] - mu) / sig;
            out[i] += a * exp(-0.5 * z * z);
        }
    }
}

/* -----------------------------------------------------------------------
 * 2. Analytic Jacobian  (n x m matrix, row-major)
 * ----------------------------------------------------------------------- */

static void
eval_jacobian(const double *x, int n, const double *params, int m,
              double *jac)
{
    int i, k, col;
    double a, mu, sig, z, g;

    memset(jac, 0, (size_t)n * (size_t)m * sizeof(double));
    for (k = 0; k < m; k += 3) {
        a   = params[k];
        mu  = params[k + 1];
        sig = params[k + 2];
        if (sig <= 0.0) sig = 1e-30;
        col = k;
        for (i = 0; i < n; i++) {
            z = (x[i] - mu) / sig;
            g = exp(-0.5 * z * z);
            jac[i * m + col]     = g;                    /* dF/da   */
            jac[i * m + col + 1] = a * g * z / sig;      /* dF/dmu  */
            jac[i * m + col + 2] = a * g * z * z / sig;  /* dF/dsig */
        }
    }
}

/* -----------------------------------------------------------------------
 * 3. Cholesky factorisation & solve  (A x = b, A is m x m SPD)
 *    Overwrites A with L and b with the solution.
 * ----------------------------------------------------------------------- */

static int
cholesky_solve(double *A, double *b, int m)
{
    int i, j, k;
    double s;

    /* Cholesky decomposition: A = L L^T */
    for (j = 0; j < m; j++) {
        s = A[j * m + j];
        for (k = 0; k < j; k++)
            s -= A[j * m + k] * A[j * m + k];
        if (s <= 0.0) return -1;  /* not positive definite */
        A[j * m + j] = sqrt(s);
        for (i = j + 1; i < m; i++) {
            s = A[i * m + j];
            for (k = 0; k < j; k++)
                s -= A[i * m + k] * A[j * m + k];
            A[i * m + j] = s / A[j * m + j];
        }
    }

    /* Forward substitution: L y = b */
    for (i = 0; i < m; i++) {
        s = b[i];
        for (k = 0; k < i; k++)
            s -= A[i * m + k] * b[k];
        b[i] = s / A[i * m + i];
    }

    /* Back substitution: L^T x = y */
    for (i = m - 1; i >= 0; i--) {
        s = b[i];
        for (k = i + 1; k < m; k++)
            s -= A[k * m + i] * b[k];
        b[i] = s / A[i * m + i];
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * 4. Bounded Levenberg-Marquardt solver
 * ----------------------------------------------------------------------- */

static int
bounded_lm_solve(const double *x, const double *y, int n,
                 double *params, int m,
                 const double *lower, const double *upper,
                 int maxfev, double ftol, double xtol, double gtol)
{
    /*
     * Working memory layout (all stack-allocated for typical sizes):
     *   r[n]         residuals
     *   model[n]     model evaluation
     *   jac[n*m]     Jacobian
     *   jtj[m*m]     J^T J + lambda * diag
     *   jtjsave[m*m] saved copy for retry
     *   grad[m]      J^T r  (gradient)
     *   delta[m]     step
     *   p_try[m]     candidate params
     */

    /* Use heap for large problems to avoid stack overflow. */
    double r_stack[STACK_N], model_stack[STACK_N];
    double jac_stack[STACK_N * MAX_PARAMS];
    double *r_buf = NULL, *model_buf = NULL, *jac_buf = NULL;
    double *r, *model, *jac;

    double jtj[MAX_PARAMS * MAX_PARAMS];
    double jtjsave[MAX_PARAMS * MAX_PARAMS];
    double grad[MAX_PARAMS], delta[MAX_PARAMS], p_try[MAX_PARAMS];

    double lambda, cost, cost_new, rho;
    double max_grad, max_delta, max_p;
    int nfev, iter;
    int i, j, k;
    int converged = 0;

    if (m > MAX_PARAMS) return -1;

    /* Allocate heap buffers if needed */
    if (n > STACK_N) {
        r_buf     = (double *)malloc((size_t)n * sizeof(double));
        model_buf = (double *)malloc((size_t)n * sizeof(double));
        jac_buf   = (double *)malloc((size_t)n * (size_t)m * sizeof(double));
        if (!r_buf || !model_buf || !jac_buf) {
            free(r_buf); free(model_buf); free(jac_buf);
            return -2;  /* allocation failure */
        }
        r     = r_buf;
        model = model_buf;
        jac   = jac_buf;
    } else {
        r     = r_stack;
        model = model_stack;
        jac   = jac_stack;
    }

    /* Clip initial params to bounds */
    for (j = 0; j < m; j++) {
        if (params[j] < lower[j]) params[j] = lower[j];
        if (params[j] > upper[j]) params[j] = upper[j];
    }

    /* Initial cost */
    eval_multi_gaussian(x, n, params, m, model);
    nfev = 1;
    cost = 0.0;
    for (i = 0; i < n; i++) {
        r[i] = model[i] - y[i];
        cost += r[i] * r[i];
    }
    cost *= 0.5;

    /* Initial lambda from max diagonal of J^T J */
    eval_jacobian(x, n, params, m, jac);
    lambda = 0.0;
    for (j = 0; j < m; j++) {
        double djj = 0.0;
        for (i = 0; i < n; i++)
            djj += jac[i * m + j] * jac[i * m + j];
        if (djj > lambda) lambda = djj;
    }
    lambda *= 1e-3;
    if (lambda < 1e-10) lambda = 1e-10;

    for (iter = 0; iter < maxfev && nfev < maxfev; iter++) {
        /* Compute J^T J and J^T r (gradient) */
        for (j = 0; j < m; j++) {
            grad[j] = 0.0;
            for (i = 0; i < n; i++)
                grad[j] += jac[i * m + j] * r[i];
        }
        for (j = 0; j < m; j++) {
            for (k = 0; k < m; k++) {
                double s = 0.0;
                for (i = 0; i < n; i++)
                    s += jac[i * m + j] * jac[i * m + k];
                jtj[j * m + k] = s;
            }
        }

        /* Check gradient convergence */
        max_grad = 0.0;
        for (j = 0; j < m; j++) {
            double ag = fabs(grad[j]);
            if (ag > max_grad) max_grad = ag;
        }
        if (max_grad < gtol) { converged = 1; break; }

        /* Damped normal equations: (J^T J + lambda * diag(J^T J)) * delta = -grad */
        memcpy(jtjsave, jtj, (size_t)m * (size_t)m * sizeof(double));
        for (j = 0; j < m; j++) {
            double d = jtj[j * m + j];
            if (d < 1e-30) d = 1e-30;
            jtj[j * m + j] = d + lambda * d;
        }
        for (j = 0; j < m; j++) delta[j] = -grad[j];

        if (cholesky_solve(jtj, delta, m) != 0) {
            /* Not positive definite; increase lambda and retry */
            lambda *= 10.0;
            memcpy(jtj, jtjsave, (size_t)m * (size_t)m * sizeof(double));
            continue;
        }

        /* Project candidate to bounds */
        for (j = 0; j < m; j++) {
            p_try[j] = params[j] + delta[j];
            if (p_try[j] < lower[j]) p_try[j] = lower[j];
            if (p_try[j] > upper[j]) p_try[j] = upper[j];
        }

        /* Evaluate candidate */
        eval_multi_gaussian(x, n, p_try, m, model);
        nfev++;
        cost_new = 0.0;
        for (i = 0; i < n; i++) {
            double ri = model[i] - y[i];
            cost_new += ri * ri;
        }
        cost_new *= 0.5;

        /* Compute gain ratio */
        {
            double predicted = 0.0;
            for (j = 0; j < m; j++)
                predicted += delta[j] * (lambda * jtjsave[j * m + j] * delta[j] - grad[j]);
            predicted *= 0.5;
            if (predicted < 1e-30) predicted = 1e-30;
            rho = (cost - cost_new) / predicted;
        }

        if (rho > 1e-4) {
            /* Accept step */
            memcpy(params, p_try, (size_t)m * sizeof(double));

            /* Update residuals and cost */
            cost = cost_new;
            for (i = 0; i < n; i++)
                r[i] = model[i] - y[i];

            /* Recompute Jacobian */
            eval_jacobian(x, n, params, m, jac);

            /* Adjust lambda (Marquardt update) */
            {
                double factor = 1.0 - (2.0 * rho - 1.0) * (2.0 * rho - 1.0) * (2.0 * rho - 1.0);
                if (factor < 1.0 / 3.0) factor = 1.0 / 3.0;
                lambda *= factor;
                if (lambda < 1e-15) lambda = 1e-15;
            }

            /* Check cost convergence (ftol) */
            if (cost_new > 0.0) {
                double rel_change = fabs(cost - cost_new) / cost_new;
                /* cost was already updated, so use the ratio of actual vs previous */
                if (fabs(cost_new) < ftol * ftol) { converged = 1; break; }
            }

            /* Check step convergence (xtol) */
            max_delta = 0.0;
            max_p = 0.0;
            for (j = 0; j < m; j++) {
                double ad = fabs(delta[j]);
                double ap = fabs(params[j]);
                if (ad > max_delta) max_delta = ad;
                if (ap > max_p) max_p = ap;
            }
            if (max_p < 1e-30) max_p = 1e-30;
            if (max_delta / max_p < xtol) { converged = 1; break; }

        } else {
            /* Reject step, increase lambda */
            lambda *= 2.5;
            if (lambda > 1e16) lambda = 1e16;
            /* Restore jtj for next iteration */
            memcpy(jtj, jtjsave, (size_t)m * (size_t)m * sizeof(double));
        }
    }

    free(r_buf);
    free(model_buf);
    free(jac_buf);

    if (!converged && nfev >= maxfev)
        return 1;  /* maxfev reached without convergence */
    return 0;
}

/* -----------------------------------------------------------------------
 * 5. Python C API wrapper
 * ----------------------------------------------------------------------- */

static PyObject *
py_bounded_lm_fit(PyObject *self, PyObject *args)
{
    PyArrayObject *x_arr, *y_arr, *p0_arr, *lower_arr, *upper_arr;
    int maxfev;
    double ftol = 1.49012e-8, xtol = 1.49012e-8, gtol = 1.49012e-8;

    (void)self;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i",
                          &PyArray_Type, &x_arr,
                          &PyArray_Type, &y_arr,
                          &PyArray_Type, &p0_arr,
                          &PyArray_Type, &lower_arr,
                          &PyArray_Type, &upper_arr,
                          &maxfev))
        return NULL;

    /* Ensure contiguous float64 arrays */
    x_arr     = (PyArrayObject *)PyArray_GETCONTIGUOUS(x_arr);
    y_arr     = (PyArrayObject *)PyArray_GETCONTIGUOUS(y_arr);
    p0_arr    = (PyArrayObject *)PyArray_GETCONTIGUOUS(p0_arr);
    lower_arr = (PyArrayObject *)PyArray_GETCONTIGUOUS(lower_arr);
    upper_arr = (PyArrayObject *)PyArray_GETCONTIGUOUS(upper_arr);

    if (!x_arr || !y_arr || !p0_arr || !lower_arr || !upper_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(p0_arr);
        Py_XDECREF(lower_arr);
        Py_XDECREF(upper_arr);
        PyErr_SetString(PyExc_TypeError, "Failed to get contiguous arrays");
        return NULL;
    }

    int n = (int)PyArray_SIZE(x_arr);
    int m = (int)PyArray_SIZE(p0_arr);

    if (m > MAX_PARAMS) {
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(p0_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_Format(PyExc_ValueError,
                     "Too many parameters (%d > %d)", m, MAX_PARAMS);
        return NULL;
    }

    if (n != (int)PyArray_SIZE(y_arr)) {
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(p0_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        PyErr_SetString(PyExc_ValueError, "x and y must have same length");
        return NULL;
    }

    /* Copy p0 so solver can modify in place */
    npy_intp dims[1] = {m};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) {
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(p0_arr);
        Py_DECREF(lower_arr);
        Py_DECREF(upper_arr);
        return NULL;
    }
    memcpy(PyArray_DATA(result), PyArray_DATA(p0_arr), (size_t)m * sizeof(double));

    const double *xd = (const double *)PyArray_DATA(x_arr);
    const double *yd = (const double *)PyArray_DATA(y_arr);
    double *pd       = (double *)PyArray_DATA(result);
    const double *ld = (const double *)PyArray_DATA(lower_arr);
    const double *ud = (const double *)PyArray_DATA(upper_arr);

    int status;
    Py_BEGIN_ALLOW_THREADS
    status = bounded_lm_solve(xd, yd, n, pd, m, ld, ud, maxfev,
                              ftol, xtol, gtol);
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(p0_arr);
    Py_DECREF(lower_arr);
    Py_DECREF(upper_arr);

    if (status == 1) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError,
                        "bounded_lm_fit: maxfev reached without convergence");
        return NULL;
    }
    if (status == -1) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError,
                        "bounded_lm_fit: too many parameters");
        return NULL;
    }
    if (status == -2) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_MemoryError,
                        "bounded_lm_fit: memory allocation failed");
        return NULL;
    }

    return (PyObject *)result;
}

/* -----------------------------------------------------------------------
 * 6. Persistence peak detection (0-dim persistent homology)
 * ----------------------------------------------------------------------- */

typedef struct {
    double birth;
    double death;
    double persistence;
    int index;
    int saddle_index;
} PeakResult;

/* Global signal pointer for qsort comparator (safe under GIL). */
static const double *g_sort_signal;

static int
cmp_idx_by_signal_desc(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double va = g_sort_signal[ia];
    double vb = g_sort_signal[ib];
    if (va > vb) return -1;
    if (va < vb) return  1;
    return (ia > ib) - (ia < ib);  /* tie-break: ascending index */
}

static int
cmp_peak_persistence_desc(const void *a, const void *b)
{
    double pa = ((const PeakResult *)a)->persistence;
    double pb = ((const PeakResult *)b)->persistence;
    if (pa > pb) return -1;
    if (pa < pb) return  1;
    return 0;
}

/* Union-find: find with path-halving compression. */
static int
uf_find(int *parent, int x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/* Union-find: union by rank, keeping the higher-valued representative. */
static int
uf_union(int *parent, int *uf_rank, int *rep, const double *signal,
         int a, int b)
{
    if (uf_rank[a] < uf_rank[b]) { int t = a; a = b; b = t; }
    parent[b] = a;
    if (uf_rank[a] == uf_rank[b])
        uf_rank[a]++;
    if (signal[rep[b]] > signal[rep[a]])
        rep[a] = rep[b];
    return a;
}

/*
 * c_find_peaks -- persistence peak detection in C.
 *
 * signal[n]          input signal values
 * min_persistence    discard peaks below this persistence
 * out_peaks[n]       caller-allocated output buffer (at most n peaks)
 * *out_count         number of peaks written
 *
 * Returns 0 on success, -1 on allocation failure.
 */
static int
c_find_peaks(const double *signal, int n, double min_persistence,
             PeakResult *out_peaks, int *out_count)
{
    int order_stack[STACK_N], parent_stack[STACK_N];
    int rank_stack[STACK_N],  rep_stack[STACK_N];
    char visited_stack[STACK_N];

    int  *order, *par, *rnk, *rep_arr;
    char *vis;
    int  *heap_int  = NULL;
    char *heap_vis  = NULL;

    int i, idx, neighbor, idx_root, nb_root, side;
    int rep_ir, rep_nr, younger;
    int comp, merged, peak_count = 0;

    *out_count = 0;
    if (n == 0) return 0;

    /* Allocate working arrays */
    if (n <= STACK_N) {
        order   = order_stack;
        par     = parent_stack;
        rnk     = rank_stack;
        rep_arr = rep_stack;
        vis     = visited_stack;
    } else {
        heap_int = (int *)malloc(4 * (size_t)n * sizeof(int));
        heap_vis = (char *)malloc((size_t)n);
        if (!heap_int || !heap_vis) {
            free(heap_int); free(heap_vis);
            return -1;
        }
        order   = heap_int;
        par     = heap_int + n;
        rnk     = heap_int + 2 * n;
        rep_arr = heap_int + 3 * n;
        vis     = heap_vis;
    }

    for (i = 0; i < n; i++) {
        order[i]   = i;
        par[i]     = i;
        rnk[i]     = 0;
        rep_arr[i] = i;
        vis[i]     = 0;
    }

    /* Argsort indices by descending signal value */
    g_sort_signal = signal;
    qsort(order, (size_t)n, sizeof(int), cmp_idx_by_signal_desc);

    /* Sweep from highest to lowest */
    for (i = 0; i < n; i++) {
        idx = order[i];
        vis[idx] = 1;
        comp = idx;

        for (side = 0; side < 2; side++) {
            neighbor = idx + (side == 0 ? -1 : 1);
            if (neighbor < 0 || neighbor >= n || !vis[neighbor])
                continue;

            nb_root  = uf_find(par, neighbor);
            idx_root = uf_find(par, comp);
            if (idx_root == nb_root)
                continue;

            rep_ir  = rep_arr[idx_root];
            rep_nr  = rep_arr[nb_root];
            younger = (signal[rep_ir] < signal[rep_nr]) ? rep_ir : rep_nr;

            {
                double death       = signal[idx];
                double birth       = signal[younger];
                double persistence = birth - death;
                if (persistence > min_persistence) {
                    out_peaks[peak_count].index       = younger;
                    out_peaks[peak_count].birth       = birth;
                    out_peaks[peak_count].death       = death;
                    out_peaks[peak_count].persistence = persistence;
                    out_peaks[peak_count].saddle_index = idx;
                    peak_count++;
                }
            }

            merged = uf_union(par, rnk, rep_arr, signal,
                              uf_find(par, comp), uf_find(par, neighbor));
            comp = merged;
        }
    }

    /* Global maximum (never dies) */
    {
        int    gmax_idx   = order[0];
        double gmax_birth = signal[gmax_idx];
        if (gmax_birth >= min_persistence) {
            out_peaks[peak_count].index       = gmax_idx;
            out_peaks[peak_count].birth       = gmax_birth;
            out_peaks[peak_count].death       = 0.0;
            out_peaks[peak_count].persistence = gmax_birth;
            out_peaks[peak_count].saddle_index = -1;
            peak_count++;
        }
    }

    /* Sort peaks by persistence (most significant first) */
    if (peak_count > 1)
        qsort(out_peaks, (size_t)peak_count, sizeof(PeakResult),
              cmp_peak_persistence_desc);

    free(heap_int);
    free(heap_vis);
    *out_count = peak_count;
    return 0;
}

/* -----------------------------------------------------------------------
 * 7. Python wrapper for persistence peak detection
 * ----------------------------------------------------------------------- */

static PyObject *
py_find_peaks(PyObject *self, PyObject *args)
{
    PyArrayObject *signal_arr;
    double min_persistence;
    int n, status, peak_count, i;
    const double *sig;
    PeakResult *peaks;
    PeakResult peaks_stack[STACK_N];
    PyObject *result, *tup;

    (void)self;

    if (!PyArg_ParseTuple(args, "O!d",
                          &PyArray_Type, &signal_arr,
                          &min_persistence))
        return NULL;

    signal_arr = (PyArrayObject *)PyArray_GETCONTIGUOUS(signal_arr);
    if (!signal_arr) {
        PyErr_SetString(PyExc_TypeError, "Failed to get contiguous array");
        return NULL;
    }

    n   = (int)PyArray_SIZE(signal_arr);
    sig = (const double *)PyArray_DATA(signal_arr);

    /* Allocate output buffer */
    if (n <= STACK_N) {
        peaks = peaks_stack;
    } else {
        peaks = (PeakResult *)malloc((size_t)n * sizeof(PeakResult));
        if (!peaks) {
            Py_DECREF(signal_arr);
            return PyErr_NoMemory();
        }
    }

    status = c_find_peaks(sig, n, min_persistence, peaks, &peak_count);
    Py_DECREF(signal_arr);

    if (status != 0) {
        if (n > STACK_N) free(peaks);
        return PyErr_NoMemory();
    }

    /* Build Python list of tuples */
    result = PyList_New(peak_count);
    if (!result) {
        if (n > STACK_N) free(peaks);
        return NULL;
    }

    for (i = 0; i < peak_count; i++) {
        tup = Py_BuildValue("(idddi)",
                            peaks[i].index,
                            peaks[i].birth,
                            peaks[i].death,
                            peaks[i].persistence,
                            peaks[i].saddle_index);
        if (!tup) {
            if (n > STACK_N) free(peaks);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, tup);
    }

    if (n > STACK_N) free(peaks);
    return result;
}

/* -----------------------------------------------------------------------
 * Module definition
 * ----------------------------------------------------------------------- */

static PyMethodDef gaussfit_methods[] = {
    {"bounded_lm_fit", py_bounded_lm_fit, METH_VARARGS,
     "bounded_lm_fit(x, y, p0, lower, upper, maxfev) -> ndarray\n\n"
     "Fit a sum of Gaussians using bounded Levenberg-Marquardt.\n"
     "Parameters are flat arrays [a0, mu0, sig0, a1, mu1, sig1, ...].\n"
     "Returns optimised parameter array."},
    {"find_peaks", py_find_peaks, METH_VARARGS,
     "find_peaks(signal, min_persistence) -> list[tuple]\n\n"
     "Find peaks by 0-dim persistent homology.\n"
     "Returns list of (index, birth, death, persistence, saddle_index) tuples."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gaussfit_module = {
    PyModuleDef_HEAD_INIT,
    "_gaussfit",
    "Bounded Levenberg-Marquardt Gaussian fitting (C extension).",
    -1,
    gaussfit_methods
};

PyMODINIT_FUNC
PyInit__gaussfit(void)
{
    import_array();
    return PyModule_Create(&gaussfit_module);
}
