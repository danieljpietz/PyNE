import numba as nb
import numpy as np
import sympy as sym
from .func import *
from .QP import qpsolve


def compile_symbolic(fun, t, x, xdot):
    fun = sym.core.add.Add(fun)
    args = list(x) + list(xdot)
    nb_args = nb.float64(*(len(args) * (nb.float64,)))

    f_njit = nb.njit(nb_args)(sym.lambdify(args, fun, "numpy"))

    args_func_symbols = sym.symbols(
        " ".join(arg.name.upper() for arg in args), cls=sym.Function
    )
    args_func_obj = [f(t) for f in args_func_symbols]

    dotf_args = list(xdot) + [sym.var(f"xddot{i}") for i in range(len(x))]
    f_subbed = fun.subs(zip(args, args_func_obj))
    d_f_subbed = f_subbed.diff(t)

    dotf_symbolic = d_f_subbed.subs(
        zip((f.diff(t) for f in args_func_obj), dotf_args)
    ).subs(zip(args_func_obj, args))
    dotf_symbolic = sym.core.add.Add(dotf_symbolic)

    dotf_args_full = list(x) + dotf_args
    dot_nb_args = nb.float64(*(len(dotf_args_full) * (nb.float64,)))
    dotf_njit = nb.njit(dot_nb_args)(
        sym.lambdify(dotf_args_full, dotf_symbolic, "numpy")
    )

    grad = [fun.diff(x) for x in args]
    g_lambda = [sym.lambdify(args, g, "numpy") for g in grad]
    g_njit = tuple((nb.njit(nb_args)(g) for g in g_lambda))

    h_njit = [[None for _ in range(len(args))] for _ in range(len(args))]

    for i in range(len(h_njit)):
        for j in range(i + 1):
            h = sym.Function.diff(grad[i], args[j])
            h_lambda = sym.lambdify(args, h, "numpy")
            h = nb.njit(nb_args)(h_lambda)
            h_njit[i][j] = h
            h_njit[j][i] = h

    for i in range(len(h_njit)):
        h_njit[i] = tuple(h_njit[i])

    h_njit = tuple(h_njit)

    def _numba_nparray_to_tuple(len):
        fun_str = f"nb.njit() \ndef __ar_to_tuple(array): \n\ttup = {', '.join([f'array[{i}]' for i in range(len)])} \n\treturn tup"
        d = {"nb": nb}
        exec(fun_str, d)
        _nj = nb.njit(d["__ar_to_tuple"])
        return _nj

    dof2 = len(args)
    _ar_to_tuple = _numba_nparray_to_tuple(dof2)

    @nb.njit(nb.float64(nb.float64[:]))
    def _cbf_compiled(X):
        return ((f_njit(*_ar_to_tuple(X))))

    dof3 = len(dotf_args_full)
    _d_ar_to_tuple = _numba_nparray_to_tuple(dof3)

    @nb.njit(nb.float64(nb.float64[:]))
    def _dot_cbf_compiled(X):
        return ((dotf_njit(*_d_ar_to_tuple(X))))

    @nb.njit(nb.float64[:](nb.float64[:]))
    def _grad_compiled(X):
        _tup = _ar_to_tuple(X)
        _g = np.zeros(dof2, dtype=nb.float64)
        for i in range(dof2):
            _g[i] = g_njit[i](*_tup)
        return _g

    @nb.njit(nb.float64[:, :](nb.float64[:]))
    def _hessian_compiled(X):
        _tup = _ar_to_tuple(X)
        _h = np.zeros((dof2, dof2), dtype=nb.float64)
        for i in range(dof2):
            for j in range(i + 1):
                _h[i, j] = h_njit[i][j](*_tup)
                _h[j, i] = _h[i][j]
        return _h

    return (_cbf_compiled,), (_dot_cbf_compiled,), (_grad_compiled,), (_hessian_compiled,)


def cbf_vars(dof):
    return (
        sym.var("t"),
        sym.Matrix((sym.var(" ".join([f"x{i}" for i in range(dof)])),)),
        sym.Matrix((sym.var(" ".join([f"xdot{i}" for i in range(dof)])),)),
    )

@nb.njit(fastmath=True, cache=True)
def cbf_eval(syspacket, cbf, uref, umin, umax):
    barrier, dot_barrier, gradient, hessian = cbf
    (
        dof,
        f,
        j_f,
        g,
    ) = syspacket

    c1, c2 = 1, 1

    lhs_mult = hessian @ f + j_f.T @ gradient

    A = lhs_mult @ (-g)
    A = np.reshape(A, (1, dof))
    A = np.concatenate((A, np.eye(dof)))
    A = np.concatenate((A, -np.eye(dof)))

    b = np.array((((lhs_mult + gradient) @ f + c1 * barrier + c2 * dot_barrier),))
    b = np.concatenate((b, umax))
    b = np.concatenate((b, -umin))
    H = np.eye(dof)  # Cost matrix
    f_ = -H @ (uref)

    return qpsolve(H, f_, A, -np.abs(b) - 1, b)
