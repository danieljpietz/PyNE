import numba as nb
import numpy as np
import sympy as sym
from .func import *
from .QP import qpsolve


def _numba_nparray_to_tuple(len):
    fun_str = f"nb.njit() \ndef __ar_to_tuple(array): \n\ttup = {', '.join([f'array[{i}]' for i in range(len)])} \n\treturn tup"
    d = {"nb": nb}
    exec(fun_str, d)
    _nj = nb.njit(d["__ar_to_tuple"])
    return _nj


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

    dof2 = len(args)
    _ar_to_tuple = _numba_nparray_to_tuple(dof2)


    def _cbf_compiled(X):
        return ((f_njit(*_ar_to_tuple(X))))

    dof3 = len(dotf_args_full)
    _d_ar_to_tuple = _numba_nparray_to_tuple(dof3)


    def _dot_cbf_compiled(X):
        return ((dotf_njit(*_d_ar_to_tuple(X))))


    def _grad_compiled(X):
        _tup = _ar_to_tuple(X)
        _g = np.zeros(dof2, dtype=float)
        for i in range(dof2):
            _g[i] = g_njit[i](*_tup)
        return _g


    def _hessian_compiled(X):
        _tup = _ar_to_tuple(X)
        _h = np.zeros((dof2, dof2), dtype=float)
        for i in range(dof2):
            for j in range(i + 1):
                _h[i, j] = h_njit[i][j](*_tup)
                _h[j, i] = _h[i][j]
        return _h

    return (_cbf_compiled,), (_dot_cbf_compiled,), (_grad_compiled,), (_hessian_compiled,)


def uref_compile(fun, t, x, xdot):
    args = list(x) + list(xdot)
    nb_args = nb.float64[:](*(len(args) * (nb.float64,)))
    fun = fun.flatten()
    vec = [sym.lambdify(args, func, "numpy") for func in fun]

    lambd = lambda *args: np.array([func(*args) for func in vec])

    _urefs_ar_tuple = _numba_nparray_to_tuple(len(args))

    dof = len(x)

    def _uref_func(X):
        _tup = _urefs_ar_tuple(X)
        return lambd(*_tup)

    return _uref_func,


def cbf_vars(dof):
    return (
        sym.var("t"),
        sym.Matrix((sym.var(" ".join([f"x{i}" for i in range(dof)])),)),
        sym.Matrix((sym.var(" ".join([f"xdot{i}" for i in range(dof)])),)),
    )



def cbf_eval(syspacket, cbf, clf, uref, umin, umax):

    barrier, dot_barrier, barrier_gradient, barrier_hessian = cbf
    lyaponov, dot_lyaponov, lyaponov_gradient, lyaponov_hessian = clf

    (
        dof,
        f,
        j_f,
        g,
    ) = syspacket

    ulen = len(uref)

    #B1, B2 = 5, 5 # 7DOF
    B1, B2 = 5, 2 # Drone
    L1, L2 = 1, 1

    lhs_mult_cbf = barrier_hessian @ f + barrier_gradient @ j_f
    lhs_mult_clf = lyaponov_hessian @ f + lyaponov_gradient @ j_f

    A = np.zeros((2, ulen + 1))
    A[0,:ulen] = (barrier_gradient + lhs_mult_cbf) @ (-g)
    A[1,:ulen] = (lyaponov_gradient + lhs_mult_clf) @ g
    A[1, ulen] = -1
    b = np.zeros((2))

    b[0] = ((lhs_mult_cbf + barrier_gradient) @ f + B1 * barrier + B2 * dot_barrier)
    b[1] = -((lhs_mult_clf + lyaponov_gradient) @ (f) + L1 * lyaponov + L2 * dot_lyaponov)

    if umax is not None:
        A = np.concatenate((A, np.eye(dof)))
        b = np.concatenate((b, umax))

    if umin is not None:
        A = np.concatenate((A, -np.eye(dof)))
        b = np.concatenate((b, -umin))



    H = np.eye(ulen + 1) # Cost matrix
    H[ulen, ulen] = 1
    f_ = -H @ (np.concatenate((uref, np.array([0]))))
    u = qpsolve(H, f_, A, -np.abs(b) - 1.25, b)

    return u[:ulen]


class ControlFunc:
    def __init__(self, cbf=1, clf=0):
        self._vars = vars
        self.cbf = cbf
        self.clf = clf

    def uref(self, x, xdot):
        return np.zeros(self.input_matrix(x, xdot).shape[1])

    def input_matrix(self, x, xdot):
        return np.eye(len(self._vars[1]))

    def _jit(self):
        return compile_symbolic(self.cbf, *self._vars), compile_symbolic(self.clf, *self._vars), self.uref, self.input_matrix, #return compile_symbolic(self.cbf, *self._vars), compile_symbolic(self.clf, *self._vars), , np.array(self.input_matrix), #

