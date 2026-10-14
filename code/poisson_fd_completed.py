"""Student scaffold: -Delta u = f on (0,1)^2 with Dirichlet data.

The example problems live in test_problem_bank.py and are imported here so
the scaffold and the bank can never drift apart.

    PROBLEM = P1  ->  Lecture 3 coding lab.  u = sin(pi x) sin(pi y) is ZERO
                      on the whole boundary, so b = f sampled at interior
                      nodes and there are NO boundary values to move across.
    PROBLEM = P5  ->  Tutorial 5 individual challenge.  u = e^x sin(pi y) is
                      NON-ZERO on x = 0 and x = 1, so TODO 2 must also move
                      the boundary values across (b[k] += u_edge / h^2).

Swap the PROBLEM line to switch examples; everything else is unchanged.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from test_problem_bank import P1, P5

PROBLEM = P1  # Lecture 3: zero Dirichlet, RHS = f only.


def manufactured_solution(x, y):
    return PROBLEM.exact(x, y)


def forcing(x, y):
    return PROBLEM.forcing(x, y)


def assemble_poisson(n):
    """Return (A,b,h) for n interior nodes per direction."""
    if n < 2:
        raise ValueError("n must be at least 2")
    h = 1.0 / (n + 1)
    grid = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    rows, cols, data = [], [], []
    b = np.zeros(n * n)
    # TODO 1: assemble the five-point matrix for -Delta.  Use k = i*n+j.
    #   Diagonal 4/h^2; each interior neighbour -1/h^2.
    # TODO 2: build b from f = -Delta u*.
    #   PROBLEM = P1:  b[k] = f(x_i, y_j)   (u* = 0 on the boundary, so b = f).
    #   PROBLEM = P5:  also move the Dirichlet values across -- for every node
    #   with a neighbour ON the boundary, b[k] += u(x_edge, y_edge)/h^2.
    for i in range(n):
        for j in range(n):
            k = i * n + j
            x, y = X[i, j], Y[i, j]
            b[k] = forcing(x, y)
            rows.append(k); cols.append(k); data.append(4.0 / h**2)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    rows.append(k)
                    cols.append(ni * n + nj)
                    data.append(-1.0 / h**2)
                else:
                    x_edge = (ni + 1) * h
                    y_edge = (nj + 1) * h
                    b[k] += manufactured_solution(x_edge, y_edge) / h**2
    A = csr_matrix((data, (rows, cols)), shape=(n*n, n*n))
    return A, b, h


def solve_and_error(n):
    A, b, h = assemble_poisson(n)
    uh = spsolve(A, b)
    grid = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    exact = manufactured_solution(X, Y).ravel()
    residual = np.linalg.norm(b - A @ uh, np.inf)
    error_l2 = h * np.linalg.norm(uh - exact)
    error_linf = np.linalg.norm(uh - exact, np.inf)
    return error_l2, error_linf, residual


def convergence_study(ns=(7, 15, 31, 63)):
    # TODO: run solve_and_error and compute observed orders under factor-2 refinement.
    results = []
    previous_error = None
    for n in ns:
        h = 1.0 / (n + 1)
        error_l2, error_linf, residual = solve_and_error(n)
        order = None if previous_error is None else np.log2(previous_error / error_l2)
        results.append((h, error_l2, error_linf, residual, order))
        previous_error = error_l2
    return results


def truncation_error(n):
    """Return ||b - A u*||_2, the local truncation error of the scheme.

    u* is the manufactured solution sampled on the interior nodes.  Because A
    approximates -Delta and b samples f = -Delta u*, the difference b - A u*
    is the pointwise misfit of the discrete operator on the exact solution.

    Norm watch: each component is O(h^2), but the vector has ~1/h^2 components
    in 2D, so the RAW 2-norm decays like O(h).  Use h*||.||_2 (the discrete L^2
    norm, as in the test bank) to recover O(h^2).
    """
    A, b, h = assemble_poisson(n)
    grid = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    # TODO: sample u* on the interior nodes and compute ||b - A @ u_star||_2.
    # u_star = manufactured_solution(X, Y).ravel()
    u_star = manufactured_solution(X, Y).ravel()
    return np.linalg.norm(b - A @ u_star, 2)


def truncation_study(ns=(7, 15, 31, 63)):
    """Return (h, ||b - A u*||_2, observed order) for each mesh."""
    # TODO: call truncation_error on each n and compute orders under
    # factor-2 refinement (mirror convergence_study).  Expected ~ 1 for the
    # raw 2-norm (see truncation_error); h*||tau||_2 would fit ~ 2.
    results = []
    previous_error = None
    for n in ns:
        h = 1.0 / (n + 1)
        error = truncation_error(n)
        order = None if previous_error is None else np.log2(previous_error / error)
        results.append((h, error, order))
        previous_error = error
    return results


def plot_solution(n=15, show=True):
    """Plot the analytical solution sampled on the mesh grid, u_h overlaid.

    TODO 3 of the coding lab: solve with spsolve, then plot u* on the grid
    points so the comparison with u_h is direct.
    """
    import matplotlib.pyplot as plt

    A, b, h = assemble_poisson(n)
    uh = spsolve(A, b)
    grid = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    u_star = manufactured_solution(X, Y)
    # TODO: filled contour of u_star; overlay u_h as a scatter or a second
    # contour so the agreement is visible at the grid points.
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, u_star)
    # ...
    # if show: plt.show() else: plt.savefig("solution_square.png")
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, u_star, levels=20)
    fig.colorbar(cf, ax=ax, label="Exact solution")
    ax.scatter(X.ravel(), Y.ravel(), c=uh, cmap=cf.cmap,
               norm=cf.norm, s=15, edgecolors="black", linewidths=0.2)
    ax.set(xlabel="x", ylabel="y", title=f"Exact contours and FDM nodes (n={n})")
    ax.set_aspect("equal")
    if show:
        plt.show()
    else:
        fig.savefig("solution_square.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # ---- Coding lab (Lecture 3): assemble, solve, verify ----
    # PROBLEM = P1: zero Dirichlet, so b = f only.  Sanity check for the lab:
    #   u_h(0.5,0.5) ~ 1.003 at n = 15.
    # For the Tutorial 5 individual challenge, set PROBLEM = P5 and check
    #   u_h(0.5,0.5) ~ 1.65 at n = 15.
    n = 15
    A, b, h = assemble_poisson(n)        # TODO 1 & 2: sparse assembly, RHS
    uh = spsolve(A, b)                   # TODO 3: solve
    center = (n // 2) * n + (n // 2)
    print("Center u_h:", uh[center])
    print("Residual:", np.linalg.norm(b - A @ uh, np.inf))
    print("Convergence (h, L2, Linf, residual, order):")
    print(convergence_study())

    # verify 1: sparsity pattern and a few selected rows
    # plt.spy(A) or print(A.getrow(0).toarray())  -- inspect before solving

    # verify 2: truncation error ||b - A u*|| and its decay with h
    print(truncation_study())

    # TODO 3 (visual): overlay the analytical solution on the mesh
    plot_solution(n)
