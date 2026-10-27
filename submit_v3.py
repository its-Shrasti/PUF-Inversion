# submit.py
"""
Submission file for Problems 1.1/1.2:
- my_kernel(X1,Z1,X2,Z2) implements the semiparametric polynomial kernel.
- my_decode(w) returns eight 32-dim non-negative delay vectors for XOR Arbiter PUF inversion.
- Includes optional CLI to process a batch file of models and save CSV (for your inspection).
"""

import numpy as np
import argparse
import time
# sklearn allowed for kernel helpers
from sklearn.metrics.pairwise import linear_kernel

# -----------------------
# Kernel (Task 3)
# -----------------------
def my_kernel(X1, Z1, X2, Z2):
    """
    Compute the kernel matrix K̃((x,z),(x',z')) = x x' (z^T z' + c)^d + 1

    Parameters:
        X1 : n1 × 1 array   (video lengths for set 1)
        Z1 : n1 × 2 array   (z features for set 1)
        X2 : n2 × 1 array   (video lengths for set 2)
        Z2 : n2 × 2 array   (z features for set 2)

    Returns:
        K : n1 × n2 kernel matrix
    """
    # Ensure input arrays are numpy
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    # Chosen hyperparameters (you can change these when tuning)
    d = 5     # polynomial degree
    c = 0.1   # offset

    # Compute (z_i^T z_j + c)^d
    Z_dot = linear_kernel(Z1, Z2)       # shape: (n1, n2)
    poly_part = (Z_dot + c) ** d        # elementwise power

    # Multiply by x_i * x_j
    x_prod = X1 @ X2.T                  # shape: (n1, n2)

    # Final kernel
    K = x_prod * poly_part + 1.0

    return K

# -----------------------
# Problem 1.2 solver (Task 5)
# -----------------------
# Config: toggle refinement for local testing (kept False for speed / determinism)
DO_REFINE = False
REFINE_ITERS = 6
SUB_ITERS = 80

Kbits = 32
N_U = Kbits + 1         # 33
W_LEN = N_U * N_U       # 1089
DELAY_PER_ARBITER = 4 * Kbits  # 128
TOTAL_DELAYS = 2 * DELAY_PER_ARBITER  # 256

def build_B_matrix(k=Kbits):
    """Build B (33 x 128) mapping 128 delays to u (33)."""
    n = k + 1
    cols = 4 * k
    B = np.zeros((n, cols), dtype=float)
    alpha_coeffs = np.array([0.5, -0.5, 0.5, -0.5])
    beta_coeffs  = np.array([0.5, -0.5, -0.5, 0.5])
    for i in range(k):
        col_base = 4 * i
        B[i, col_base:col_base+4] += alpha_coeffs
        if i <= k-2:
            B[i+1, col_base:col_base+4] += beta_coeffs
        if i == k-1:
            B[k, col_base:col_base+4] += beta_coeffs
    return B

B_U = build_B_matrix(Kbits)

def de_kron(w):
    """De-Kronecker via SVD: return u (33,), v (33,) such that kron(u,v) ≈ w."""
    M = np.reshape(w, (N_U, N_U), order='C')
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    sigma1 = S[0]
    if sigma1 <= 0:
        return np.zeros(N_U), np.zeros(N_U)
    u = np.sqrt(sigma1) * U[:, 0]
    v = np.sqrt(sigma1) * Vt[0, :]
    # fix sign to best match w
    w_recon = np.kron(u, v)
    if np.linalg.norm(w_recon - w) > np.linalg.norm(np.kron(-u, -v) - w):
        u = -u
        v = -v
    return u, v

def u_to_delays(u, k=Kbits):
    """Deterministic inversion from u (33,) to delays a,b,c,d each (32,)."""
    if u.shape[0] != k + 1:
        raise ValueError("u must be length k+1")
    alpha = np.zeros(k, dtype=float)
    beta  = np.zeros(k, dtype=float)
    alpha[0] = u[0]
    if k > 1:
        alpha[1:k] = u[1:k]
    beta[:-1] = 0.0
    beta[-1] = u[k]
    a = np.zeros(k, dtype=float)
    b = np.zeros(k, dtype=float)
    c = np.zeros(k, dtype=float)
    d = np.zeros(k, dtype=float)
    for i in range(k):
        d1 = alpha[i] + beta[i]
        d2 = alpha[i] - beta[i]
        a[i] = max(d1, 0.0)
        b[i] = max(-d1, 0.0)
        c[i] = max(d2, 0.0)
        d[i] = max(-d2, 0.0)
    return a, b, c, d

def build_A_for_fixed_v(v, B=B_U):
    """Build A (1089 x 128) when v is fixed: columns = kron(B[:,j], v)."""
    ncols = B.shape[1]
    A = np.zeros((N_U * N_U, ncols), dtype=float)
    for j in range(ncols):
        bj = B[:, j]
        A[:, j] = np.kron(bj, v)
    return A

def build_A_for_fixed_u(u, B=B_U):
    """Build A (1089 x 128) when u is fixed: columns = kron(u, B[:,j])."""
    ncols = B.shape[1]
    A = np.zeros((N_U * N_U, ncols), dtype=float)
    for j in range(ncols):
        bj = B[:, j]
        A[:, j] = np.kron(u, bj)
    return A

def projected_gradient_nnls(A, w, x0=None, max_iters=200, tol=1e-6):
    """Projected gradient for NNLS: min ||A x - w||^2 subject to x>=0."""
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float).copy()
    try:
        s = np.linalg.svd(A, compute_uv=False)
        L = float(s[0])**2
    except Exception:
        L = 1.0
    if L <= 0:
        L = 1.0
    eta = 1.0 / L
    At = A.T
    prev_obj = np.inf
    for _ in range(max_iters):
        grad = At @ (A @ x - w)
        x -= eta * grad
        np.maximum(x, 0.0, out=x)
        obj = np.linalg.norm(A @ x - w)
        if abs(prev_obj - obj) < tol * (1 + prev_obj):
            break
        prev_obj = obj
    return x

def my_decode(w):
    """
    Required API for grading.
    Input:
      w: 1D numpy array length 1089
    Output:
      tuple (a,b,c,d, p,q,r,s) each numpy array shape (32,) with non-negative entries.
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.size != W_LEN:
        raise ValueError(f"Input w must be length {W_LEN}")
    # de-Kronecker via SVD
    u_hat, v_hat = de_kron(w)
    # deterministic inversion
    a, b, c, d = u_to_delays(u_hat)
    p, q, r, s = u_to_delays(v_hat)
    # optional refinement via alternating NNLS (disabled by default)
    if DO_REFINE:
        x1 = np.vstack([a, b, c, d]).T.reshape(-1)
        x2 = np.vstack([p, q, r, s]).T.reshape(-1)
        for _ in range(REFINE_ITERS):
            v_curr = B_U @ x2
            A1 = build_A_for_fixed_v(v_curr, B_U)
            x1 = projected_gradient_nnls(A1, w, x0=x1, max_iters=SUB_ITERS)
            u_curr = B_U @ x1
            A2 = build_A_for_fixed_u(u_curr, B_U)
            x2 = projected_gradient_nnls(A2, w, x0=x2, max_iters=SUB_ITERS)
        x1 = np.maximum(x1, 0.0); x2 = np.maximum(x2, 0.0)
        first = x1.reshape(Kbits, 4); second = x2.reshape(Kbits, 4)
        a = first[:,0].copy(); b = first[:,1].copy(); c = first[:,2].copy(); d = first[:,3].copy()
        p = second[:,0].copy(); q = second[:,1].copy(); r = second[:,2].copy(); s = second[:,3].copy()
    # ensure non-negativity
    a = np.maximum(a, 0.0); b = np.maximum(b, 0.0)
    c = np.maximum(c, 0.0); d = np.maximum(d, 0.0)
    p = np.maximum(p, 0.0); q = np.maximum(q, 0.0)
    r = np.maximum(r, 0.0); s = np.maximum(s, 0.0)
    return a, b, c, d, p, q, r, s

# -----------------------
# Helpers (for local testing & CSV output)
# -----------------------
def delays_to_u(a,b,c,d):
    k = a.size
    u = np.zeros(k+1, dtype=float)
    alpha = (a - b + c - d) / 2.0
    beta  = (a - b - c + d) / 2.0
    u[0] = alpha[0]
    for i in range(1, k):
        u[i] = alpha[i] + beta[i-1]
    u[k] = beta[k-1]
    return u

def delays_to_w(a,b,c,d,p,q,r,s):
    u = delays_to_u(a,b,c,d)
    v = delays_to_u(p,q,r,s)
    return np.kron(u, v)

def load_models(path):
    path = str(path)
    if path.endswith('.npy'):
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.size == W_LEN:
            return arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] == W_LEN:
            return arr
        raise ValueError("Loaded .npy must have shape (N,1089) or (1089,)")
    if path.endswith('.npz'):
        data = np.load(path)
        if 'models' in data:
            arr = data['models']
        else:
            arr = data[list(data.files)[0]]
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.size == W_LEN:
            return arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] == W_LEN:
            return arr
        raise ValueError("Loaded .npz does not contain (N,1089) array")
    arr = np.loadtxt(path, delimiter=None)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != W_LEN:
        raise ValueError("Text/CSV file must have 1089 numbers per line")
    return arr

def save_delays_csv_only(delays_all, base_out):
    np.savetxt(base_out + '.csv', delays_all, delimiter=',')
    print(f"Saved CSV: {base_out}.csv")

def process_file(input_path, output_base="delays_output"):
    models = load_models(input_path)
    N = models.shape[0]
    delays_all = np.zeros((N, TOTAL_DELAYS), dtype=float)
    recon_errors = np.zeros(N, dtype=float)
    timings = np.zeros(N, dtype=float)
    for i in range(N):
        w = models[i].ravel()
        t0 = time.time()
        a,b,c,d,p,q,r,s = my_decode(w)
        t1 = time.time()
        timings[i] = t1 - t0
        first = np.vstack([a,b,c,d]).T.reshape(-1)
        second = np.vstack([p,q,r,s]).T.reshape(-1)
        delays256 = np.concatenate([first, second])
        delays_all[i] = delays256
        w_rec = delays_to_w(a,b,c,d,p,q,r,s)
        recon_errors[i] = np.linalg.norm(w - w_rec)
        print(f"Model {i+1}/{N}: time={timings[i]:.4f}s, recon_error={recon_errors[i]:.3e}, min_delay={delays256.min():.3e}, max_delay={delays256.max():.3e}")
    save_delays_csv_only(delays_all, output_base)
    print("Summary: avg_time={:.4f}s, avg_recon_err={:.3e}".format(timings.mean(), recon_errors.mean()))
    return delays_all, recon_errors, timings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert XOR Arbiter PUF linear models to non-negative delays.")
    parser.add_argument('--input', '-i', required=True, help="Input models file (.npy, .npz, .txt/.csv). If .npy it should be shape (N,1089).")
    parser.add_argument('--output', '-o', default='delays_output', help="Output base name (default 'delays_output').")
    parser.add_argument('--no-refine', action='store_true', help="Disable alternating refinement (faster).")
    args = parser.parse_args()
    if args.no_refine and DO_REFINE:
        print("Note: DO_REFINE was set True in file; --no-refine will not override. Edit DO_REFINE in file to permanently change.")
    process_file(args.input, args.output)
