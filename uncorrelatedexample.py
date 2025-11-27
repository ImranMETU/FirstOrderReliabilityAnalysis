import math
import numpy as np

# ==========================================
# STEP 1 — Random variables + limit state g
# ==========================================

class NormalRV:
    def __init__(self, mean, cov):
        self.mean = mean
        self.std  = mean * cov      # physical std

class LognormalRV:
    """Lognormal defined by mean and cov in PHYSICAL space."""
    def __init__(self, mean, cov):
        self.mean = mean
        self.std  = mean * cov
        cv2 = (self.std / self.mean)**2
        self.sigma_y = math.sqrt(math.log(1.0 + cv2))
        self.mu_y    = math.log(mean) - 0.5 * self.sigma_y**2

def g_limit_state(x):
    """
    x[0] = Fy (MPa)
    x[1] = Zx (mm^3)

    Limit state:
        g(Fy, Zx) = Fy * Zx * 1e-6 - 1900   [kNm]
    Failure when g <= 0.
    """
    Fy, Zx = x
    return Fy * Zx * 1e-6 - 1900.0

# Instantiate RVs for your CE589 example
Fy = LognormalRV(mean=373.0, cov=0.10)   # Fy ~ LN(373 MPa, COV=0.10)
Zx = NormalRV   (mean=8772e3, cov=0.05)  # Zx ~ N(8.772e6 mm^3, COV=0.05)

# ==========================================
# STEP 2 — Initial reliability index β(0)
# ==========================================

beta_0 = 3.0    # standard starting value in FORM

# ==========================================
# STEP 3 — Initial checking point x* (0)
# ==========================================

# vector of means: x*0 = [μ_Fy, μ_Zx]
x_star_0 = np.array([Fy.mean, Zx.mean], dtype=float)

# Quick diagnostics
print("Step 2: initial beta")
print(f"  beta^(0) = {beta_0:.3f}\n")

print("Step 3: initial checking point (means)")
print(f"  Fy* (0) = {x_star_0[0]:.3f} MPa")
print(f"  Zx* (0) = {x_star_0[1]:.3f} mm^3")

g0 = g_limit_state(x_star_0)
print("\nLimit-state value at initial point:")
print(f"  g(Fy*(0), Zx*(0)) = {g0:.3f} kNm")


# ============================================================
# Standard-normal helpers + equivalent-normal transformation
# ============================================================

def equivalent_normal(Fy: LognormalRV, Zx: NormalRV, x_star: np.ndarray):
    """Compute equivalent-normal mu_N and sigma_N for Fy (LN) and Zx (N)

    Uses closed-form RF simplification for lognormal variables:
      ζ^2 = ln(1 + COV^2),  λ = ln(mean) - 0.5 ζ^2
      σ_N = ζ * x*,   μ_N = x* * (1 - ln(x*) + λ)

    Returns (mu_Fy_N, sigma_Fy_N, mu_Zx_N, sigma_Zx_N).
    """
    Fy_star, Zx_star = float(x_star[0]), float(x_star[1])

    # Lognormal: closed-form equivalence
    zeta = Fy.sigma_y
    lam = Fy.mu_y

    sigma_Fy_N = zeta * Fy_star
    mu_Fy_N = Fy_star * (1.0 - math.log(Fy_star) + lam)

    # Normal Zx: equivalent is same as physical
    mu_Zx_N = Zx.mean
    sigma_Zx_N = Zx.std

    return mu_Fy_N, sigma_Fy_N, mu_Zx_N, sigma_Zx_N


# Print equivalent-normal parameters at the initial checking point
muFyN, sigFyN, muZxN, sigZxN = equivalent_normal(Fy, Zx, x_star_0)

print("\nEquivalent-normal parameters at x*(0):")
print(f"  mu_Fy^N  = {muFyN:.6f},  sigma_Fy^N = {sigFyN:.6f}")
print(f"  mu_Zx^N  = {muZxN:.6f},  sigma_Zx^N = {sigZxN:.6f}")

# Note: Direction-cosine printing moved to module main guard below so
# the helper functions and `rvs` are defined before use.



# ------------------------------------------------------------------
# Closed-form batch equivalent using the same formulas but returning
# arrays for use inside the FORM loop (equivalent_normal_closed_form)
# ------------------------------------------------------------------
def equivalent_normal_closed_form(x_star: np.ndarray, rvs):
    """Return arrays mu_N, sigma_N for all RVs at checking point x_star.

    Uses closed-form RF equivalence for lognormal variables and passes
    through normal variables unchanged.
    """
    n = len(rvs)
    mu_N = np.zeros(n, dtype=float)
    sigma_N = np.zeros(n, dtype=float)

    for i, rv in enumerate(rvs):
        x_i = float(x_star[i])
        if isinstance(rv, LognormalRV):
            zeta = rv.sigma_y
            lam = rv.mu_y
            sigma_i = zeta * x_i
            mu_i = x_i * (1.0 - math.log(x_i) + lam)
        elif isinstance(rv, NormalRV):
            mu_i = rv.mean
            sigma_i = rv.std
        else:
            raise TypeError("Unsupported RV type in equivalent_normal_closed_form")

        mu_N[i] = mu_i
        sigma_N[i] = sigma_i

    return mu_N, sigma_N


# ------------------------------------------------------------------
# Step 5: gradient in physical space and direction cosines α
# ------------------------------------------------------------------
def grad_g_physical(x_star: np.ndarray) -> np.ndarray:
    """Gradient of g(Fy, Zx) in physical space at x_star.

    For g = Fy * Zx * 1e-6 - 1900,
    ∂g/∂Fy = Zx * 1e-6,  ∂g/∂Zx = Fy * 1e-6.
    """
    Fy_star, Zx_star = float(x_star[0]), float(x_star[1])
    dg_dFy = Zx_star * 1e-6
    dg_dZx = Fy_star * 1e-6
    return np.array([dg_dFy, dg_dZx], dtype=float)


def direction_cosines(x_star: np.ndarray, rvs):
    """Compute mu_N, sigma_N and direction cosines alpha at x_star.

    Returns (mu_N, sigma_N, alpha).
    """
    mu_N, sigma_N = equivalent_normal_closed_form(x_star, rvs)

    grad_x = grad_g_physical(x_star)
    grad_N = grad_x * sigma_N
    norm_grad_N = np.linalg.norm(grad_N)
    if norm_grad_N == 0.0:
        raise RuntimeError("Zero gradient in normal space.")
    alpha = grad_N / norm_grad_N
    return mu_N, sigma_N, alpha


def compute_alphas(x_star, sigma_N):
    """Compute direction cosines α from physical checking point and sigma_N.

    This is the compact form you provided: transform the physical gradient
    into normal-space derivatives and normalize.
    """
    Fy_star, Zx_star = x_star
    sigma_Fy_N, sigma_Zx_N = sigma_N

    # derivatives in normal space: (∂g/∂x_i) * σ_i^N
    dgdX_FyN = (Zx_star * 1e-6) * sigma_Fy_N
    dgdX_ZxN = (Fy_star * 1e-6) * sigma_Zx_N

    denom = math.sqrt(dgdX_FyN**2 + dgdX_ZxN**2)
    if denom == 0.0:
        raise RuntimeError("Zero norm when computing alphas")

    alpha_Fy = dgdX_FyN / denom
    alpha_Zx = dgdX_ZxN / denom

    return np.array([alpha_Fy, alpha_Zx], dtype=float)


def update_checking_point(x_star, beta, rvs):
    """
    One RF update (Step 4–6):
      - compute μ^N, σ^N (equivalent normals)
      - compute direction cosines α
      - update x* using x*_{k+1} = μ^N - α β σ^N
    Returns:
      x_star_new, mu_N, sigma_N, alpha
    """
    # Step 4: equivalent-normal parameters
    mu_N, sigma_N = equivalent_normal_closed_form(x_star, rvs)

    # Step 5: direction cosines (alpha)
    alpha = compute_alphas(x_star, sigma_N)

    # Step 6: update checking point
    x_star_new = mu_N - alpha * beta * sigma_N

    return x_star_new, mu_N, sigma_N, alpha


def solve_beta(mu_N, sigma_N, alpha, M=1900.0):
    """
    Solves for beta from the limit-state equation:
        (Fy* = μFyN - αFy β σFyN)
        (Zx* = μZxN - αZx β σZxN)
        Fy* Zx* 1e-6 - M = 0

    Returns the positive physical root for β.
    """
    mu_FyN, mu_ZxN = mu_N
    sigma_FyN, sigma_ZxN = sigma_N
    alpha_Fy, alpha_Zx = alpha

    A = mu_FyN
    B = alpha_Fy * sigma_FyN
    C = mu_ZxN
    D = alpha_Zx * sigma_ZxN

    K = M * 1e6

    a = B * D
    b = -(A * D + B * C)
    c = A * C - K

    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("Negative discriminant — check your inputs.")

    sqrt_disc = math.sqrt(disc)
    beta1 = (-b + sqrt_disc) / (2 * a)
    beta2 = (-b - sqrt_disc) / (2 * a)

    candidates = [beta1, beta2]
    beta_pos = [b for b in candidates if b > 0]
    if len(beta_pos) == 0:
        raise RuntimeError("No positive beta root found.")

    beta_new = min(beta_pos)
    return beta_new


def form_iteration(x_star_init, beta_init, rvs, tol=1e-6, max_iter=50, M=1900.0):
    """
    Run Rackwitz–Fiessler FORM iterations until convergence.

    At each iteration:
      - Step 4: compute μ^N, σ^N
      - Step 5: compute α
      - Step 6: update x* = μ^N - α β σ^N
      - Step 7: recompute β by solving the quadratic
      - Step 8: check convergence (change in β and x*)

    Returns (history, converged_flag) where history is a list of dicts
    containing per-iteration values.
    """
    history = []
    x_star = x_star_init.astype(float).copy()
    beta = float(beta_init)

    for k in range(1, max_iter + 1):
        mu_N, sigma_N = equivalent_normal_closed_form(x_star, rvs)
        alpha = compute_alphas(x_star, sigma_N)

        # store previous values for convergence check
        x_prev = x_star.copy()
        beta_prev = beta

        # compute equivalent normals and alpha for current x_star
        # (mu_N, sigma_N already computed above)

        # update checking point (uses current beta)
        x_star, mu_N, sigma_N, alpha = update_checking_point(x_star, beta, rvs)

        # recompute beta from the updated normal parameters
        beta = solve_beta(mu_N, sigma_N, alpha, M=M)

        # record iteration including previous and new values
        gval = g_limit_state(x_star)
        history.append({
            "iter": k,
            "beta_prev": beta_prev,
            "beta": beta,
            "x_prev": x_prev.copy(),
            "x_star": x_star.copy(),
            "mu_N": mu_N.copy(),
            "sigma_N": sigma_N.copy(),
            "alpha": alpha.copy(),
            "g": gval,
        })

        # convergence checks: absolute change in beta and relative change in x*
        dbeta = abs(beta - beta_prev)
        dx_rel = np.linalg.norm(x_star - x_prev) / (np.linalg.norm(x_prev) + 1e-12)

        if dbeta < tol and dx_rel < tol:
            return history, True

    return history, False


def export_history_csv(history, filename="form_history.csv"):
    """Export iteration history to CSV with a consistent column order."""
    import csv

    fieldnames = [
        "iter",
        "beta_prev",
        "beta",
        "Fy_prev",
        "Zx_prev",
        "muFyN",
        "sigmaFyN",
        "muZxN",
        "sigmaZxN",
        "alphaFy",
        "alphaZx",
        "g",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in history:
            writer.writerow({
                "iter": it["iter"],
                "beta_prev": f"{it['beta_prev']:.6f}",
                "beta": f"{it['beta']:.6f}",
                "Fy_prev": f"{it['x_prev'][0]:.6f}",
                "Zx_prev": f"{it['x_prev'][1]:.6f}",
                "muFyN": f"{it['mu_N'][0]:.6f}",
                "sigmaFyN": f"{it['sigma_N'][0]:.6f}",
                "muZxN": f"{it['mu_N'][1]:.6f}",
                "sigmaZxN": f"{it['sigma_N'][1]:.6f}",
                "alphaFy": f"{it['alpha'][0]:.6f}",
                "alphaZx": f"{it['alpha'][1]:.6f}",
                "g": f"{it['g']:.6f}",
            })


if __name__ == "__main__":
    # Define the RV list in the same order as x_star
    rvs = [Fy, Zx]

    # Compute direction cosines using both methods and print
    muN_arr, sigN_arr, alpha_dc = direction_cosines(x_star_0, rvs)
    alpha_compact = compute_alphas(x_star_0, sigN_arr)

    print("\nDirection cosines at x*(0):")
    print(f"  alpha (direction_cosines) : [{alpha_dc[0]:.6f}, {alpha_dc[1]:.6f}]")
    print(f"  alpha (compute_alphas)    : [{alpha_compact[0]:.6f}, {alpha_compact[1]:.6f}]")

    # Perform one RF update (Step 4-6) and print new values
    x_star_new, mu_N, sigma_N, alpha = update_checking_point(x_star_0, beta_0, rvs)

    print("\nAfter one update_checking_point:")
    print(f"  x_star_new                 : [{x_star_new[0]:.6f}, {x_star_new[1]:.6f}]")
    print(f"  mu_N                      : [{mu_N[0]:.6f}, {mu_N[1]:.6f}]")
    print(f"  sigma_N                   : [{sigma_N[0]:.6f}, {sigma_N[1]:.6f}]")
    print(f"  alpha                     : [{alpha[0]:.6f}, {alpha[1]:.6f}]")

    # Solve for beta given the updated normal parameters and alpha
    beta_new = solve_beta(mu_N, sigma_N, alpha, M=1900.0)
    print(f"\nSolved beta from limit-state: beta = {beta_new:.6f}")

    # Run full FORM iterations (Steps 4-9)
    hist, converged = form_iteration(x_star_0, beta_0, rvs, tol=1e-8, max_iter=50, M=1900.0)

    print("\nFORM Iterations:")
    # header: beta_prev | beta | Fy* (prev) | Zx* (prev) | muFyN | sigmaFyN | muZxN | sigmaZxN | alphaFy | alphaZx | g
    print(" it | beta_prev |   beta   |    Fy*     |      Zx*       | muFyN    | sigFyN   | muZxN        | sigZxN     | alphaFy  | alphaZx  |    g")
    for it in hist:
        i = it["iter"]
        b_prev = it["beta_prev"]
        b = it["beta"]
        x_prev = it["x_prev"]
        xs = it["x_star"]
        muN = it["mu_N"]
        sigN = it["sigma_N"]
        a = it["alpha"]
        gval = it["g"]

        print(f"{i:3d} | {b_prev:9.6f} | {b:9.6f} | {x_prev[0]:10.6f} | {x_prev[1]:14.6f} | {muN[0]:9.6f} | {sigN[0]:9.6f} | {muN[1]:13.6f} | {sigN[1]:11.6f} | {a[0]:8.6f} | {a[1]:8.6f} | {gval:9.3f}")

    print(f"\nConverged: {converged}, iterations = {len(hist)}")
    if len(hist) > 0:
        print(f"Final beta = {hist[-1]['beta']:.6f}")
        pf = 0.5 * (1.0 - math.erf(hist[-1]['beta'] / math.sqrt(2.0)))
        print(f"Estimated failure probability p_f = {pf:.6e}")

    # Export CSV for the history
    csv_name = "form_history.csv"
    export_history_csv(hist, filename=csv_name)
    print(f"\nWrote iteration history to: {csv_name}")
