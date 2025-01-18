import numpy as np
import matplotlib.pyplot as plt
import os

def f_mode(u, s):
    """
    Implements the mode sampling function f_mode(u; s).
    """
    return 1 - u - s * (np.cos(np.pi/2 * u)**2 - 1 + u)

def f_mode_inverse(t, s):
    """
    Numerically computes the inverse of f_mode for a given t and s.
    """
    from scipy.optimize import root_scalar
    
    def objective(u):
        return f_mode(u, s) - t
    
    try:
        result = root_scalar(objective, bracket=[0, 1], method='brentq')
        return result.root
    except:
        return np.nan

def uniform_pdf(x):
    """Uniform distribution density on [0,1]"""
    return np.ones_like(x)

def exponential_pdf(x, a=2):
    """Exponential distribution density with parameter a"""
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

def logit_normal_pdf(x, m=0, s=1):
    """
    Logit-normal distribution density:
    π_ln(t; m, s) = 1/(s√(2π)) * 1/(t(1-t)) * exp(-(logit(t)-m)^2/(2s^2))
    where logit(t) = log(t/(1-t))
    """
    logit = np.log(x/(1-x))
    return (1/(s * np.sqrt(2*np.pi))) * (1/(x*(1-x))) * np.exp(-(logit - m)**2/(2*s**2))

def mode_pdf(x, s=0.5, eps=1e-6):
    """
    Computes the mode sampling density π_mode(t; s).
    
    Parameters:
    - x: Input values (numpy array)
    - s: Scale parameter (float), must be in [-1, 2/(π-2)]
    - eps: Step size for numerical differentiation
    
    Returns:
    - Density values at input points x
    """
    # Validate s parameter
    s_max = 2/(np.pi - 2)
    if s < -1 or s > s_max:
        raise ValueError(f"s must be in [-1, {s_max}]")
    
    # Vectorize the computation
    densities = np.zeros_like(x, dtype=float)
    
    for i, t in enumerate(x):
        # Skip values too close to 0 or 1 for numerical stability
        if t < eps or t > 1 - eps:
            densities[i] = np.nan
            continue
            
        # Compute derivative using central differences
        t_plus = min(t + eps, 1 - eps)
        t_minus = max(t - eps, eps)
        
        u_plus = f_mode_inverse(t_plus, s)
        u_minus = f_mode_inverse(t_minus, s)
        
        if np.isnan(u_plus) or np.isnan(u_minus):
            densities[i] = np.nan
        else:
            # Absolute value of derivative
            densities[i] = abs((u_plus - u_minus) / (2 * eps))
    
    # Interpolate any NaN values
    mask = np.isnan(densities)
    if np.any(mask):
        x_valid = x[~mask]
        densities_valid = densities[~mask]
        from scipy.interpolate import interp1d
        f = interp1d(x_valid, densities_valid, bounds_error=False, fill_value='extrapolate')
        densities[mask] = f(x[mask])
    
    return densities

def lognormal_pdf(x, mu=0, sigma=1):
    """
    Lognormal distribution density function using NumPy.
    
    Parameters:
    - x: Input values (ndarray, must be positive).
    - mu: Mean of the log of the distribution (default: 0).
    - sigma: Standard deviation of the log of the distribution (default: 1).
    
    Returns:
    - Lognormal PDF evaluated at x (ndarray).
    """
    # Avoid division by zero for x <= 0
    x = np.clip(x, 1e-10, 1 - 1e-10)

    def logit(x):
        """Logit function: log(t / (1 - t))."""
        x = np.clip(x, 1e-10, 1 - 1e-10)  # Avoid division by zero or log of zero
        return np.log(x / (1 - x))

    logit_x = logit(x)
    # Compute the lognormal PDF
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi) * x * (1 - x))
    exponent = -0.5 * ((logit_x - mu) / sigma) ** 2
    pdf = coefficient * np.exp(exponent)
    return pdf

def cosmap_pdf(x):
    """
    CosMap density:
    π_CosMap(t) = 2/(π - 2πt + 2πt²)
    """
    return 2/(np.pi - 2*np.pi*x + 2*np.pi*x**2)

def beta_pdf(x, a=2, b=1.2):
    """
    Beta distribution density:
    π_Beta(t; a, b) = t^(a-1) * (1-t)^(b-1) / B(a, b)
    where B(a, b) is the beta function
    """
    from scipy.special import beta
    return x**(a-1) * (1-x)**(b-1) / beta(a, b)

def plot_theoretical_densities(save_dir='.', exponential_a=2, mode_s=0.5, logit_m=0, logit_s=1):
    plt.figure(figsize=(10, 8))
    
    # Generate x values for theoretical curves, avoiding exact 0 and 1
    x = np.linspace(0.001, 0.999, 1000)
    
    # Plot theoretical density functions
    plt.plot(x, uniform_pdf(x), '-', color='#1f77b4', label='Uniform', linewidth=2)
    # plt.plot(x, exponential_pdf(x, exponential_a), '-', color='#ff7f0e', label=f'Exponential (a={exponential_a})', linewidth=2)
    plt.plot(x, lognormal_pdf(x, logit_m, logit_s), '-', color='#2ca02c', label=f'Logit-Normal (m={logit_m}, s={logit_s})', linewidth=2)
    plt.plot(x, mode_pdf(x, mode_s), '-', color='#d62728', label=f'Mode (s={mode_s})', linewidth=2)
    plt.plot(x, cosmap_pdf(x), '-', color='#9467bd', label='CosMap', linewidth=2)
    plt.plot(x, beta_pdf(x), '-', color='#8c564b', label=f'Beta (a=2, b=1.2)', linewidth=2)
    
    # plt.title('Theoretical Sampling Distribution Densities', fontsize=14)
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=20)

    # Increase the size of tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'theoretical_densities.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_theoretical_densities(
        save_dir='.',
        exponential_a=2,
        mode_s=0.5,
        logit_m=0,
        logit_s=1
    )