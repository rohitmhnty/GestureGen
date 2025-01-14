import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

def sample_t_fast(num_samples, a=2, t_min=1e-5, t_max=1-1e-5):
    # Direct inverse sampling for exponential distribution
    C = a / (np.exp(a) - 1)
    
    # Generate uniform samples
    u = torch.rand(num_samples * 2)
    
    # Inverse transform sampling formula for the exponential PDF
    # F^(-1)(u) = (1/a) * ln(1 + u*(exp(a) - 1))
    t = (1/a) * torch.log(1 + u * (np.exp(a) - 1))
    
    # Combine t and 1-t
    t = torch.cat([t, 1 - t])
    
    # Random permutation and slice
    t = t[torch.randperm(t.shape[0])][:num_samples]
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def sample_t(num_samples):
    # uniform sampling from 0 to 1
    t = torch.rand(num_samples)
    return t

def sample_mode(num_samples, s=0.0, t_min=1e-5, t_max=1-1e-5, device='cpu'):
    """
    Mode sampling with heavy tails.
    Args:
        num_samples: Number of samples to generate
        s: Scale parameter (-1 ≤ s ≤ 2/(π-2))
        t_min, t_max: Range limits to avoid numerical issues
    """
    # Generate uniform samples
    u = torch.rand(num_samples, device=device)
    
    # Apply the mode sampling function
    pi_half = torch.pi / 2
    cos_term = torch.cos(pi_half * u)
    t = 1 - u - s * (cos_term * cos_term - 1 + u)
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t


def sample_cosmap(num_samples, t_min=1e-5, t_max=1-1e-5, device='cpu'):
    """
    CosMap sampling.
    Args:
        num_samples: Number of samples to generate
        t_min, t_max: Range limits to avoid numerical issues
    """
    # Generate uniform samples
    u = torch.rand(num_samples, device=device)
    
    # Apply the cosine mapping
    pi_half = torch.pi / 2
    t = 1 - 1 / (torch.tan(pi_half * u) + 1)
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t
    

def sample_logits_t(num_samples, a=2):
    t = torch.sigmoid(torch.randn((num_samples,), dtype=torch.float32))
    return t

import time

def compare_sampling_distributions(num_samples=50000, save_dir='.', exponential_a=2, mode_s=0.5, device='cpu'):
    """Compare all sampling methods with separate plots"""
    
    # Set style for publication-quality figures
    plt.style.use('seaborn-whitegrid')
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    
    # Generate samples from all methods
    samples = {
        'Uniform': sample_t(num_samples).numpy(),
        'Exponential': sample_t_fast(num_samples, a=exponential_a).numpy(),
        'Logit-Normal': sample_logits_t(num_samples).numpy(),
        'Mode': sample_mode(num_samples, s=mode_s, device=device).cpu().numpy(),
        'CosMap': sample_cosmap(num_samples, device=device).cpu().numpy()
    }

    # Create individual plots
    plt.figure(figsize=(20, 8))
    
    # Plot settings
    plot_params = {
        'Uniform': {'color': '#1f77b4', 'title': 'Uniform Sampling'},
        'Exponential': {'color': '#ff7f0e', 'title': f'Exponential Sampling (a={exponential_a})'},
        'Logit-Normal': {'color': '#2ca02c', 'title': 'Logit-Normal Sampling'},
        'Mode': {'color': '#d62728', 'title': f'Mode Sampling (s={mode_s})'},
        'CosMap': {'color': '#9467bd', 'title': 'CosMap Sampling'}
    }
    
    for idx, (name, data) in enumerate(samples.items(), 1):
        plt.subplot(1, 5, idx)
        
        # Create histogram
        hist, bins, _ = plt.hist(data, bins=50, density=True, 
                               alpha=0.7, 
                               color=plot_params[name]['color'],
                               label='Samples')
        
        # Calculate mean and std
        mean = np.mean(data)
        std = np.std(data)
        
        plt.title(plot_params[name]['title'])
        plt.xlabel('t')
        plt.ylabel('Density' if idx == 1 else '')
        
        # Add text with statistics
        plt.text(0.05, 0.95, f'μ = {mean:.3f}\nσ = {std:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(hist) * 2)  # Give some headroom for text

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sampling_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate and return statistics
    stats = {}
    for name, data in samples.items():
        stats[name] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            '25th': np.percentile(data, 25),
            '75th': np.percentile(data, 75)
        }

    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 100)
    print(f"{'Method':<15} {'Mean':>10} {'Std':>10} {'Median':>10} {'25th':>10} {'75th':>10} {'Min':>10} {'Max':>10}")
    print("-" * 100)
    for method, stat in stats.items():
        print(f"{method:<15} {stat['mean']:10.4f} {stat['std']:10.4f} {stat['median']:10.4f} "
              f"{stat['25th']:10.4f} {stat['75th']:10.4f} {stat['min']:10.4f} {stat['max']:10.4f}")
    print("-" * 100)

    return stats

if __name__ == "__main__":
    results = compare_sampling_distributions(
        num_samples=50000,
        save_dir='.',
        exponential_a=2,
        mode_s=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

