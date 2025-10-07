import torch
import numpy as np
import pandas as pd
from typing import Dict

# Function for the Gaussian distribution
def gaussian_distribution(y, mu, sigma):
    """
    Compute the Gaussian probability density function values for given y, mu, and sigma.
    
    This function was taken from Neural Hydrology [#]_ and adapted for our specific case. 
    
    Parameters:
    y (Tensor): Target values.
    mu (Tensor): Means of the Gaussian components.
    sigma (Tensor): Standard deviations of the Gaussian components.
    
    Returns:
    Tensor: The probability density values for the Gaussian distribution.
    """
    
    # Avoid dividing by zero by adding a small epsilon to sigma
    sigma = sigma + 1e-8

    # Compute the exponent part: exp(-0.5 * ((y - mu) / sigma)^2)
    result = -0.5 * ((y - mu) / sigma)**2

    # Return the Gaussian PDF: exp(result) / (sigma * sqrt(2 * pi))
    return torch.exp(result) / (sigma * np.sqrt(2.0 * np.pi))

LOG_SQRT_TWO_PI = np.log(np.sqrt(2.0 * np.pi))  # log(1 / sqrt(2π))

def nll_loss(pi, mu, sigma, y):
    """
    Negative Log-Likelihood loss for Mixture Density Networks.
    
    Parameters:
    pi (Tensor): Mixture weights (batch_size, n_components) — should be softmaxed already.
    mu (Tensor): Means of each component (batch_size, n_components).
    sigma (Tensor): Standard deviations (batch_size, n_components), positive via softplus/exp.
    y (Tensor): Target values (batch_size, 1) or (batch_size,).
    
    Returns:
    Tensor: Scalar negative log-likelihood.
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # Expand y to match shape of (batch_size, num_mixtures)
    y = y.expand_as(mu)

    # Gaussian log-likelihood for each component
    log_prob = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - LOG_SQRT_TWO_PI

    # Weighted log-likelihood (log π_k + log N_k)
    weighted_log_prob = torch.log(pi + 1e-8) + log_prob  # +ε to avoid log(0)

    # Log-sum-exp across mixtures to get total log likelihood per sample
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)

    # Return average negative log-likelihood over the batch
    return -torch.mean(log_sum)
