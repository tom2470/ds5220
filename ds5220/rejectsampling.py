# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def target_distribution(x):
    # Example target distribution, you can replace this with your own distribution
    return np.sin(x) + 2

def envelope_distribution(x):
    # Example envelope distribution (the proposal distribution), you can replace this with your own distribution
    return norm.pdf(x, loc=0, scale=3)

def reject_sampling(target_pdf, envelope_pdf, envelope_M, num_samples):
    accepted_samples = []
    rejected_samples = []

    for _ in range(num_samples):
        # Sample from the envelope distribution
        x = np.random.normal(0, 3)

        # Calculate acceptance probability
        acceptance_prob = target_pdf(x) / (envelope_M * envelope_pdf(x))

        # Accept or reject based on a uniform random value
        if np.random.rand() < acceptance_prob:
            accepted_samples.append(x)
        else:
            rejected_samples.append(x)

    return np.array(accepted_samples), np.array(rejected_samples)

# Example usage
target_x = np.linspace(-5, 5, 1000)
target_y = target_distribution(target_x)

plt.figure(figsize=(8, 6))
plt.plot(target_x, target_y, label='Target Distribution')
plt.title('Original Target Distribution')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Set up the envelope distribution
envelope_M = 2.5  # Adjust this value based on your knowledge of the target distribution
envelope_x = np.linspace(-5, 5, 1000)
envelope_y = envelope_distribution(envelope_x)

plt.figure(figsize=(8, 6))
plt.plot(envelope_x, envelope_y, label='Envelop Distribution')
plt.title('Envelope Distribution')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Perform reject sampling
accepted_samples, _ = reject_sampling(target_distribution, envelope_distribution, envelope_M, num_samples=1000)

# Plot the sampling distribution
plt.figure(figsize=(8, 6))
plt.hist(accepted_samples, bins=30, density=True, alpha=0.7, label='Sampling Distribution')
plt.title('Sampling Distribution using Reject Sampling')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
