# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 22:49:25 2025

@author: Nicole
"""

import numpy as np
import matplotlib.pyplot as plt

# Observed data
num_customers = 800  # Total customers surveyed
num_satisfied = 680   # Customers who reported being satisfied

# Prior hyperparameters (Uniform prior Beta(1,1))
prior_alpha = 1
prior_beta = 1

# Posterior parameters update
posterior_alpha = prior_alpha + num_satisfied
posterior_beta = prior_beta + (num_customers - num_satisfied)

# Generate samples from the posterior Beta distribution
posterior_samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)

# Plot the posterior distribution
plt.figure(figsize=(10, 6))
plt.hist(posterior_samples, bins=30, density=True, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(posterior_samples.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title('Posterior Distribution of Customer Satisfaction Rate')
plt.xlabel('Satisfaction Rate')
plt.ylabel('Density')
plt.legend()
plt.show()

# Calculate summary statistics
mean_satisfaction = posterior_alpha / (posterior_alpha + posterior_beta)
mode_satisfaction = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)

# Display results
print("Mean satisfaction rate:", mean_satisfaction)
print("Mode satisfaction rate:", mode_satisfaction)
