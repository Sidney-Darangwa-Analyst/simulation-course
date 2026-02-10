# Test script to verify Python and packages are working
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data (Monte Carlo style)
np.random.seed(42)  # For reproducibility
samples = np.random.normal(loc=100, scale=15, size=1000)

# Create a DataFrame
df = pd.DataFrame({'values': samples})

# Print summary statistics
print("Summary Statistics:")
print(df.describe())

# Create a histogram
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of 1000 Random Samples')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('test_histogram.png')
plt.show()

print("\nHistogram saved as 'test_histogram.png'")
print("Python setup is complete!")