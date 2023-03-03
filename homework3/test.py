from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
correct_samples = np.random.normal(0, 1, size=1000)
leakage_samples = np.random.normal(2, 1, size=1000)

# Perform the t-test
t, p = stats.ttest_ind(correct_samples, leakage_samples, equal_var=False)

# Plot the results
fig, ax = plt.subplots()
ax.hist(correct_samples, alpha=0.5, label='Correct')
ax.hist(leakage_samples, alpha=0.5, label='Leakage')
ax.legend()
ax.set_title('Sample Distributions')
ax.set_xlabel('Value')
ax.set_ylabel('Count')

fig, ax = plt.subplots()
ax.bar(['Correct', 'Leakage'], [np.mean(correct_samples), np.mean(leakage_samples)], yerr=[np.std(correct_samples), np.std(leakage_samples)], capsize=10)
ax.set_title('Means of Sample Distributions')
ax.set_ylabel('Value')

fig, ax = plt.subplots()
ax.bar(['t-statistic'], [t], yerr=[np.std(correct_samples)], capsize=10)
ax.axhline(y=0, color='black')
ax.set_title('t-test Results')
ax.set_ylabel('t-statistic')

plt.show()
