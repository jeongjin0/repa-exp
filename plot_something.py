import matplotlib.pyplot as plt

# Data
methods = ['flow', 'REPA', 'independent']
steps = [40000, 80000, 100000]
values = [
    [30.41, 11.53, 8.17],
    [27.65, 10.46, 9.11],
    [28.04, 10.67, 10.13]
]

# Plot
plt.figure(figsize=(6,4))
for method, val in zip(methods, values):
    plt.plot(steps, val, marker='o', label=method)

plt.xlabel('Steps')
plt.ylabel('FID')
plt.ylim(0, 30)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
output_path = "fid_comparison.png"
plt.savefig(output_path, dpi=300)
plt.close()
