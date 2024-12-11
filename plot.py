import numpy as np
import matplotlib.pyplot as plt

# Load the last 10 gradients
all_gradients = np.load('last_10_gradients.npy', allow_pickle=True)

# Compute gradient norms for the first variable over the last 10 steps
grad_norms = [np.linalg.norm(step_grads[0]) for step_grads in all_gradients]

# Plot the gradient norms
plt.plot(range(len(grad_norms)), grad_norms)
plt.title('Gradient Norms of the First Variable Over Last 10 Steps')
plt.xlabel('Training Step (relative to last 10 steps)')
plt.ylabel('Gradient Norm')
# Save the plot
plt.savefig('gradient_norms.png')
plt.show()
