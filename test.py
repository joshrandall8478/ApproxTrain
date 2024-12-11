# simple_train.py

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import random
from models.resnet import build_resnet_cifar
# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



# Prepare the model
input_shape = x_train.shape[1:]
model = build_resnet_cifar(input_shape=input_shape, num_classes=10, depth=20, lut_file='', FPMode="FP8E5M2")
# model = build_resnet20(input_shape=input_shape, num_classes=10)
model.summary()
# Define loss function and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Prepare the metrics
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()

# Training loop with gradient observation
batch_size = 128
epochs = 2  # Set to 1 for simplicity
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=50000, seed=random_seed).batch(batch_size)

last_gradients = None  # Variable to store the last gradients

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(x_batch, training=True)
            # Compute loss
            loss_value = loss_fn(y_batch, logits)
        # Compute gradients
        gradients = tape.gradient(loss_value, model.trainable_variables)
        
        # Observe gradients (for example, print the norm of the gradients)
        grad_norms = [tf.norm(g).numpy() for g in gradients]
        print(f"Step {step}, Loss: {loss_value.numpy():.4f}, Gradient Norms: {grad_norms[:3]}...")
        
        # Store the gradients (will overwrite at each step)
        last_gradients = [g.numpy() for g in gradients]
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(loss_value)
        train_accuracy.update_state(y_batch, logits)
    
    # Display metrics at the end of each epoch
    print(f"Epoch {epoch+1}, Loss: {train_loss.result().numpy():.4f}, Accuracy: {train_accuracy.result().numpy():.4f}")
    # Reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

# After training, save the last gradients to a text file
with open('last_gradients.txt', 'w') as f:
    for var, grad in zip(model.trainable_variables, last_gradients):
        f.write(f"Variable: {var.name}\n")
        np.savetxt(f, grad.flatten(), newline=' ', fmt='%.15f')
        f.write('\n\n')
print("Last gradients have been saved to 'last_gradients.txt'.")
