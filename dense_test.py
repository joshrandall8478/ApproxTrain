import numpy as np
import tensorflow as tf
from python.keras.layers.amdenselayer import denseam
from models.resnet import build_resnet_cifar

# Build the ResNet model (e.g., ResNet20 for CIFAR)
model = build_resnet_cifar(input_shape=(32, 32, 3), num_classes=10, depth=20, lut_file='', FPMode="FP8HYB")

# Extract configurations of AMDense layers, rebuild layers, and also rebuild corresponding Dense layers
AMDense_layers = []
Dense_layers = []
input_shapes = []

for layer in model.layers:
    if isinstance(layer, denseam):
        config = layer.get_config()
        # Rebuild AMDense from config
        config.pop('kernel_regularizer', None)
        AMDense = denseam(**config)
        
        # Create a Dense layer from the same config (minus AMDense-specific args)
        dense_config = config.copy()
        dense_config.pop('mant_mul_lut', None)
        dense_config.pop('FPMode', None)
        Dense = tf.keras.layers.Dense(**dense_config)

        AMDense_layers.append(AMDense)
        Dense_layers.append(Dense)
        shape = layer.input_shape
        # replace None with 128, assuming batch size
        shape = [128 if x is None else x for x in shape]
        input_shapes.append(shape)

# Iterate over AMDense layers, Dense layers and input shapes
for AMDense, Dense, input_shape in zip(AMDense_layers, Dense_layers, input_shapes):
    # Initialize weights of AMDense and Dense layers
    AMDense.build(input_shape)
    Dense.build(input_shape)
    
    # Set weights of AMDense layer to Dense layer
    Dense.set_weights(AMDense.get_weights())
    
    # Generate random input
    x = np.random.rand(*input_shape)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Check if weights of AMDense and Dense are the same
    np.testing.assert_allclose(AMDense.get_weights()[0], Dense.get_weights()[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(AMDense.get_weights()[1], Dense.get_weights()[1], rtol=1e-5, atol=1e-5)

    # Get outputs
    y_amdense = AMDense(x)
    y_dense = Dense(x)

    # Compare the outputs
    try:
        np.testing.assert_allclose(y_amdense, y_dense, rtol=1e-5, atol=1e-5)
        print(f"Input shape: {input_shape}, Output shape: {y_amdense.shape}")
        print("Outputs match within the specified tolerance.")
    except AssertionError as e:
        print(f"Input shape: {input_shape}, Output shape: {y_amdense.shape}")
        print("Warning: Outputs do not match within the specified tolerance.")
        print(e)

    # Create a random upstream gradient for both AMDense and Dense
    random_grad = tf.random.normal(tf.shape(y_amdense))

    # Compute gradients for AMDense (input and weights)
    with tf.GradientTape(persistent=True) as tape_am:
        tape_am.watch(x)
        y_amdense = AMDense(x)
    grad_input_amdense = tape_am.gradient(y_amdense, x, output_gradients=random_grad)
    grad_weights_amdense = tape_am.gradient(y_amdense, AMDense.trainable_variables, output_gradients=random_grad)
    del tape_am

    # Compute gradients for Dense (input and weights)
    with tf.GradientTape(persistent=True) as tape_d:
        tape_d.watch(x)
        y_dense = Dense(x)
    grad_input_dense = tape_d.gradient(y_dense, x, output_gradients=random_grad)
    grad_weights_dense = tape_d.gradient(y_dense, Dense.trainable_variables, output_gradients=random_grad)
    del tape_d

    # Compare input gradients
    try:
        np.testing.assert_allclose(grad_input_amdense, grad_input_dense, rtol=1e-5, atol=1e-5)
        print(f"Input shape: {input_shape}, Output shape: {y_amdense.shape}")
        print("Input gradients match within the specified tolerance.")
    except AssertionError as e:
        print(f"Input shape: {input_shape}, Output shape: {y_amdense.shape}")
        print("Warning: Input gradients do not match within the specified tolerance.")
        print(e)
        diff = np.abs(grad_input_amdense - grad_input_dense)
        max_diff = np.max(diff)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Max diff (input grad): {max_diff} at index {max_diff_idx}")
        print(f"AMDense input gradient: {grad_input_amdense[max_diff_idx]}")
        print(f"Dense input gradient: {grad_input_dense[max_diff_idx]}")

    # Compare weight gradients
    # Both should have the same number of trainable variables (kernel, bias)
    for i, (gw_am, gw_d) in enumerate(zip(grad_weights_amdense, grad_weights_dense)):
        try:
            np.testing.assert_allclose(gw_am, gw_d, rtol=1e-5, atol=1e-5)
            print(f"Weight gradient {i} matches within the specified tolerance.")
        except AssertionError as e:
            print(f"Warning: Weight gradient {i} does not match within the specified tolerance.")
            print(e)
            diff = np.abs(gw_am - gw_d)
            max_diff = np.max(diff)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Max diff (weight grad {i}): {max_diff} at index {max_diff_idx}")
            print(f"AMDense weight gradient: {gw_am[max_diff_idx]}")
            print(f"Dense weight gradient: {gw_d[max_diff_idx]}")
