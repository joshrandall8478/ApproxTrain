import numpy as np
import tensorflow as tf
from python.keras.layers.am_convolutional import AMConv2D
from models.resnet import build_resnet_cifar

# Build the ResNet model (e.g., ResNet20 for CIFAR)
model = build_resnet_cifar(input_shape=(32, 32, 3), num_classes=10, depth=20, lut_file='', FPMode="FP32", AccumMode="SEARNE", trunk_size=16)

# Extract configurations of AMConv2D layers, rebuild layers, and also rebuild corresponding Conv2D layers
AMConv_layers = []
Conv2D_layers = []
input_shapes = []
for layer in model.layers:
    if isinstance(layer, AMConv2D):
        config = layer.get_config()
        
        # Rebuild AMConv2D from config (remove unsupported args)
        config.pop('kernel_regularizer', None)
        # set channel to 1
        config['filters'] = 1
        AMConv = AMConv2D(**config)

        # Create a Conv2D layer from the same config (minus AMConv2D-specific args)
        conv_config = config.copy()
        conv_config.pop('mant_mul_lut', None)
        conv_config.pop('FPMode', None)
        conv_config.pop('AccumMode', None)
        conv_config.pop('trunk_size', None)
        conv_config.pop('e4m3_exponent_bias', None)
        conv_config.pop('e5m2_exponent_bias', None)

        Conv2D_layer = tf.keras.layers.Conv2D(**conv_config)
        
        AMConv_layers.append(AMConv)
        Conv2D_layers.append(Conv2D_layer)
        
        # Replace None batch size with 128
        shape = layer.input_shape
        shape = [128 if x is None else x for x in shape]
        # Replace Channel with 1
        shape[3] = 10
        shape[1] = 3
        shape[2] = 3
        input_shapes.append(shape)
print_e = True
# Iterate over AMConv2D layers, Conv2D layers, and input shapes
for AMConv, Conv2D, input_shape in zip(AMConv_layers, Conv2D_layers, input_shapes):
    # Initialize weights of AMConv2D and Conv2D layers using same seed
    # set seed
    tf.random.set_seed(0)
    # set np seed
    np.random.seed(0)
    AMConv.build(input_shape)
    Conv2D.build(input_shape)
    
    # Set weights of AMConv2D layer to Conv2D layer (identical weights)
    Conv2D.set_weights(AMConv.get_weights())
    
    # Generate random input integer then convert to float 32
    # x = np.random.randint(0, 10, size=input_shape)
    # x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = np.random.rand(*input_shape)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Check if weights of AMConv2D and Conv2D are the same
    np.testing.assert_allclose(AMConv.get_weights(), Conv2D.get_weights(), rtol=1e-5, atol=1e-5)
    # print following configs:
    # 'filters': , 'kernel_size': , 'strides': , 'padding': , 'data_format':, 
    print(f"Input Size: {input_shape}, AMConv2D filters: {AMConv.filters}, kernel_size: {AMConv.kernel_size}, strides: {AMConv.strides}, padding: {AMConv.padding}, data_format: {AMConv.data_format}")

    # Get output of AMConv2D layer
    y_amconv = AMConv(x)
    # Get output of Conv2D layer
    y_conv2d = Conv2D(x)

    # Compare the outputs
    try:
        np.testing.assert_allclose(y_amconv, y_conv2d, rtol=1e-5, atol=1e-5)
        print("Outputs match within the specified tolerance.")
    except AssertionError as e:
        print("Warning: Outputs did not match within the specified tolerance.")
        if print_e:
            print(str(e))

    # Compute gradients with respect to inputs and weights using a random upstream gradient
    random_grad_amconv = None
    
    # Gradients for AMConv2D
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_amconv = AMConv(x)
        # Create a random upstream gradient for AMConv output
        random_grad_amconv = tf.random.normal(tf.shape(y_amconv))
        #random_grad_amconv = np.random.randint(0, 10, tf.shape(y_amconv))
        #random_grad_amconv = tf.convert_to_tensor(random_grad_amconv, dtype=tf.float32)
        
        # Grad w.r.t input
        grad_amconv = tape.gradient(y_amconv, x, output_gradients=random_grad_amconv)
        # Grad w.r.t weights
        grad_weights_amconv = tape.gradient(y_amconv, AMConv.trainable_variables, output_gradients=random_grad_amconv)
        
    # Gradients for Conv2D
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_conv2d = Conv2D(x)
        
        # Use the same upstream gradient for comparison
        grad_conv2d = tape.gradient(y_conv2d, x, output_gradients=random_grad_amconv)
        grad_weights_conv2d = tape.gradient(y_conv2d, Conv2D.trainable_variables, output_gradients=random_grad_amconv)

    # Compare input gradients
    try:
        np.testing.assert_allclose(grad_amconv, grad_conv2d, rtol=1e-5, atol=1e-5)
        print("Input Gradients match within the specified tolerance.")
    except AssertionError as e:
        print("Warning: Input Gradients did not match within the specified tolerance.")
        if print_e:
            print(str(e))
            diff = np.abs(grad_amconv.numpy() - grad_conv2d.numpy())
            # get index of all differences

            max_diff = np.max(diff)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Max difference (input grad): {max_diff} at index {max_diff_idx}")
            print(f"AMConv2D Input Gradient: {grad_amconv[max_diff_idx]}")
            print(f"Conv2D Input Gradient: {grad_conv2d[max_diff_idx]}")

    # Compare weight gradients
    # Both sets of weights should have the same number of variables (e.g., kernel and bias)
    for i, (g_am, g_c2) in enumerate(zip(grad_weights_amconv, grad_weights_conv2d)):
        try:
            np.testing.assert_allclose(g_am, g_c2, rtol=1e-5, atol=1e-5)
            print(f"Weight Gradient {i}: match within the specified tolerance.")
        except AssertionError as e:
            print(f"Weight Gradient {i}: did not match within the specified tolerance.")
            if print_e:
                print(str(e))
                diff = np.abs(g_am.numpy() - g_c2.numpy())
                max_diff = np.max(diff)
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"Max difference (weight grad {i}): {max_diff} at index {max_diff_idx}")
                print(f"AMConv2D Weight Gradient: {g_am[max_diff_idx]}")
                print(f"Conv2D Weight Gradient: {g_c2[max_diff_idx]}")
  
