def LeNet(x):    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6. Relu Activation.
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    # Layer 2: Convolutional. Output = 10x10x16. Relu Activation.
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    # Flatten. Input = 5x5x16. Output = 400.
    # Layer 3: Fully Connected. Input = 400. Output = 120. Activation
    # Layer 4: Fully Connected. Input = 120. Output = 84. Activation.
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    
    conv1 = conv_layer(x, 5, 3, 6, name="conv1")
    pool1 = maxpool2x2_layer(conv1, name="maxpool1")
    conv2 = conv_layer(pool1, 5, 6, 16, name="conv2")
    pool2 = maxpool2x2_layer(conv2, name="maxpool2")
    f0 = flatten(pool2)
    f1 = fc_layer(f0, 400, 120, name="fc1", relu = 1)
    f2 = fc_layer(f1, 120, 84, name="fc2", relu = 1)
    f3 = fc_layer(f2, 84, 43, name="fc3", relu = 0)

    return(f3)

def LeNetDOL(x):  
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6. Relu Activation.
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    # Layer 2: Convolutional. Output = 10x10x16. Relu Activation.
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    # Flatten. Input = 5x5x16. Output = 400.
    # Layer 3: Fully Connected. Input = 400. Output = 120. Activation
    # Layer 4: Dropout layer
    # Layer 5: Fully Connected. Input = 120. Output = 84. Activation.
    # Layer 6: Dropout layer
    # Layer 7: Fully Connected. Input = 84. Output = 43.
    
    conv1 = conv_layer(x, 5, 3, 6, name="conv1")
    pool1 = maxpool2x2_layer(conv1, name="maxpool1")
    conv2 = conv_layer(pool1, 5, 6, 16, name="conv2")
    pool2 = maxpool2x2_layer(conv2, name="maxpool2")
    f0 = flatten(pool2)
    f1 = fc_layer(f0, 400, 120, name="fc1", relu = 1)
    f1 = DO_layer(f1, prob = keep_prob, name ="DOL1")
    f2 = fc_layer(f1, 120, 84, name="fc2", relu = 1)
    f2 = DO_layer(f2, prob = keep_prob, name ="DOL2"):
    f3 = fc_layer(f2, 84, 43, name="fc3", relu = 0)

    return(f3)


    