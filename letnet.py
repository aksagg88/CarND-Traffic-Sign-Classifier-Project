def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6
    # weight shape = patch width x pathc height x input feature depth x fiter size
    # bias shape is same as filter output size
    # 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides = [1,1,1,1], padding = 'VALID') + conv1_b

    # TODO: Activation.
    # relu activation function
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape = (5,5,6,16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,conv2_W, strides = [1,1,1,1], padding = 'VALID') + conv2_b


    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    f0 = flatten(conv2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    f1_w = tf.Variable(tf.truncated_normal(shape =(400,120), mean = mu, stddev = sigma))
    f1_b = tf.Variable(tf.zeros(120))
    f1 = tf.matmul(f0,f1_w) + f1_b  

    # TODO: Activation.
    f1 = tf.nn.relu(f1)

    # ADDITION: Drouput
    #keep_prob = tf.placeholder(tf.float32) # probability to keep units
    #f1 = tf.nn.dropout(f1, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    f2_w  = tf.Variable(tf.truncated_normal(shape =(120,84), mean = mu, stddev = sigma))
    f2_b = tf.Variable(tf.zeros(84))
    f2 = tf.matmul(f1, f2_w) + f2_b


    # TODO: Activation.
    f2 = tf.nn.relu(f2)

    # ADDITION: Drouput
    #f2 = tf.nn.dropout(f2, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    f3_w = tf.Variable(tf.truncated_normal(shape =(84,43), mean = mu, stddev = sigma))
    f3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(f2, f3_w) + f3_b

    return(logits)