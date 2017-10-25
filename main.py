import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

import time

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return startTime_for_tictoc

def toc():
    if 'startTime_for_tictoc' in globals():
        endTime = time.time()
        return endTime - startTime_for_tictoc
    else:
        return None

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    
    FCN-8 - Encoder in class lecture
    paper: "Fully Convolutional Networks for Semantic Segmentation"
    
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # laod the model from the given vgg_path
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # extract the layers of the vgg to modify into a FCN
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)    
    return w1,keep,w3,w4,w7 
tests.test_load_vgg(load_vgg, tf)


# # custom init with the seed set to 0 by default
def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
    return tf.random_normal(shape, dtype=dtype, seed=seed)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    
    FCN-8 - Decoder, skip connections
    
    """
    # FCN-8 - Decoder
    # To build the decoder portion of FCN-8, we’ll upsample the input to the
    # original image size.  The shape of the tensor after the final
    # convolutional transpose layer will be 4-dimensional:
    #    (batch_size, original_height, original_width, num_classes).    
    
    
    # TODO: Implement function
    init = tf.truncated_normal_initializer(stddev = 0.01)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    def conv_1x1(x, num_classes, init = init, reg = reg):
        return tf.layers.conv2d(x, num_classes, 1, padding = 'same', kernel_initializer = init ,kernel_regularizer=reg)
    
    def upsample(x, num_classes, depth, strides, init = init, reg = reg):
        return tf.layers.conv2d_transpose(x, num_classes, depth, strides, padding = 'same', kernel_initializer = init, kernel_regularizer=reg)
    
    layer_7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
    layer_4_1x1 = conv_1x1(vgg_layer4_out, num_classes)
    layer_3_1x1 = conv_1x1(vgg_layer3_out, num_classes)
    
    #implement the first transposed convolution layer
    layer1 = upsample(layer_7_1x1, num_classes, 4, 2)
    layer1 = tf.layers.batch_normalization(layer1)
    #add the first skip connection from the layer_4_1x1    
    layer1 = tf.add(layer1, layer_4_1x1)

    #implement the another transposed convolution layer
    layer2 = upsample(layer1, num_classes, 4, 2)
    layer2 = tf.layers.batch_normalization(layer2)
    #add the second skip connection from the layer_3_1x1        
    layer2 = tf.add(layer2, layer_3_1x1)
    
    output = upsample(layer2, num_classes, 16, 8) 
    
#     conv_1x1 = tf.layer.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
#     output = tf.layers.conv2d_transpose(conv1x1, num_class, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # upsampling by 2 by 2 then by 8. from skip connection in classroom or paper
    
    # # make sure the shapes are the same!
    # input = tf.add(input, pool_4)
    
    # input = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))
    
    # input = tf.add(input, pool_3)
    # Input = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8))
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    
    FCN-8 - Classification & Loss
    
    """
    # TODO: Implement function
    # reshape the 4D output and label tensors to 2D:
    # so each row represent a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    y = tf.reshape(correct_label, (-1, num_classes))
    
    # logits = tf.reshape(input, (-1, num_classes))
    
    # now define a loss function and a trainer/optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))  
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer()) 
    
    # start_time = tic()
    output_file = os.path.join('training_history.txt')
    thefile = open(output_file,'a')
    for epoch in range(epochs):
        # train on batches
        batch_num = 0
        for images, labels in get_batches_fn(batch_size):
            
            batch_num += 1
            tic()
            _, loss = sess.run([train_op, cross_entropy_loss],feed_dict={input_image: images, correct_label: labels, keep_prob:0.75, learning_rate:0.0001})
            thefile.write("Epoch {0}/{1}... Batch {2}... Training Loss: {3:.4f}... processing_time: {4:.1f} sec \n".format(epoch+1, epochs, batch_num, loss, toc()))
            print("Epoch {0}/{1}... Batch {2}... Training Loss: {3:.4f}... processing_time: {4:.1f} sec \n".format(epoch+1, epochs, batch_num, loss, toc()))
    # end_time = tic()
    # thefile.write("total training time: {:.2f} minute".format((end_time-start_time)/60))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    
    # training hyper parameters
    epochs = 50
    batch_size = 32
    lr = 0.0001
    learning_rate = tf.constant(lr)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        # TODO: Train NN using the train_nn function
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        shape = [None, image_shape[0], image_shape[1], 3]
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
