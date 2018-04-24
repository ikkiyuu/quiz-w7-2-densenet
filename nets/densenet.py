"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

# def dense_block(net, growth, scope='dense_block'):
#     input_0 = net

#   	# 1st layer 
#     bottleneck = bn_act_conv_drp(input_0, 4 * growth, [1, 1], 
#                                 scope=scope + '_conv1x1' + '_1')
#     tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
#                                 scope=scope + '_conv3x3' + '_1')
#     input_1 = tf.concat(axis=3, values=[input_0, tmp])

#     # 2nd layer
#     bottleneck = bn_act_conv_drp(input_1, 4 * growth, [1, 1],
#                                        scope=scope + '_conv1x1' + '_2')
#     tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
#                                 scope=scope + '_conv3x3' + '_2')
#     input_2 = tf.concat(axis=3, values=[input_0, input_1, tmp])

#     # 3rd layer
#     bottleneck = bn_act_conv_drp(input_2, 4 * growth, [1, 1],
#                                        scope=scope + '_conv1x1' + '_3')
#     tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
#                                 scope=scope + '_conv3x3' + '_3')
#     input_3 = tf.concat(axis=3, values=[input_0, input_1, input_2, tmp])

#     # 4th layer
#     bottleneck = bn_act_conv_drp(input_3, 4 * growth, [1, 1],
#                                        scope=scope + '_conv1x1' + '_4')
#     tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
#                                 scope=scope + '_conv3x3' + '_4')
#     input_4 = tf.concat(axis=3, values=[input_0, input_1, input_2, input_3,tmp])

#     # 5th layer
#     bottleneck = bn_act_conv_drp(input_4, 4 * growth, [1, 1],
#                                        scope=scope + '_conv1x1' + '_5')
#     tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
#                                 scope=scope + '_conv3x3' + '_5')
#     output = tf.concat(axis=3, values=[input_0, input_1, input_2, input_3, input_4, tmp])

#     return output

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 12
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'Dense', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            with slim.arg_scope(densenet_arg_scope(weight_decay=0.004)):
                pass
                ##########################
                # Put your code here.
                ##########################
                # At the beginning, the input was just the input image.
                # The image was convoluted by a kernel size of [3,3], 
                # the output channel was also set to 2 * growth. 
                num_channels = 2 * growth

                net = slim.conv2d(images, num_channels, [3,3], stride=2, scope = 'first_conv')

                end_points['first_conv'] = net

                # The output of the first convolution was sent to the first dense block.
                # We already have the codes for the dense block: 
                # block(net, layers, growth, scope='block'). 
                # As mentioned in the paper, the l layer recieves the Feature maps of all preceding layers. 
                # However, in my opinion, in this code, the input of each layer might only concatenate 
                # the output and input of the last layer instead of all the inputs and outputs
                # of all the preceding layers. 
                # I also tried an fully connection of all the inputs, but a resoruce exhaust error occurred.
                
                #1st dense block
                net = block(net, layers=5, growth=growth, scope='dense_block_1')
                # net = dense_block(net, growth=growth, scope='dense_block_1')
                end_points['dense_block_1'] = net

                # Add a transition layer at the end of the dense block
                # The channels of the conv2d was set to the number of the channels at the fisrt conv layer
                # plus the growth rate * the number of layers in each dense

                num_channels = 12 + growth * 5
                net = slim.batch_norm(net, scope='trans_batch_1')
                net = slim.conv2d(net,num_channels, [1,1], scope='trans_conv2d_1')
                net = slim.avg_pool2d(net,[2,2], stride=2, padding="same", scope='trans_pool_1')

                #Add the 2nd dense block
                net = block(net, layers=5, growth=growth, scope='dense_block_2')
                # net = dense_block(net, growth=growth, scope='dense_block_2')
                end_points['dense_block_2'] = net

                #Add a pooling layer at the end of the dense block

                num_channels += growth * 5
                net = slim.batch_norm(net, scope='trans_batch_2')
                net = slim.conv2d(net,num_channels, [1,1], scope='trans_conv2d_2')
                net = slim.avg_pool2d(net,[2,2], stride=2, padding="same", scope='trans_pool_2')

                #Add the 3nd dense block
                net = block(net, layers=5, growth=growth, scope='dense_block_3')
                # net = dense_block(net, growth=growth, scope='dense_block_3')
                end_points['dense_block_3'] = net

                #Add a pooling layer at the end of the dense block

                num_channels += growth * 5
                net = slim.batch_norm(net, scope='trans_batch_3')
                net = slim.conv2d(net,num_channels, [1,1], scope='trans_conv2d_3')
                net = slim.avg_pool2d(net,net.shape[1:3], stride=1, padding="valid", scope='trans_pool_3')


                if not num_classes:
                  return net, end_points
                logits = slim.fully_connected(net, 
                                      num_classes,
                                      biases_initializer=tf.zeros_initializer(),
                                      weights_initializer=trunc_normal(0.01),
                                      weights_regularizer=None,
                                      activation_fn=None,
                                      scope='logits')
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits


                end_points['Predictions'] = slim.softmax(logits, scope='predictions')


    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
