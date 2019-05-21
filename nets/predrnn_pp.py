__author__ = 'yunbo'
import numpy as np
import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm
from layers.CausalLSTMCell import STLSTMCell as stlstm

from tensorflow.contrib import rnn

def convlstm(batch_size, x, in_shape,out_chan,layer):
    #print(in_shape)
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
        conv_ndims=2,
        input_shape=in_shape,
        output_channels=out_chan,
        kernel_shape=[5, 5],
        use_bias=True,
        skip_connection=False,
        forget_bias=1.0,
        initializers=None,
        name="conv_lstm_cell"+str(layer))

    initial_state = convlstm_layer.zero_state(batch_size, dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(convlstm_layer, x, initial_state=initial_state, time_major=False, dtype="float32")
    return outputs


def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1] # RGB = 3
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")
            gen_images.append(x_gen)
            #print("length of gen_images",len(gen_images))
    np_gen_images = np.stack(gen_images)
    #print("shape of gen_images:", np_gen_images.shape)
    #print(output_channels)
    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images[:,-1,:,:,:] - images[:,-1,:,:,:])
    #loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]


def res_rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1] # RGB = 3
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)


            res_layer = tf.add(hidden[num_layers-1], hidden[num_layers-3], name='stack')

            x_gen = tf.layers.conv2d(inputs=res_layer,
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")

            gen_images.append(x_gen)
            #print("length of gen_images",len(gen_images))
    np_gen_images = np.stack(gen_images)
    #print("shape of gen_images:", np_gen_images.shape)
    #print(output_channels)
    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images[:,-1,:,:,:] - images[:,-1,:,:,:])
    #loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]

def unet_rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    #print(shape)
    output_channels = shape[-1] # RGB = 3
    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None
    image_in = images[:,0:-1,:,:,:]
    in_shape = image_in.get_shape().as_list()
    print(in_shape)
    #for t in range(seq_length-1):
    #    reuse = bool(gen_images)
    with tf.variable_scope('unet_predrnn_pp',reuse =False):
    #        if t < input_length:
    #            inputs = images[:,t]
    #        else:
    #            inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen
            hid_0 = convlstm(shape[0], image_in, [shape[-3],shape[-2],shape[-1]], num_hidden[0],0)
            hid_1 = convlstm(shape[0], hid_0, [shape[-3],shape[-2],num_hidden[0]], num_hidden[1],1)
            hid_2 = convlstm(shape[0], hid_1, [shape[-3],shape[-2],num_hidden[1]], num_hidden[2],2)
            u_0_2 = tf.concat([hid_2, hid_0],axis=-1, name='stack_0_2')
            hid_3 = convlstm(shape[0], u_0_2, [shape[-3],shape[-2],num_hidden[2]+num_hidden[0]], num_hidden[3],3)
            u_in_3 = tf.concat([hid_3, image_in],axis=-1, name='stack_in_3')
            #hid_4 = convlstm(shape[0], hid_3, [shape[-3],shape[-2],shape[-1]], num_hidden[2], stride=1)

            #hidden[0], cell[0], mem = convlstm2d(inputs, hidden[0], cell[0], mem)
            #z_t = gradient_highway(hidden[0], z_t)
            #hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)
            #hidden[2], cell[2], mem = lstm[2](hidden[1], hidden[2], cell[2], mem)
            #hidden[3], cell[3], mem = lstm[3](hidden[2], hidden[3], cell[3], mem)

            #for i in range(2, num_layers):
            #    hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)


            #res_layer = tf.add(hidden[num_layers-1], hidden[num_layers-3], name='stack')
            gen_images = convlstm(shape[0], u_in_3, [shape[-3], shape[-2], num_hidden[3] + shape[-1]], output_channels, "out")

            #x_gen = tf.layers.conv2d(inputs=u_in_3,
            #                         filters=output_channels,
            #                         kernel_size=1,
            #                         strides=1,
            #                         padding='same',
            #                         name="back_to_pixel")

            #gen_images.append(x_gen)
            #print("length of gen_images",len(gen_images))
    #np_gen_images = np.stack(gen_images)
    #print("shape of gen_images:", np_gen_images.shape)
    #print(output_channels)
    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    #gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    print(images[:,-1,:,:,:].shape)
    #loss = tf.nn.l2_loss(gen_images[:,-1,:,:,:] - images[:,-1,:,:,:])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]

