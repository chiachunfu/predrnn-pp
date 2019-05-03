__author__ = 'yunbo'
from keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback

import subprocess
import os
import os.path
import time
import datetime
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
import glob
from nets import models_factory
from PIL import Image
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
from skimage.measure import compare_ssim
from keras import backend as K

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'mnist',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predrnn_pp',
                            'dir to store trained net.')

# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
if 0:
    tf.app.flags.DEFINE_integer('input_length', 10,
                                'encoder hidden states.')
    tf.app.flags.DEFINE_integer('seq_length', 20,
                                'total input and output length.')
    tf.app.flags.DEFINE_integer('img_width', 64,
                                'input image width.')
    tf.app.flags.DEFINE_integer('img_channel', 1,
                                'number of image channel.')
    tf.app.flags.DEFINE_integer('patch_size', 4,
                                'patch size on one dimension.')
    tf.app.flags.DEFINE_integer('batch_size', 8,
                                'batch size for training.')
    tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                               'COMMA separated number of units in a convlstm layer.')
    tf.app.flags.DEFINE_float('lr', 0.001,
                              'base learning rate.')
    tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                                'number of iters saving models.')
    tf.app.flags.DEFINE_integer('test_interval', 2000,
                                'number of iters for test.')
    tf.app.flags.DEFINE_string('gen_frm_dir', 'results/mnist_predrnn_pp',
                               'dir to store result.')
else:
    tf.app.flags.DEFINE_integer('input_length', 5,
                                'encoder hidden states.')
    tf.app.flags.DEFINE_integer('seq_length', 6,
                                'total input and output length.')
    tf.app.flags.DEFINE_integer('img_width', 48,
                                'input image width.')
    tf.app.flags.DEFINE_integer('img_channel', 3,
                                'number of image channel.')
    tf.app.flags.DEFINE_integer('patch_size', 1,
                                'patch size on one dimension.')
    tf.app.flags.DEFINE_integer('batch_size', 8,
                                'batch size for training.')
    tf.app.flags.DEFINE_string('num_hidden', '64,48,32,32',
                               'COMMA separated number of units in a convlstm layer.')
    tf.app.flags.DEFINE_float('lr', 0.001,
                              'base learning rate.')
    tf.app.flags.DEFINE_integer('snapshot_interval', 5,
                                'number of iters saving models.')
    tf.app.flags.DEFINE_integer('test_interval', 5,
                                'number of iters for test.')
    tf.app.flags.DEFINE_string('gen_frm_dir', 'result_debug/catz_predrnn_pp',
                               'dir to store result.')

tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')

tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization

tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')

tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')


class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 int(FLAGS.img_width/FLAGS.patch_size),
                                 int(FLAGS.img_width/FLAGS.patch_size),
                                 FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel])

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length-FLAGS.input_length-1,
                                         int(FLAGS.img_width/FLAGS.patch_size),
                                         int(FLAGS.img_width/FLAGS.patch_size),
                                         FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden) #layers = number of hidden layer filters passed in
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory.construct_model(
                FLAGS.model_name, self.x,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,FLAGS.input_length-1:]
            self.loss_train = loss / FLAGS.batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def validate(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        loss = self.sess.run((self.loss_train), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(15, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)


def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.width, config.height, 3 * 5))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [cv2.resize(cv2.imread(img), (FLAGS.img_width,FLAGS.img_width)) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        yield (input_images, output_images)
        counter += batch_size

def my_predrnnpp_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    if 1:
    #while True:
        input_images = np.zeros(
            (batch_size, 6, FLAGS.img_width, FLAGS.img_width, 3))
        #output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [cv2.resize(cv2.imread(img), (FLAGS.img_width,FLAGS.img_width))/255. for img in sorted(input_imgs)]
            #input_images[i] = np.concatenate(imgs, axis=2)

            imgs.append(cv2.resize(cv2.imread(
                cat_dirs[counter + i] + "/cat_result.jpg"), (FLAGS.img_width,FLAGS.img_width))/255.)
            input_images[i] = np.asarray(imgs)
        #yield (input_images, output_images)
        yield (input_images)
        counter += batch_size

class batch_generator():
    def __init__(self, data_dir):
        self.cat_dirs = glob.glob(data_dir + "/*")
        self.counter = 0
        self.total_len = len(self.cat_dirs)
        #print(self.total_len)
        random.shuffle(self.cat_dirs)
    #def batch_gen(self, batch_size, is_add_noise):
    def batch_gen(self, batch_size):

        input_images = np.zeros(
            (batch_size, 6, FLAGS.img_width, FLAGS.img_width, 3))
        input_images_unscaled = np.zeros(
            (batch_size, 6, 96, 96, 3))
        #if ((self.counter+1)*batch_size >= len(self.cat_dirs)):
            #counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(self.cat_dirs[self.counter + i] + "/cat_[0-5]*")
            imgs = [cv2.resize(cv2.imread(img), (FLAGS.img_width,FLAGS.img_width))/255. for img in sorted(input_imgs)]
            imgs_unscaled = [cv2.imread(img) for img in sorted(input_imgs)]
            #input_images[i] = np.concatenate(imgs, axis=2)

            imgs.append(cv2.resize(cv2.imread(
                self.cat_dirs[self.counter + i] + "/cat_result.jpg"), (FLAGS.img_width, FLAGS.img_width)) / 255.)
            imgs_unscaled.append(cv2.imread(
                self.cat_dirs[self.counter + i] + "/cat_result.jpg"))
            input_images[i] = np.asarray(imgs)
            input_images_unscaled[i] = np.asarray(imgs_unscaled)
        self.counter += batch_size
        #yield (input_images, output_images)
        return (input_images, input_images_unscaled)

    def check_batch_left(self):
        if (self.counter+1)*FLAGS.batch_size > self.total_len:
            self.counter = 0 #reset counter
            return False
        else:
            return True

def perceptual_distance(y_true, y_pred): # changed to cv2 format brg
    rmean = (y_true[:, :, :, 1] + y_pred[:, :, :, 1]) / 2
    b = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    r = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    g = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    #return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))
    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    # automatically get the data if it doesn't exist
    if not os.path.exists("catz"):
        print("Downloading catz dataset...")
        subprocess.check_output(
            "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)

    # load data
    #train_input_handle, test_input_handle = datasets_factory.data_provider(
    #    FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
    #    FLAGS.batch_size, FLAGS.img_width)

    print("Initializing models")
    model = Model()
    lr = FLAGS.lr

    delta = 0.00002
    base = 0.99998
    eta = 1
    orig_img_width = 96

    val_dir = 'catz/test'
    train_dir = 'catz/train'
    #train_dir = 'catz_overfit'
    #val_dir = 'catz_overfit'

    TrainData = batch_generator(train_dir)
    ValData = batch_generator(val_dir)
    log_start_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H'))

    train_log_file_name = 'debug_train_loss_'+log_start_time+'.txt'
    train_fh = open(train_log_file_name,'w')
    val_log_file_name = 'debug_val_loss_'+log_start_time+'.txt'
    val_fh = open(val_log_file_name,'w')

    metrics_log_file_name = 'debug_metrics_' + log_start_time + '.txt'
    metrics_fh = open(metrics_log_file_name, 'w')

    for itr in range(1, FLAGS.max_iterations + 1):
        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (FLAGS.batch_size, FLAGS.seq_length - FLAGS.input_length - 1))
        true_token = (random_flip < eta)
        # true_token = (random_flip < pow(base,itr))
        ones = np.ones((int(FLAGS.img_width / FLAGS.patch_size),
                        int(FLAGS.img_width / FLAGS.patch_size),
                        FLAGS.patch_size ** 2 * FLAGS.img_channel))
        zeros = np.zeros((int(FLAGS.img_width / FLAGS.patch_size),
                          int(FLAGS.img_width / FLAGS.patch_size),
                          FLAGS.patch_size ** 2 * FLAGS.img_channel))
        mask_true = []
        for i in range(FLAGS.batch_size):
            for j in range(FLAGS.seq_length - FLAGS.input_length - 1):
                if true_token[i, j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (FLAGS.batch_size,
                                           FLAGS.seq_length - FLAGS.input_length - 1,
                                           int(FLAGS.img_width / FLAGS.patch_size),
                                           int(FLAGS.img_width / FLAGS.patch_size),
                                           FLAGS.patch_size ** 2 * FLAGS.img_channel))
        train_cost_avg = 0
        train_batch_num = 0
        while TrainData.check_batch_left():
            batch_train_seq, _ = TrainData.batch_gen(FLAGS.batch_size)

            cost = model.train(batch_train_seq, lr, mask_true)

            if FLAGS.reverse_input:
                batch_train_seq_rev = batch_train_seq[:, ::-1]
                cost += model.train(batch_train_seq_rev, lr, mask_true)
                cost = cost / 2
            train_batch_num += 1
            train_cost_avg = ((train_cost_avg * (TrainData.counter - FLAGS.batch_size) +
                               cost * FLAGS.batch_size) / TrainData.counter)
            print("iter: ", itr, ", batch: ", train_batch_num, ". current batch training loss: ", cost, ", avg training batch loss: ",
                  train_cost_avg)
            log_str = "iter: " + str(itr) + ", batch: " + str(train_batch_num) + ", current batch training loss: "+ str(cost) +\
                      ", avg training batch loss: " + str(train_cost_avg) + "\n"
            train_fh.write(log_str)

        val_cost_avg = 0
        val_batch_num = 0
        batch_id = 0
        while ValData.check_batch_left():
            batch_val_seq, batch_val_seq_unscaled = ValData.batch_gen(FLAGS.batch_size)
            val_batch_num += 1
            mask_true = np.zeros((FLAGS.batch_size,
                                  FLAGS.seq_length - FLAGS.input_length - 1,
                                  int(FLAGS.img_width / FLAGS.patch_size),
                                  int(FLAGS.img_width / FLAGS.patch_size),
                                  FLAGS.patch_size ** 2 * FLAGS.img_channel))
            cost = model.validate(batch_val_seq, mask_true)
            val_cost_avg = ((val_cost_avg * (ValData.counter - FLAGS.batch_size) + cost * FLAGS.batch_size) / ValData.counter)
            print("iter: ", itr, ", batch: ", val_batch_num, ", current batch validation loss: ", cost, "avg validation batch loss: ",
                  val_cost_avg)
            log_str = "iter: " + str(itr) + ", batch: " + str(val_batch_num) + ", current batch validation loss: " + str(cost) + \
                      ", avg validation batch loss: " + str(train_cost_avg) + "\n"
            val_fh.write(log_str)
            #batch_id = 0
            if itr % FLAGS.test_interval == 0:
                res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
                if not os.path.exists(res_path):
                    os.mkdir(res_path)
                #avg_mse = 0
                #batch_id = 0
                #img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []

                batch_id = batch_id + 1
                batch_val_seq_unscaled_ = batch_val_seq_unscaled
                batch_val_seq_unscaled = batch_val_seq_unscaled[:,-1,:,:,:]
                img_gen = model.test(batch_val_seq, mask_true)
                img_gen = np.asarray(img_gen)
                img_gen = np.reshape(img_gen, [FLAGS.batch_size,FLAGS.seq_length-FLAGS.input_length, FLAGS.img_width, FLAGS.img_width,FLAGS.img_channel])
                #print(img_gen.shape)
                img_gen_ = img_gen
                img_gen = img_gen[:, -1, :, :, :]
                #print(img_gen.shape)
                    # concat outputs of different gpus along batch
                    #img_gen = np.concatenate(img_gen)
                #img_gen_tmp = []
                img_gen_scaled_back = np.zeros(
                    (FLAGS.batch_size, 1, 96, 96, 3))
                for ii, b in enumerate(img_gen):
                    img_gen_tmp = []
                    #img_tmp = b[0]
                    img_gen_tmp.append(cv2.resize(b, (orig_img_width, orig_img_width)) * 255.)
                    img_gen_scaled_back[ii] = np.asarray(img_gen_tmp)
                img_gen = img_gen_scaled_back
                img_gen = img_gen[:, -1, :, :, :]
                fmae = metrics.batch_mae_frame_float(batch_val_seq_unscaled, img_gen)
                mse = np.square(batch_val_seq_unscaled - img_gen).sum()
                #        img_mse[i] += mse
                #       avg_mse += mse

                #        real_frm = np.uint8(x * 255)
                #        pred_frm = np.uint8(gx * 255)
                psnr = metrics.batch_psnr(batch_val_seq_unscaled, img_gen)
                #        for b in range(FLAGS.batch_size):
                #            sharp[i] += np.max(
                #                cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                #            score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True)
                #            ssim[i] += score

                    # save prediction examples
                if batch_id <= 25:
                    path = os.path.join(res_path,str(batch_id))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    for b_idx in range(len(batch_val_seq_unscaled)):
                        for i in range(len(batch_val_seq_unscaled_[0])):
                            name = 'gt_unscaled' + "_img_" + str(b_idx) +"_ seq+ str(1+i)" + '.png'
                            file_name = os.path.join(path, name)
                            #print(batch_val_seq_unscaled.shape)
                            img_gt = np.uint8(batch_val_seq_unscaled_[b_idx, i,:, :, :])
                            cv2.imwrite(file_name, img_gt)
                            #name = 'gt_model' + str(1 + i) + '.png'
                            #file_name = os.path.join(path, name)
                            # print(batch_val_seq_unscaled.shape)
                            #img_gt = np.uint8(batch_val_seq[0, i, :, :, :]*255)

                            #cv2.imwrite(file_name, img_gt)
                        for i in range(len(img_gen_scaled_back[0])):
                            name = 'pd_scaledback'+ "_img_" + str(b_idx) +"_ seq+ str(1+i)" + '.png'
                            file_name = os.path.join(path, name)
                                #img_pd = img_gen[0, i, :, :, :]
                                #img_pd = np.maximum(img_pd, 0)
                                #img_pd = np.minimum(img_pd, 1)
                                #img_pd = np.uint8(img_pd * 255)
                            img_pd = np.uint8(img_gen_scaled_back[b_idx, i, :, :, :])

                            #print(img_pd.shape)
                            cv2.imwrite(file_name, img_pd)
                        if 0 :
                            for i in range(len(img_gen_[0])):
                                name = 'pd_modeloutput'+ "_img_" + str(b_idx) +"_ seq+ str(1+i)" + '.png'
                                file_name = os.path.join(path, name)
                                # img_pd = img_gen[0, i, :, :, :]
                                # img_pd = np.maximum(img_pd, 0)
                                # img_pd = np.minimum(img_pd, 1)
                                # img_pd = np.uint8(img_pd * 255)
                                img_pd = np.uint8(img_gen_[b_idx, i, :, :, :]*255)
                                img_pd = np.reshape(img_pd, [FLAGS.img_width, FLAGS.img_width, FLAGS.img_channel])
                                # print(img_pd.shape)
                                cv2.imwrite(file_name, img_pd)
                        #cv2.imwrite(file_name, img_gen)
                    #test_input_handle.next()
                #avg_mse = avg_mse / (batch_id * FLAGS.batch_size)
                print('mse per seq: ' + str(mse))
                log_str = 'mse per seq: ' + str(mse)
                metrics_fh.write(log_str)
                print('fmae per seq: ' + str(fmae))
                log_str = 'fmae per seq: ' + str(fmae)
                metrics_fh.write(log_str)
                #for i in range(FLAGS.seq_length - FLAGS.input_length):
                    #print(img_mse[i] / (batch_id * FLAGS.batch_size))
                psnr = np.asarray(psnr, dtype=np.float32) #/ batch_id
                #fmae = np.asarray(fmae, dtype=np.float32) #/ batch_id
                #ssim = np.asarray(ssim, dtype=np.float32) #/ (FLAGS.batch_size * batch_id)
                #sharp = np.asarray(sharp, dtype=np.float32) / (FLAGS.batch_size * batch_id)
                print('psnr per frame: ' + str(psnr))
                log_str = 'psnr per seq: ' + str(psnr)
                metrics_fh.write(log_str)
                #for b_idx in range(len(img_gen)):
                    #img_pd = img_gen[b_idx]
                    #img_gt = batch_val_seq_unscaled[b_idx]
                #dist = perceptual_distance(Ｋ.variable(value=batch_val_seq_unscaled), K.variable(value=img_gen))
                dist = perceptual_distance(batch_val_seq_unscaled, img_gen)
                print("iter: "+str(itr)+ ", batch: " + str(val_batch_num) + ". current perceptual dist: "+str(dist))
                metrics_fh.write("iter: "+str(itr)+ ", batch: " + str(val_batch_num) + ", current perceptual dist: "+str(dist))
                #for i in range(FLAGS.seq_length - FLAGS.input_length):
                    #print(psnr[i])
                #print('fmae per frame: ' + str(np.mean(fmae)))
                #for i in range(FLAGS.seq_length - FLAGS.input_length):
                    #print(fmae[i])
                ##print('ssim per frame: ' + str(np.mean(ssim)))
                #for i in range(FLAGS.seq_length - FLAGS.input_length):
                ##    print(ssim[i])
                ##print('sharpness per frame: ' + str(np.mean(sharp)))
                #for i in range(FLAGS.seq_length - FLAGS.input_length):
                #    print(sharp[i])
        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)
    '''
    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        print(ims.shape)
        ims = preprocess.reshape_patch(ims, FLAGS.patch_size)
        print(ims.shape)
        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (FLAGS.batch_size,FLAGS.seq_length-FLAGS.input_length-1))
        true_token = (random_flip < eta)
        #true_token = (random_flip < pow(base,itr))
        ones = np.ones((int(FLAGS.img_width/FLAGS.patch_size),
                        int(FLAGS.img_width/FLAGS.patch_size),
                        FLAGS.patch_size**2*FLAGS.img_channel))
        zeros = np.zeros((int(FLAGS.img_width/FLAGS.patch_size),
                          int(FLAGS.img_width/FLAGS.patch_size),
                          FLAGS.patch_size**2*FLAGS.img_channel))
        mask_true = []
        for i in range(FLAGS.batch_size):
            for j in range(FLAGS.seq_length-FLAGS.input_length-1):
                if true_token[i,j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (FLAGS.batch_size,
                                           FLAGS.seq_length-FLAGS.input_length-1,
                                           int(FLAGS.img_width/FLAGS.patch_size),
                                           int(FLAGS.img_width/FLAGS.patch_size),
                                           FLAGS.patch_size**2*FLAGS.img_channel))
        print("ims shape:",ims.shape)
        print("mask_true shape: ", mask_true.shape)
        cost = model.train(ims, lr, mask_true)
        if FLAGS.reverse_input:
            ims_rev = ims[:,::-1]
            cost += model.train(ims_rev, lr, mask_true)
            cost = cost/2

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))

        if itr % FLAGS.test_interval == 0:
            print('test...')
            test_input_handle.begin(do_shuffle = False)
            res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            os.mkdir(res_path)
            avg_mse = 0
            batch_id = 0
            img_mse,ssim,psnr,fmae,sharp= [],[],[],[],[]
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                fmae.append(0)
                sharp.append(0)
            mask_true = np.zeros((FLAGS.batch_size,
                                  FLAGS.seq_length-FLAGS.input_length-1,
                                  int(FLAGS.img_width/FLAGS.patch_size),
                                  int(FLAGS.img_width/FLAGS.patch_size),
                                  FLAGS.patch_size**2*FLAGS.img_channel))
            while(test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims = test_input_handle.get_batch()
                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
                img_gen = model.test(test_dat, mask_true)
            
                # concat outputs of different gpus along batch
                img_gen = np.concatenate(img_gen)
                img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size)
                # MSE per frame
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    x = test_ims[:,i + FLAGS.input_length,:,:,0]
                    gx = img_gen[:,i,:,:,0]
                    fmae[i] += metrics.batch_mae_frame_float(gx, x)
                    gx = np.maximum(gx, 0)
                    gx = np.minimum(gx, 1)
                    mse = np.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)
                    psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                    for b in range(FLAGS.batch_size):
                        sharp[i] += np.max(
                            cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b],3)))
                        score, _ = compare_ssim(pred_frm[b],real_frm[b],full=True)
                        ssim[i] += score

                # save prediction examples
                if batch_id <= 10:
                    path = os.path.join(res_path, str(batch_id))
                    os.mkdir(path)
                    for i in range(FLAGS.seq_length):
                        name = 'gt' + str(i+1) + '.png'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims[0,i,:,:,:] * 255)
                        cv2.imwrite(file_name, img_gt)
                    for i in range(FLAGS.seq_length-FLAGS.input_length):
                        name = 'pd' + str(i+1+FLAGS.input_length) + '.png'
                        file_name = os.path.join(path, name)
                        img_pd = img_gen[0,i,:,:,:]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)
                test_input_handle.next()
            avg_mse = avg_mse / (batch_id*FLAGS.batch_size)
            print('mse per seq: ' + str(avg_mse))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(img_mse[i] / (batch_id*FLAGS.batch_size))
            psnr = np.asarray(psnr, dtype=np.float32)/batch_id
            fmae = np.asarray(fmae, dtype=np.float32)/batch_id
            ssim = np.asarray(ssim, dtype=np.float32)/(FLAGS.batch_size*batch_id)
            sharp = np.asarray(sharp, dtype=np.float32)/(FLAGS.batch_size*batch_id)
            print('psnr per frame: ' + str(np.mean(psnr)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(psnr[i])
            print('fmae per frame: ' + str(np.mean(fmae)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(fmae[i])
            print('ssim per frame: ' + str(np.mean(ssim)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(ssim[i])
            print('sharpness per frame: ' + str(np.mean(sharp)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(sharp[i])

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        train_input_handle.next()
    '''
if __name__ == '__main__':
    tf.app.run()

