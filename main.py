import numpy as np
import os
# from PIL import Image
import time
import logging
from model.resnet_v1 import *
from model.lenet import *
import pic_preprocessing as image_preprocess
import tensorflow as tf

flags = tf.app.flags
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "Whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "Whether to random constrast")
tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 112, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "Whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 30000000, 'The max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 2201, "The step num to eval")
tf.app.flags.DEFINE_float('learning_rate', 0.0001, "The steps to save")
tf.app.flags.DEFINE_float('decay_rate', 0.97, "The steps to save")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint3755/', 'The checkpoint dir')
tf.app.flags.DEFINE_string('log_dir', './log3755', 'the logging dir')
tf.app.flags.DEFINE_string('train_data_dir', './data_tfrecord/train/', 'The train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data_tfrecord/test/', 'The test dataset dir')
tf.app.flags.DEFINE_boolean('restore', True, 'Whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 50, 'Number of epoches')
tf.app.flags.DEFINE_integer('decay_steps', 2201, 'How many steps to decrease the learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 256, 'How many images to train every step.')
tf.app.flags.DEFINE_integer('top_k', 5, 'To inference the most top_k results.')
tf.app.flags.DEFINE_integer('gpu', 0, 'Which gpu to use.')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "validation", "inference"}')
tf.app.flags.DEFINE_string('lexicon_path', './lexicon3755.txt', "The path of the dictionary.")
tf.app.flags.DEFINE_string('png_dir', 'G:/why_workspace/hw_data/1_112_1/2/61.png', 'A few images used to inference.')
FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
print("-----------------------------main.py start--------------------------")

def file_name(file_dir, lb):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if lb == 'train':
                if os.path.splitext(file)[0] == 'train_data':
                    L.append(os.path.join(root, file))
            elif lb == 'test':
                if os.path.splitext(file)[0] == 'test_data':
                    L.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[0] == 'val_data':
                    L.append(os.path.join(root, file))
    return L


def get_batch_train(dirpath, lb):
    filenames = file_name(dirpath, lb)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
            'image': tf.FixedLenFeature([], tf.string, default_value=""),
            'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'height': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'width': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        width = tf.cast(parsed["width"], tf.int32)
        height = tf.cast(parsed["height"], tf.int32)
        image = tf.reshape(image, [height, width, 1])
        image = tf.image.resize_images(image, (FLAGS.image_size, FLAGS.image_size))
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 1])
        new_image = image_preprocess.preprocess_image(image, FLAGS.image_size, FLAGS.image_size)
        label = tf.cast(parsed["label"], tf.int32)
        return new_image, label
    dataset = dataset.map(parser)
    dataset = dataset.repeat(FLAGS.epoch)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


def get_batch_test(dirpath, lb):
    filenames = file_name(dirpath, lb)
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'height': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'width': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
            }
        parsed = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert to [0,1]
        image = tf.subtract(1.0, image)
        width = tf.cast(parsed["width"], tf.int32)
        height = tf.cast(parsed["height"], tf.int32)
        image = tf.reshape(image, [height, width, 1])
        image = tf.image.resize_images(image, (FLAGS.image_size, FLAGS.image_size))
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 1])
        label = tf.cast(parsed["label"], tf.int32)
        return image, label
    dataset = dataset.map(parser)
    dataset = dataset.repeat(1).batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


def build_graph(num_classes, top_k, is_train, is_test):
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    if is_train:
        net, end_points = resnet_v1_18(images, num_classes=num_classes, is_training=True, reuse=False)
        #net, end_points = lenet(images, num_classes=num_classes, is_training=True, reuse=False)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
        pre_label = tf.argmax(net, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), labels), tf.float32))
        probabilities = tf.nn.softmax(net)
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    else:
        if is_test == False:
            vali_net, vali_end_points = resnet_v1_18(images, num_classes=num_classes, is_training=False, reuse=True)
            #vali_net, vali_end_points = lenet(images, num_classes=num_classes, is_training=False, reuse=True)
        else:
            vali_net, vali_end_points = resnet_v1_18(images, num_classes=num_classes, is_training=False, reuse=False)
            #vali_net, vali_end_points = lenet(images, num_classes=num_classes, is_training=False, reuse=False)
        vali_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vali_net, labels=labels))
        vali_pre_label = tf.argmax(vali_net, 1)
        vali_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vali_net, 1), labels), tf.float32))
        vali_probabilities = tf.nn.softmax(vali_net)
        vali_predicted_val_top_k, vali_predicted_index_top_k = tf.nn.top_k(vali_probabilities, k=top_k)
        vali_accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(vali_probabilities, labels, top_k), tf.float32))
        return {'images': images,
                'labels': labels,
                'logits': vali_net,
                'top_k': top_k,
                'loss': vali_loss,
                'accuracy': vali_accuracy,
                'predicted': vali_pre_label,
                'accuracy_top_k': vali_accuracy_in_top_k,
                'predicted_distribution': vali_probabilities,
                'predicted_index_top_k': vali_predicted_index_top_k,
                'predicted_val_top_k': vali_predicted_val_top_k}

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                                      staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=rate)
    train_op = opt.minimize(loss, global_step=global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    return {'images': images,
            'labels': labels,
            'logits': net,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted': pre_label,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


acc = []
result = []
step = 0

def train():
    global step
    print('Begin training')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    with sess:
        train_images, train_labels = get_batch_train(dirpath=FLAGS.train_data_dir, lb='train')
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=FLAGS.top_k, is_train=True, is_test=False)
        vali_graph = build_graph(is_train=False, num_classes=FLAGS.charset_size, top_k=FLAGS.top_k, is_test=False)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("Resnet18-3755 restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])
        num = 0  # compute epoch
        count = 0  # early stop num
        maxAcc = 0  # best accuracy
        allAcc1Train = 0.0
        allAcc10Train = 0.0
        trainIter = 0
        sum_losstrain = 0.0
        start_time = time.time()
        while True:
            try:
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch
                             }
                _, loss_train, acc1_train, acc10_train, train_summary, step, logit = sess.run(
                    [graph['train_op'], graph['loss'], graph['accuracy'], graph['accuracy_top_k'],
                     graph['merged_summary_op'],
                     graph['global_step'], graph['logits']], feed_dict=feed_dict)
                trainIter = trainIter + 1
                allAcc1Train = allAcc1Train + acc1_train
                allAcc10Train = allAcc10Train + acc10_train
                sum_losstrain += loss_train
                if step % 100 == 0:
                    end_time = time.time()
                    logger.info('step {0} loss: {1} takes time: {2}'.format(step, sum_losstrain / 100, end_time-start_time))
                    start_time = time.time()
                    sum_losstrain = 0.0
                train_writer.add_summary(train_summary, step)
                # Print images
                '''
                for i in range(FLAGS.batch_size):
                    images = train_images_batch[i]
                    #h, w, c = images.shape
                    #assert c == 1
                    images = images.reshape(112, 112)
                    print(train_labels_batch[i])
                    plt.imshow(images, cmap='gray')
                    plt.show()
                '''
                if step % FLAGS.eval_steps == 0:
                    val_images, val_labels = get_batch_test(dirpath=FLAGS.test_data_dir, lb='test')

                    allAcc1Train = allAcc1Train / trainIter
                    allAcc10Train = allAcc10Train / trainIter
                    logger.info("@Resnet18-3755 Accuracy of train set util {0}: {1}/{2}".format(step, allAcc1Train, allAcc10Train))
                    allAcc1Train = 0.0
                    allAcc10Train = 0.0
                    trainIter = 0.0
                    coord = tf.train.Coordinator()
                    while True:
                        try:
                            i = 0
                            acc_top_1 = 0.0
                            acc_top_k = 0.0
                            while not coord.should_stop():
                                i += 1
                                val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                                feed_dict = {vali_graph['images']: val_images_batch,
                                             vali_graph['labels']: val_labels_batch
                                             }
                                acc_1, acc_10, vali_predicted = sess.run([vali_graph['accuracy'],
                                                                          vali_graph['accuracy_top_k'],
                                                                          vali_graph['predicted']],
                                                                         feed_dict=feed_dict)
                                acc_top_1 += acc_1
                                acc_top_k += acc_10
                        except tf.errors.OutOfRangeError:
                            acc_top_1 = acc_top_1 / (i - 1)
                            acc_top_k = acc_top_k / (i - 1)
                            logger.info(
                                'Resnet18-3755 epoch # {0}, top 1 accuracy: {1}, top {2} accuracy: {3}'.format(num,
                                                                                                               acc_top_1,
                                                                                                               FLAGS.top_k,
                                                                                                               acc_top_k))
                            break
                    acc.append(acc_top_1)
                    if maxAcc < acc[num]:
                        logger.info('Accuracy has increased, count = {0}'.format(count))
                        maxAcc = acc[num]
                        count = 0
                        logger.info('@Resnet18-3755 Save the ckpt of {0} step(s)/{1} epoch(es)'.format(step, num))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                                   global_step=graph['global_step'])
                    else:
                        count = count + 1
                        logger.info('Accuracy has not increased, count = {0}'.format(count))
                        logger.info('@Resnet18-3755 Early stop num: {0}'.format(count))
                    num = num + 1
                if step > FLAGS.max_steps:
                    logger.info('@Resnet18-3755 Best Accuracy: {0} (Step over)'.format(maxAcc))
                    break
                if count == 10:
                    logger.info('@Resnet18-3755 Best Accuracy: {0} (Early stopped)'.format(maxAcc))
                    break
            except tf.errors.OutOfRangeError:
                logger.info('@Resnet18-3755 Best Accuracy: {0} (Out of range) step : {1}'.format(maxAcc, step))
                break
    train_writer.close()


def validation():
    print('validation')
    with tf.Session() as sess:
        val_images, val_labels = get_batch_test(dirpath=FLAGS.test_data_dir, lb='test')
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=FLAGS.top_k, is_train=False, is_test=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
        coord = tf.train.Coordinator()
        logger.info(':::Start validation:::')
        while True:
            try:
                i = 0
                acc_top_1, acc_top_k = 0.0, 0.0
                while not coord.should_stop():
                    i += 1
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    feed_dict = {graph['images']: val_images_batch,
                                 graph['labels']: val_labels_batch,
                                 }
                    acc_1, acc_k, pre_label, index_top_k = sess.run([graph['accuracy'],
                                                                     graph['accuracy_top_k'],
                                                                     graph['predicted'],
                                                                     graph['predicted_index_top_k']],
                                                                    feed_dict=feed_dict)
                    acc_top_1 += acc_1
                    acc_top_k += acc_k
            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_1 = acc_top_1 / (i - 1)
                acc_top_k = acc_top_k / (i - 1)
                logger.info('top 1 accuracy {0}, top {1} accuracy {2}'.format(acc_top_1, FLAGS.top_k, acc_top_k))
                break
    return {'acc_top_1': acc_top_1}


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = tf.subtract(1.0, temp_image)
    temp_image = tf.reshape(temp_image, shape=[-1, 112, 112, 1])
    with tf.Session() as sess:
        logger.info('========start inference============')
        graph = build_graph(num_classes=FLAGS.charset_size, top_k=FLAGS.top_k, is_train=False, is_test=True)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        temp_image = sess.run(temp_image)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image})
    return predict_val, predict_index


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "validation":
        validation()
    elif FLAGS.mode == "inference":
        f = open(FLAGS.lexicon_path, 'r')
        lines = f.readlines()
        f.close()
        image_path = FLAGS.png_dir
        final_predict_val, final_predict_index = inference(image_path)
        for i in range(FLAGS.top_k):
            logger.info(
                'Predict top-{0} character:{1}It\'s predict_val:{2}.'.format(i+1, lines[int(final_predict_index[0][i])],
                                                                             final_predict_val[0][i]))


if __name__ == "__main__":
    tf.app.run()
