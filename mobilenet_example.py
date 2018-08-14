import tensorflow as tf
import numpy as np
from mobilenet import mobilenet_v1
from tensorflow.contrib import slim


def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def create_readable_names_for_imagenet_labels():
    filename = "imagenet_lsvrc_2015_synsets.txt"
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename = "imagenet_metadata.txt"
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names

if __name__ == '__main__':
    from datasets.VOC_Data import VOC_Data
    annotation_path = "./config/test.txt"
    classes_path = "./config/pet_label_map.pbtxt"
    anchors_path = "./config/yolo2_anchors.txt"
    weight_path = "model/416_tree_mobilev1.ckpt"
    dataset = VOC_Data(annotation_path, classes_path, anchors_path, train=False, input_shape=(416, 416))
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True, batch_norm_decay=0.0)):
        im_inputs = tf.placeholder(tf.float32, [None, dataset.input_shape[0], dataset.input_shape[1], 3], name="inputs")
        y_true = tf.placeholder(tf.float32, [None, dataset.num_classes], name="labels")
        logits, endpoints = mobilenet_v1.mobilenet_v1(im_inputs, num_classes=dataset.num_classes, is_training=True,
                                                      global_pool=True)
    vars = slim.get_model_variables()
    saver = tf.train.Saver(vars)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        print("load:", weight_path)
        saver.restore(sess, weight_path)
        validation_accuracys = 0
        x, y = dataset.read_data_label(20)
        for i in range(int(len(x)/20)):
            validation_accuracy = sess.run(evaluation_step, feed_dict={im_inputs: x[20*i:20*(i+1)], y_true: y[20*i:20*(i+1)]})
            validation_accuracys += validation_accuracy
            print("iter:{},acc:{}".format(i, validation_accuracys/(i+1)))