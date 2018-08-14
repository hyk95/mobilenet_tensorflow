import time
import tensorflow as tf
from mobilenet import mobilenet_v1
from tensorflow.contrib import slim


def variable_summaries(var, name):
    with tf.name_scope("summar"+name):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)


def get_tuned_variables(vars, scopes):
    TRAINABLE_SCOPES = scopes
    variables_to_restore = []
    for var in vars:
        excluded = True
        if var.op.name.startswith(TRAINABLE_SCOPES):
            excluded = False
        if excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def data_augmentation(images):
    images = tf.image.random_brightness(images, 0.6, seed=None)
    images = tf.image.random_contrast(images, 0.1, 0.5, seed=None)
    return images


def loss_function(y_true, y_pred, dataset):
    basic_loss = tf.losses.softmax_cross_entropy(logits=y_pred[:, :2], onehot_labels=y_true[:, :2], scope="basic_loss")
    cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred[:, 2:dataset.dog_index],
                                                                        labels=y_true[:, 2:dataset.dog_index])
                             )
    dog_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred[:, dataset.dog_index:],
                                                                        labels=y_true[:, dataset.dog_index:])
                             )
    total_loss = cat_loss + dog_loss
    total_loss = tf.Print(total_loss, [total_loss, cat_loss, dog_loss, basic_loss], message='loss: ')
    tf.losses.add_loss(total_loss)


def train(dataset, epochs, batch_size, weight_path):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True, batch_norm_decay=0.99)):
        im_inputs = tf.placeholder(tf.float32, [None, dataset.input_shape[0], dataset.input_shape[1], 3], name="inputs")
        # images_arg = data_augmentation(im_inputs)
        y_true = tf.placeholder(tf.float32, [None, dataset.num_classes], name="labels")
        logits, endpoints = mobilenet_v1.mobilenet_v1(im_inputs, num_classes=dataset.num_classes, is_training=True, global_pool=True)
    net_out_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true))
    weight_loss = tf.losses.get_regularization_losses()
    # net_out_loss = tf.losses.get_losses()
    variable_summaries(net_out_loss, "net_loss")
    all_loss = weight_loss
    cost = tf.add_n(all_loss) + net_out_loss
    variable_summaries(cost, "total_loss")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.Variable(0, trainable=False)
    with tf.control_dependencies(update_ops):
        Adam_optim = tf.train.AdamOptimizer(learning_rate=0.0001)
        Momentum_optim = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.0001)
        optim = slim.learning.create_train_op(cost, Momentum_optim, global_step=global_step)
        # Momentum_optim = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.001).minimize(cost, global_step=global_step)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_summaries(evaluation_step, "accuracy")
    train_writer = tf.summary.FileWriter("log", tf.get_default_graph())
    merge_summary = tf.summary.merge_all()
    vars = slim.get_model_variables()
    saver = tf.train.Saver(tf.global_variables())
    load_fn = slim.assign_from_checkpoint_fn(
        weight_path,
        tf.global_variables(),
        ignore_missing_vars=True)
    with tf.Session() as sess:
        print("load:", weight_path)
        saver.restore(sess, weight_path)
        for epoch in range(epochs):
            startTime = time.time()
            for iter_ in range(dataset.num_data // batch_size):
                x, y = dataset.read_data_label(batch_size)
                if iter_ % 50 == 0:
                    loss, _, train_summary, step = sess.run([cost, optim, merge_summary, global_step], feed_dict={im_inputs: x, y_true: y})
                    val_loss, validation_accuracy = sess.run([cost, evaluation_step], feed_dict={im_inputs: x, y_true: y})
                    train_writer.add_summary(train_summary, step)
                    print("epoch:{};iter:{};train_loss:{};train_loss:{};val_acc{}:step:{}".format(epoch, iter_, loss, val_loss, validation_accuracy, step))
                else:
                    _ = sess.run([optim], feed_dict={im_inputs: x, y_true: y})
            endTime = time.time()
            print("epoch_time:{}".format(endTime - startTime))
        saver.save(sess, "model/416_tree_mobilev1.ckpt")


def switch_optim(optim, epoch):
    if epoch > 100:
        print("using_optime:Mom")
        return optim[1]
    else:
        print("using_optime:Adam")
        return optim[0]


if __name__ == '__main__':
    from datasets.VOC_Data import VOC_Data
    annotation_path = "./config/train.txt"
    test_annotation_path = "./config/test.txt"
    classes_path = "./config/pet_label_map.pbtxt"
    anchors_path = "./config/yolo2_anchors.txt"
    weight_path = "F:\\deep_learning_models\\tf\\mobilenet_v1_1.0_224\\mobilenet_v1_1.0_224.ckpt"
    train_dataset = VOC_Data(annotation_path, classes_path, anchors_path, input_shape=(416, 416))
    train(train_dataset, epochs=50, batch_size=4, weight_path="model/416_tree_mobilev1.ckpt")
