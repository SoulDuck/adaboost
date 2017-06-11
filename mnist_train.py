import tensorflow as tf
from cnn  import convolution2d , max_pool , algorithm , affine
import data
import batch
##########################setting############################
image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
##########################structure##########################
layer = convolution2d('conv1', x_, 64)
layer = max_pool(layer)
top_conv = convolution2d('top_conv', x_, 128)
layer = max_pool(top_conv)
layer= tf.reshape(layer, [-1 , layer[1]*layer[2]*layer[3]])
layer = affine('fully_connect', layer, 1024)
y_conv=affine('end_layer' , layer , n_classes)
#############################################################
#cam = get_class_map('gap', top_conv, 0, im_width=image_width)
pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, 0.005)
saver = tf.train.Saver()
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
try:
    saver.restore(sess, './model/best_acc.ckpt')
    print 'model was restored!'
except tf.errors.NotFoundError:
    print 'there was no model'
########################training##############################
max_val = 0
check_point = 1000
for step in range(100):
    if step % check_point == 0:
        #inspect_cam(sess, cam, top_conv, test_imgs, test_labs, step, 50, x_, y_, y_conv)
        val_acc, val_loss = sess.run([accuracy, cost], feed_dict={x_: test_imgs[:100], y_: test_labs[:100]})
        print val_acc, val_loss
        if val_acc > max_val:
            saver.save(sess, './model/best_acc.ckpt')
            print 'model was saved!'
    batch_xs, batch_ys = batch.next_batch(train_imgs, train_labs, batch_size=60)
    train_acc, train_loss, _ = sess.run([accuracy, cost, train_op], feed_dict={x_: batch_xs, y_: batch_ys})
