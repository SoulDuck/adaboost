import tensorflow as tf
import data
import numpy as np
import utils

def eval_mnist_train(inputs , labels , model_folder_path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_folder_path+'/best_acc.ckpt.meta')
    saver.restore(sess, model_folder_path+'/best_acc.ckpt')
    tf.get_default_graph()
    softmax = tf.get_default_graph().get_tensor_by_name('softmax:0')
    pred_cls = tf.get_default_graph().get_tensor_by_name('pred_cls:0')
    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    batch_img_list , batch_lab_list=utils.divide_images_labels_from_batch(inputs , labels , batch_size=60)
    batch_img_lab=zip(batch_img_list , batch_lab_list)
    n=len(batch_img_lab)
    n_classes=10
    pred_list=[];pred_cls_list=[]
    for i,(batch_xs , batch_ys) in enumerate(batch_img_lab):
        utils.show_progress(i , n)
        prediction , prediction_cls=sess.run([softmax ,pred_cls] , feed_dict={x_ : batch_xs  , y_ : batch_ys})
        pred_list.extend(prediction) ; pred_cls_list.extend(prediction_cls)
    return pred_list , pred_cls_list


if __name__=='__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./cnn_model/best_acc.ckpt.meta')
    saver.restore(sess, './cnn_model/best_acc.ckpt')
    tf.get_default_graph()
    softmax = tf.get_default_graph().get_tensor_by_name('softmax:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv/relu:0')
    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')

    image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs=data.mnist_28x28()
    sample_img=test_imgs[0:1]
    pred=sess.run([softmax], feed_dict={x_:sample_img})
    print pred
