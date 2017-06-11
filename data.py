import numpy as np
def load_simpleData():
    """

    :return: return inputs , labels
    """
    inputs=np.matrix([[1. ,2.1],
                      [2. ,1.1],
                      [1.3, 1.],
                      [1. , 1.],
                      [2. , 1.]])
    labels=[1.0 , 1.0,-1.0,-1.0 , 1.0]
    return inputs , labels

def load_data(filepath):
    n_feat=len(open(filepath).readline().split('\t'))
    data=[] ; labels=[]
    f=open(filepath)
    for line in f.readlines():
        lineArr=[]
        elements=line.strip().split('\t')
        for ele in elements[:-1]:
            lineArr.append(float(ele))
        data.append(lineArr)
        labels.append(float(elements[-1]))
    return data , labels
from tensorflow.examples.tutorials.mnist import input_data

def mnist_28x28():

    mnist = input_data.read_data_sets('MNIST_DATA_SET', one_hot=True)
    mnist_train_imgs=np.reshape(mnist.train.images , (55000 ,28,28,1))
    mnist_train_labs=mnist.train.labels
    mnist_test_imgs = np.reshape(mnist.test.images, (10000, 28, 28, 1))
    mnist_test_labs = mnist.test.labels

    print mnist_test_imgs.shape , mnist_train_imgs.shape , mnist_train_labs.shape , mnist_test_labs.shape

    ####
    image_height = 28
    image_width = 28
    image_color_ch = 1
    n_classes = 10
    train_imgs=mnist_train_imgs
    train_labs=mnist_train_labs
    test_imgs=mnist_test_imgs
    test_labs=mnist_test_labs
    return image_height , image_width , image_color_ch , n_classes, train_imgs , train_labs , test_imgs, test_labs
    #####

if __name__ == '__main__':
    filepath='/Users/seongjungkim/Desktop/git/machinelearninginaction/Ch07/horseColicTraining2.txt'
    inputs,labels= load_data(filepath)

    print len(inputs)
    print len(labels)
    print len(inputs[0])
