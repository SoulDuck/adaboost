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


if __name__ == '__main__':
    filepath='/Users/seongjungkim/Desktop/git/machinelearninginaction/Ch07/horseColicTraining2.txt'
    inputs,labels= load_data(filepath)

    print len(inputs)
    print len(labels)
    print len(inputs[0])
