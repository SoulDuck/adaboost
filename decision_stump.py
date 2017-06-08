import numpy as np
import data
import matplotlib.pyplot as plt
debug_flag=False
def classify(inputs,dimen,threshVal , threshIneq):
    ret_np=np.ones((np.shape(inputs)[0],1))
    #(np.shape(inputs)[0] =5
    if threshIneq=='lt':
        ret_np[inputs[:, dimen] <= threshVal]=-1.0
    elif threshIneq=='gt':
        ret_np[inputs[:,dimen] > threshVal] = -1.0

    if __debug__==debug_flag:
        print 'return np',ret_np.T
    return ret_np

def build_stump(inputs , labels , D):
    """
    :param inputs:
    :param labels:
    :param D:
    :return: best_stump , minError , best_clasEst

    """
    count=0
    inputs=np.mat(inputs);labels=np.mat(labels).T
    m,n=inputs.shape
    iter=10;best_stump={};best_clasEst=np.mat(np.zeros((m,1)))
    minError=np.inf
    for i in range(n):
        column_min = inputs[:, i].min() ; column_max=inputs[:, i].max()
        stepSize=float(column_max-column_min)/float(iter)
        for j in range(-1,int(iter)+1):
            for inequal in ['lt', 'gt']:
                count+=1
                if __debug__ == debug_flag:
                    print '---------------------------------------------------------'
                    print 'min:',column_min
                    print 'stepSize:',stepSize
                    print 'j',j
                threshVal=(column_min+float(j)*stepSize)
                predictedVals=classify(inputs,i, threshVal , inequal)
                err_np = np.mat(np.ones((m,1))) #m=5
                err_np[predictedVals == labels]=0
                weightedError=D.T * err_np
                if __debug__==debug_flag:
                    print 'i : ',i , 'j :',j
                    print 'threshold',threshVal
                    print 'inequal' , inequal
                    print 'input data' , inputs[:,i].T
                    print 'predicted Values:',predictedVals.T
                    print 'labels',labels.T
                    print 'error array:',err_np.T
                    print 'weight Error :', weightedError
                    print 'index ', count
                #print '---------------------------------------------------------'
                if weightedError < minError:
                    minError = weightedError
                    best_clasEst=predictedVals.copy()
                    best_stump['dim']=i
                    best_stump['thresh'] = threshVal
                    best_stump['ineq'] = inequal
                    best_stump['index']=count
                    print 'best model was saved'
    return best_stump , minError , best_clasEst

    if __debug__==debug_flag:
        print 'input shape',m,n
        print 'column_min',column_min
        print 'column_max',column_max
        print inputs[:,0]
        print inputs[:,1]

if __name__ =='__main__':
    inputs, labels = data.load_simpleData()
    #classify(inputs , 0 , 1 , 'gt')
    print 'vector D'
    D=np.mat(np.ones([5,1])/5)
    print D.T
    best_stump , minError , best_clasEst =build_stump(inputs, labels, D )
    print '******************************result************************************'
    print 'best stump', best_stump
    print 'min Error', minError
    print 'best_clasEst',best_clasEst.T
    print '************************************************************************'


