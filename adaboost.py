import numpy as np
import data
import decision_stump
import utils

def train(inputs, labels, iter=40):
    weakClass=[]
    m,n=np.shape(inputs);
    D=np.mat(np.ones([m,1])/m)
    if __debug__==True:
        print D
    aggClassEst =  np.mat(np.ones([m,1]))
    for i in range(iter):
        best_stump, minError, best_clasEst=decision_stump.build_stump(inputs, labels , D)
        alpha= float(0.5 * np.log((1.0-minError)/max(minError , 1e-16))) #
        best_stump['alpha'] = alpha # add 'alpha' featrue to best stump
        weakClass.append(best_stump)
        expon = np.multiply(-1*alpha*np.mat(labels).T , best_clasEst)
        D = np.multiply(D , np.exp(expon))
        D=D/D.sum()
        aggClassEst += alpha*best_clasEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(labels).T , np.ones([m,1]))
        errorRate=aggErrors.sum()/m

        if __debug__==True:
            print 'total error:',errorRate,"\n"
            print '**************************************************************************************'
            print 'iter:',i
            print 'best Stump :',best_stump
            print 'alpha :', alpha
            print '(-1 * alpha) * np.mat(labels) :',(-1 * alpha) * np.mat(labels)
            print 'D :',D.T
            print 'best_clasEst :',best_clasEst.T
            print 'weak class :' , weakClass
            print '-1*alpha*np.map(labels).T :' , -1*alpha*np.mat(labels)
            print 'classEstimate' , best_clasEst.T
            print 'D :' , D.T
            print 'aggClassEst :',aggClassEst.T
            print 'expon :',expon.T
            print 'aggClassEst : ' , best_clasEst.T
            print 'np.sign(aggClassEst)',np.sign(aggClassEst).T
            print 'label :',labels
            print 'np.sign(aggClassEst) != np.mat(labels).T :',(np.sign(aggClassEst) != np.mat(labels).T).T
            print 'aggErrors :', aggErrors.T
        if errorRate ==0.0:
            break;

    return weakClass , aggClassEst

def classify(input,weakClass):
    input_np=np.mat(input)
    m=np.shape(input_np)[0]
    aggClassEst=np.mat(np.zeros([m,1]))
    for i in range(len(weakClass)):
        #print weakClass[i]
        classEst=decision_stump.classify(input_np, weakClass[i]['dim'],weakClass[i]['thresh'],weakClass[i]['ineq'])
        aggClassEst+=weakClass[i]['alpha']*classEst
        if __debug__==True:
            print 'aggClassEstimate:',aggClassEst.T
    return np.sign(aggClassEst) , aggClassEst
if __name__ == '__main__':
    #inputs , labels =data.load_simpleData()
    filepath = '/Users/seongjungkim/Desktop/git/machinelearninginaction/Ch07/horseColicTraining2.txt'
    inputs, labels=data.load_data(filepath)
    weakClass , aggClassEst = train(inputs, labels )
    filepath = '/Users/seongjungkim/Desktop/git/machinelearninginaction/Ch07/horseColicTest2.txt'
    inputs, labels = data.load_data(filepath)
    result,aggClassEst=classify(inputs , weakClass)
    errArr=np.mat(np.ones([67,1]))
    errCount=errArr[result != np.mat(labels).T].sum()
    utils.plotROC(aggClassEst,labels)
    print 'Error Count:', errCount
    #print result
