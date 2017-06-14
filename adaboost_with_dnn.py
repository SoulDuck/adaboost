import numpy as np
import data
import decision_stump
import utils
import data
from mnist_eval import eval_mnist_train
import os

def build_dnn(inputs , labels , D):
    n=len(labels)
    minError = np.inf
    model_folder_path='./cnn_model'
    model_names=os.walk(model_folder_path).next()[1]
    best_model={}
    for i,model_name in enumerate(model_names):
        model_path=os.path.join(model_folder_path , model_name)
        cls = np.argmax(labels ,axis=1)
        pred, pred_cls = eval_mnist_train( inputs , labels , model_path )
        pred_cls=np.asarray(pred_cls)
        err_np=np.ones([n])
        err_np[pred_cls == cls] = 0
        err_np=err_np.reshape([1,len(err_np)])
        D=D.reshape([len(D) ,1 ])
        weightErr = np.matmul(err_np,D)
        print 'cls sampel', cls[:10]
        if __debug__ ==True:
            print '###############debug#################'
            print 'predcition cls sample' , pred_cls[:10]
            print 'the number of cls' , np.shape(cls)
            print 'the number of prediction cls',len(pred_cls)
            print 'err_shape',err_np.shape
            print 'D shape',D.shape
            print 'weightErr',weightErr
            print '####################################'
        if weightErr <minError:
            best_model_path=model_path
            best_model_pred_cls = pred_cls
            minError= weightErr
            best_model['model_path']=model_path
            best_model['pred_cls']=pred_cls
            best_model['weightErr']=weightErr

            if __debug__==True:
                print '#####best model####'
                print 'best model path' , best_model_path
                print 'best_model pred cls sample' , pred_cls[:10]
                print 'best model min Error' , weightErr

    return  best_model




def train(inputs, labels, iter=2):
    cls = np.argmax(labels, axis=1)
    weakClass=[]
    n,h,w,ch=np.shape(inputs)
    D=np.mat(np.ones([n,1])/n)
    if __debug__==True:
        print D
    aggClassEst =  np.mat(np.ones([n,1]))
    for i in range(iter):
        best_model=build_dnn(inputs, labels , D)
        alpha= float(0.5 * np.log((1.0-best_model['weightErr'])/max(best_model['weightErr'] , 1e-16))) #
        print 'alpha',alpha
        best_model['alpha'] = alpha # add 'alpha' featrue to best stump
        weakClass.append(best_model)
        expon = np.multiply(-1*alpha*(cls) , best_model['pred_cls'])
        expon=expon.reshape([100,1])
        print 'expon',np.shape(expon)
        print 'D',D.shape
        D = np.multiply(D , np.exp(expon))
        D=D/D.sum()
        print alpha
        best_model['pred_cls']=best_model['pred_cls'].reshape([100,1])
        print np.shape(best_model['pred_cls'])

        aggClassEst += alpha*best_model['pred_cls']
        print aggClassEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(cls).T , np.ones([n,1]))
        print aggErrors
        errorRate=aggErrors.sum()/n

        if __debug__==True:
            print 'total error:',errorRate,"\n"
            print '**************************************************************************************'
            print 'iter:',i
            print 'best Stump :',best_model
            print 'alpha :', alpha
            print '(-1 * alpha) * np.mat(labels) :',(-1 * alpha) * np.mat(labels)
            print 'D :',D.T
            print 'best_clasEst :',best_model['pred_cls']
            print 'weak class :' , weakClass
            print 'classEstimate' , best_model['pred_cls']
            print 'D :' , D.T
            print 'aggClassEst :',aggClassEst.T
            print 'expon :',expon.T , expon.shape
            print 'aggClassEst : ' , aggClassEst.T
            print 'np.sign(aggClassEst)',np.sign(aggClassEst).T
            print 'label :',labels
            print 'np.sign(aggClassEst) != np.mat(labels).T :',(np.sign(aggClassEst) != np.mat(labels).T)
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
    image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
    #print os.walk('./cnn_model').next()[1]
    limit=100
    n=len(train_labs[:limit])
    D=np.ones([n])/n
    build_dnn(train_imgs[:limit] , train_labs[:limit] , D )
    train(train_imgs[:limit] , train_labs[:limit] )
