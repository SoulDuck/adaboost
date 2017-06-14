import matplotlib.pyplot as plt
import numpy as np
import sys
debug_flag=True
def plotROC(predStrength , labels):
    cursor=(1.0,1.0) #initial cursor
    ySum= 0.0 # for AUC curve
    n_pos=np.sum(np.array(labels) ==1)
    n_neg=len(labels)-n_pos
    y_step=1/float(n_pos)
    x_step=1/float(n_neg)
    n_est_pos=0
    sortedIndices=np.argsort(predStrength , axis=0)
    fig= plt.figure()
    fig.clf()
    ax=plt.subplot(1,1,1)
    if __debug__ == debug_flag:
        print 'labels',labels
        print 'predStrength',predStrength.T
        print 'sortedIndices',sortedIndices.T
    for ind in sortedIndices.tolist():
        ind=int(ind[0]);
        if labels[ind] ==1.0:
            DelX=0; DelY=y_step
        else :
            DelX=x_step ; DelY=0
            ySum += cursor[1]
        ax.plot([ cursor[0] , cursor[0]-DelX ] , [ cursor[1] , cursor[1]-DelY])
        cursor=(cursor[0]-DelX , cursor[1] -DelY)
        if __debug__ == debug_flag:
            print 'label',labels[ind]
            print 'delX',
            print 'sortedIndices', sortedIndices.T
            print 'DelX:',DelX,'DelY:',DelY
            print 'cursor[0]-DelX :',cursor[0],'cursor[1]-DelY :',cursor[1]
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    if __debug__==debug_flag:
        print '# of True :' ,n_pos
        print '# of False :' ,n_neg
    plt.show()
    print 'The Area Under Curve is :' , ySum*x_step

def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
def divide_images_labels_from_batch(images, labels ,batch_size):
    batch_img_list=[]
    batch_lab_list = []
    share=len(labels)/batch_size
    #print len(images)
    #print len(labels)
    #print 'share :',share

    for i in range(share):

        imgs=images[i*batch_size:(i+1)*batch_size]
        labs=labels[i * batch_size:(i + 1) * batch_size]
       # print i , len(imgs) , len(labs)
        batch_img_list.append(imgs)
        batch_lab_list.append(labs)
        if i==share-1:
            imgs = images[(i+1)*batch_size:]
            labs = labels[(i+1)*batch_size:]
            #print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
    if __debug__==True:
        print 'the number of lists',len(batch_img_list)
        print 'the number of labels:',len(labels)
    return batch_img_list , batch_lab_list