import numpy as np
a=np.array([[0,1],[0,1]])
b=np.array([[0,1],[1,0]])
d=np.array([0,1])
c=np.argmax(a , axis=1)
err_np=np.ones([2])
print c
print d
print err_np
print c