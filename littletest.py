import numpy as np
a = np.random.rand(3,3)
ma = np.array([False,False,True])
ma = np.tile(ma, (3,1))
print(ma)
np.place(a, ma,np.round(a[ma]))


print(a)