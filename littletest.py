import numpy as np
a = 4*np.random.rand(3,6)
ma = np.array([True,False,True,False,True,False])
ma = np.tile(ma, (3,1))
print(ma)
np.place(a, ma,np.round(a[ma]))

print(a)
b = [3,7]
if b:
    print("F")