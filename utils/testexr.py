import h5py
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
# path = "D:dataset/rectestalphas/0/"
# files= os.listdir(path)
# img = cv2.imread(path+files[0], cv2.IMREAD_UNCHANGED)
# print(img[:50,:50,3])
f2 = h5py.File('D:/doubledataset/Holopix50k2.hdf5','r')
list(f2.keys())
f2c=f2['right_color']
print(f2c[2345,:].shape)
plt.imshow(np.float32((f2c[23451,:])))
plt.show()