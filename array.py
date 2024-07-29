"""Creating an image with Numpy"""

import numpy as np 
from PIL import Image
a=np.zeros([4])
print(a)
a=np.zeros([4],dtype=np.uint8)
print(a)
b=np.zeros([4,3],dtype=np.uint8)
print(b)
b[2]=[20]
print(b)
b[0:2,2]=[70]
print(b)

c=np.zeros([4,2,3],dtype=np.uint8)
print(c)
c[:,:]=30
print(c)
myImage=Image.fromarray(c)
myImage.save('photo.jpg')
c=np.zeros([40,20,3],dtype=np.uint8)
print(c)
c[:,:]=[0,255,0]
print(c)
c=np.zeros([400,200,3],dtype=np.uint8)
print(c)
c[0:200,:]=[0,255,0]
c[200:,:]=[0,0,255]
print(c)
myImage=Image.fromarray(c)
Image.save('photo.jpg')
c[0:200,:100]=[255,0,0]
c[0:200,100:]=[0,0,255]
myImage=Image.fromarray(c)
Image.save('photo.jpg')


