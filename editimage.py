import numpy as np
from PIL import Image
oim=Image.open('dog.jpg')
print(oim)
oarray=np.array(oim)
print(oarray)
print(oarray.shape)
marray=oarray[:,0:300]
print(marray)
print (marray.shape)
myImage=Image.fromarray(marray)
myImage.save('dog.jpg')
marray[:214,:250]=[255,0,255]
myImage=Image.fromarray(marray)
myImage.save('dog.jpg')





