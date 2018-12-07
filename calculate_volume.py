#calculating the size of drops
import time
import sys, os
import cv2
import numpy as np
#only using this because all images are dumb
def rotateImage(image,angle):
image_center = tuple(np.array(image.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
rot_img = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags = cv2.INTER_LINEAR)
return rot_img
def findVolume(rot_img,ii,density,a,b,deltaSize,cf_resize):
#Finding the volume of each row
deltaSize = deltaSize/float(cf_resize) #Changing the reference size based on the scaling
output = rot_img[ii,a:b]
pixels = len(np.nonzero(output)[0])
radius = deltaSize*pixels
area = np.pi*radius**2
volume = density*area*deltaSize*1 #[g = g/cm^3 *cm^3]. 1 pixel tall
return volume
#Needed to take a reference image to determine the relative size of each pixel
#Using a caliper, 3mm distance in 'ideal' focus, analized in ImageJ
deltaSize = 0.105/61 #Average number of pixel difference = 183. [cm/pixel]
density = 0.998 #[g/cm^3]
cf_resize = 2 #correction factor from the resize in process_drops.py
drop = cv2.imread('/home/pi/nanoRIMS/DROPZ/Trial 14/new_drop_thresh.png')
rot_img = rotateImage(drop,180)
#Have to convert image to greyscale to get rid of RGB
rot_img = cv2.cvtColor(rot_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Rotated Image',rot_img)
cv2.waitKey(0)
198
#caclulate the moment of the binary image
M = cv2.moments(rot_img)
#find centroids (Center of Mass)
col_COM = int(M["m10"] / M["m00"])
row_COM = int(M["m01"] / M["m00"])
copy = rot_img.copy() #have to do this due to the attributes
num_rows,num_cols = copy.shape
print('rows',num_rows,'cols',num_cols)
volumes_leftside = []
volumes_rightside = []
for ii in range(num_rows):
#[0,col_COM+1] --> exclusive end. add 1
volume_left = findVolume(rot_img,ii,density,0,col_COM+1,deltaSize,cf_resize) #defined function
volume_right = findVolume(rot_img,ii,density,col_COM,num_cols+1,deltaSize,cf_resize) #From Column COM to number of columns
volumes_leftside.append(volume_left)
volumes_rightside.append(volume_right)
#masking zero's in array
volumes_leftside = np.ma.masked_equal(volumes_leftside,0)
volumes_rightside = np.ma.masked_equal(volumes_rightside,0)
print('mean left',volumes_leftside.sum())
print('mean right',volumes_rightside.sum())
mean_volume = (volumes_leftside.sum() + volumes_rightside.sum())/float(2)
print('mean mass',mean_volume)
