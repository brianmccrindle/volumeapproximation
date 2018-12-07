from picamera import PiCamera
import time
import sys, os
import RPi.GPIO as GPIO
import cv2
import numpy as np
#for looking at images:
##cv2.waitKey(0)
##cv2.destroyAllWindows()
dir_path = os.getcwd() #check current directory
imgInit = cv2.imread(dir_path + '/Trial 5/init.jpg')
drop = cv2.imread(dir_path + '/Trial 14/new_drop_bright.png')
#convert images to grayscale
imgInit = cv2.cvtColor(imgInit,cv2.COLOR_BGR2GRAY)
drop = cv2.cvtColor(drop,cv2.COLOR_BGR2GRAY)
##cv2.namedWindow('Output',cv2.WINDOW_NORMAL) #Change the imshow images to monitor size
imgInit_S = cv2.resize(imgInit,(int(1920/2),int(1080/2)))
imgDrop_S = cv2.resize(drop,(int(1920/2),int(1080/2)))#To fit image on the monitor
#50% Resolution, greater error on image
#Native Resolution: 3280 x 2464
fromCenter = False
r = cv2.selectROI(imgDrop_S,fromCenter) #Region of Interest
cv2.destroyWindow('ROI selector')
imgInit_crop = imgInit_S[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
imgDrop_crop = imgDrop_S[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
cv2.imshow('Cropped Inital Image',imgDrop_crop)
cv2.waitKey(0)
196
#Canny Edge Filter
edges = cv2.Canny(imgDrop_crop,50,150)
##retval,imgThresh = cv2.threshold(imgDrop_crop,150,255,cv2.THRESH_BINARY)
##cv2.imshow('Binary Image',imgThresh)
##cv2.waitKey(0)
cv2.imshow('Edges',edges)
cv2.waitKey(0)
#Dilate and Erode to clean up the edges of the image
kernal_d = np.ones([30,30]) #Rectangular kernals. could be changes to circular, or ellipse
kernal_e = np.ones([20,20])
kernal_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
kernal_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
dil_img = cv2.dilate(edges,kernal_d)
ero_img = cv2.erode(dil_img,kernal_e)
cv2.imshow('Morph',ero_img)
cv2.waitKey(0)
#Binary Image --> Floodfill from (0,0) --> Invert Floodfill image --> Bitwise or
edge_img = ero_img.copy() #MUST copy() the image for this to work
h,w = edge_img.shape[:2]
mask = np.zeros((h+2,w+2),np.uint8)
#floodfill
cv2.floodFill(edge_img,mask,(0,0),255)
cv2.imshow('flood',edge_img)
cv2.waitKey(0)
#Here we should have a large blob containing the droplet
inv_floodFill = cv2.bitwise_not(edge_img)
cv2.imshow('inverse',inv_floodFill)
cv2.waitKey(0)
#We need to fill in the potentail dark areas within the drop from a light source
final_img = ero_img | inv_floodFill
cv2.imshow('final',final_img)
cv2.waitKey(0)
cv2.imwrite('/home/pi/nanoRIMS/DROPZ/Trial 14/thresh_img.png',final_img)
