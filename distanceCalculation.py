#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:00:58 2020

@author: vcroopana
"""
import numpy as np
from yolo.yolo_v3_predict import getYOLOBBox, BoundBox
from pathlib import Path
from transform import four_point_transform, resizePatches
from keras.models import load_model
import cv2
from matplotlib import pyplot
import scipy.io
import pandas as pd

"""
Input: pPersons = pixel locations of the persons in the image (x,y)
        pixel_meter_conv = 1 pixel distance = ? distance in meters
Output: An array of size num_pixels that contains the original point, 
        its closest neighbor, and the distance between them
        
        no of social distance violations
"""
def getViolations(pPersons, pixel_meter_conv):
    nPersons = len(pPersons)
    nViolations = 0
    for iCurrPix in range(0, nPersons):
        min_dist = 1.8288 # 6 foot = 1.8288 meters
        
        pCurrPix = pPersons[iCurrPix] # pixel location of current person x,y
        for iCompPix in range(iCurrPix+1, nPersons):
            # Calculate point distance
            pCompPix = pPersons[iCompPix]
            dist = np.sqrt((pCurrPix[0]-pCompPix[0])**2 + (pCurrPix[1]-pCompPix[1])**2)
            dist = dist * pixel_meter_conv
            # print("Distance bewteen "+ str(pCurrPix)+" and "+ str(pCompPix)+" is:"+ str(dist))
            if (dist < min_dist):
                nViolations = nViolations+1
    return nViolations

if __name__ == '__main__':

    
    # define the expected input shape for the model
    target_w, target_h = 416, 416
    # target_w, target_h = 320, 240

    yolo_model = load_model('yolo_v3_keras/model.h5')
    print("model loaded")
    
    folderPaths = []
    matFileNames = []
    # folderPaths.append("/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/TinyImgs/")
    # matFileNames.append('BBox_PPM.mat')
    # folderPaths.append("/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/TinyImgs_2/")
    # matFileNames.append('BBox_ConversionRate.mat')
    # folderPaths.append("/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/")
    # matFileNames.append('BBox_Conv.mat')
    folderPaths.append("/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs_2/")
    matFileNames.append('BBox_Conv_2.mat')
    
    names = []
    n_persons = []
    n_viols = []
    avg_viols = []
    v_boxes_list = []
    v_labels_list = []
    v_scores_list = []
    
    reqd_imgs = [4,	7,	11,	17,	19,	21,	22,	25,	28,	32,	33,	36,	40,	46,	48,	58,	62,	63,	65,	66,	67,	80,	85,	89,	92,	93,	94,	97,	98,	103,104,	105,125,	126]
    reqd_img_2 = [0,	6,	8,	11,	15,	16,	18,	24,	25,	26,	29,	31,	36,	52,	60]
    result = pd.DataFrame(columns = ["name", "n_persons", "n_violations", "avg_violations", 'bboxes', 'labels', 'scores'])
    #TODO use label scores to account for accuracy of whole approach
    for k in range(0, len(folderPaths)):
        folderPath = folderPaths[k]
        matFileName = matFileNames[k]
        
        mat = scipy.io.loadmat(folderPath + matFileName)
        if matFileName == 'BBox_Conv.mat':
            mat = list(mat['bbox_coords'])
        elif matFileName == 'BBox_Conv_2.mat':
            mat = list(mat['bbox_conv'])
        else:
            mat = list(mat['img_data'])
        
        for i in range(0, len(mat)):
            
            if i not in reqd_img_2:
                continue
    
            if matFileName == 'BBox_ConversionRate.mat' or matFileName == 'BBox_Conv.mat' or matFileName == 'BBox_Conv_2.mat':
                photo_filename = folderPath+"Img_"+ str(i+1)+".png"
                
            else:
                photo_filename = folderPath+"Img_"+ str(i)+".png"
        
            image = cv2.imread(photo_filename)
            
            if image is not None and len(image.shape) == 3:
               image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
            
            if np.array(mat[i])[0].shape[0] != 4 or np.array(mat[i])[0].shape[1] != 2 :
                continue
            pts = np.array(mat[i])[0]
            
            patch_orig, patch_warped, rect, dst, M, warped, warped_whole = four_point_transform(image, pts)
            # pyplot.imshow(warped_whole)
            # pyplot.show()
            print(photo_filename)
            v_boxes, v_labels, v_scores = getYOLOBBox(yolo_model, photo_filename, target_w, target_h, True) # set to true to draw bboxes
            nPersons = len(v_scores)
            
            v_boxes_coords = [bbox.getCoordinates() for bbox in v_boxes]
            v_boxes_list.append(v_boxes_coords)
            v_labels_list.append(v_labels)
            v_scores_list.append(v_scores)
            
            transformed_bboxes = []
            bottom_lefts = [] # stored in last row  order: (tl, tr, br, bl)
            width_bbox = 0.0
            homo = False
            for bbox in v_boxes:
                yolo_bbox = bbox.getCoordinates()
                currWidth = yolo_bbox[1,1] - yolo_bbox[0,1]
                if currWidth> width_bbox:
                    width_bbox = currWidth
                
                transformed_bbox = yolo_bbox
                
                if homo:
                    
                    for j in range(0, 4): # since 4 vertices for each bbox
                        point_before = np.ones((1,3))
                        point_after = np.zeros((1,2))
                        
                        point_before[0][0]= bbox.getCoordinates()[j][0]
                        point_before[0][1] = bbox.getCoordinates()[j][1]
                        
                        row_3_prod = np.sum(M[2]*point_before)
                        point_after[0][0] = np.sum(M[0]*point_before)/row_3_prod
                        point_after[0][1] = np.sum(M[1]*point_before)/row_3_prod
                        transformed_bbox[j] = point_after
                
                bottom_lefts.append(transformed_bbox[3])
                transformed_bboxes.append(transformed_bbox)
                # print(transformed_bbox)
            
            # ht_person = 1.6616 #Average height of person = 5'45'', 5'7'' male, 5'2'' female , avg = 5.45 
            width_person = 0.4 # Average shoulder width of person = 0.38 + 0.2 for hands etc
            if width_bbox!=0: 
                pixel_meter_conv = width_person/width_bbox
                #find distance between all bottom lefts
                nViolations = getViolations(bottom_lefts, pixel_meter_conv)
                nViol_per_person = nViolations/ len(bottom_lefts)

                print("n violations:"+ str(nViolations))
                print("n violations per person :"+ str(nViol_per_person))
            else:
                nViolations = -1
                nViol_per_person = -1
               
            print("img: "+ str(k)+"_"+ str(i))
            print("n persons:"+ str(nPersons))
            names.append(photo_filename)
            n_persons.append(nPersons)
            n_viols.append(nViolations)
            avg_viols.append(nViol_per_person)            
            
    result['name'] = names
    result['n_persons'] = n_persons
    result['n_violations'] = n_viols
    result['avg_violations'] = avg_viols
    result['bboxes']  = v_boxes_list
    result['labels'] = v_labels_list
    result['scores'] = v_scores_list
    result.to_csv('/Users/vcroopana/python_workspace/compVision/cv_proj_py/results_no_homo_2.csv')
   
   
   
   
   
   
   
   
   
