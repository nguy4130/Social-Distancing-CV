# Social-Distancing-ComputerVision-Fall2020
This repository has code to calculate the number of social distancing violations given in an image. We implemented a CNN to identify ground plane and perform homography transformation to get bird's eye view of the image. We ran object detection algorithm - YOLOv3 to identify all the persons. If distance between bounding boxes in the bird's eye view is < 6 feet, it is considered a violation of social distancing. 

## Modules in the repository
### YOLOv3
Has source code and pretrained model of YOLOv3 algorithm used for object detection

### Ground Plane Detection
* Code to build ground truth of ground plane using openCV 
* CNN to predict homography matrix

### Distance Estimation
Code to run object detection, warp images and calculate pairwise distances between people and count the number of social distancing violations(< 6 feet apart)

### Baseline
An existing implementation that uses similar approach of object detection and distance estimation to evaluating social distancing 

We used Yang and Yurtsever's paper and code [here](https://github.com/dongfang-steven-yang/social-distancing-monitoring) for the people counting baseline. Please give them a reference if you used their work.
```
@misc{yang2020visionbased,
      title={A Vision-based Social Distancing and Critical Density Detection System for COVID-19}, 
      author={Dongfang Yang and Ekim Yurtsever and Vishnu Renganathan and Keith A. Redmill and Ümit Özgüner},
      year={2020},
      eprint={2007.03578},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
To run the people counting with Faster R-CNN, open the Google Colab Notebook [baseline.ipynb](https://github.com/nguy4130/Social-Distancing-CV/blob/main/baseline.ipynb)

### Images:
Link to our dataset: In form of [individual images](https://drive.google.com/file/d/1XLzIjKbUafkdz5T_jM_RwI44TkizzaaG/view?usp=sharing) and [video](https://drive.google.com/file/d/1XYTxtSbneh4NQOrSUtVovzC1_HzcdvM0/view?usp=sharing) for the baseline model (to run with Faster R-CNN)

