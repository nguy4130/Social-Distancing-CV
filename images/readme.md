## Dataset for ground plane detection:

Existing datasets to generate a birds eye view image make use of below additional information
* Horizon lines, Vanishing points
* Itrinsic parameters of camera 
* Single target captured by multiple cameras
* Single target captured at mutiple distances

However, our objective is to identify the groud plane in an image from a single image without any information about intrinsic and extrinsic parameters of the camera. To achieve this we put together our dataset with 580 images by web scraping and annotating the ground plane in matlab. 
This is used to train our CNN model to identify homography matrix to convert a given image to bird's eye view. 

Our dataset can be accessed here: https://drive.google.com/file/d/1XLzIjKbUafkdz5T_jM_RwI44TkizzaaG/view
