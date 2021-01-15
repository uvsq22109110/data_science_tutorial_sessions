### Task 

1. Define **Region proposal algorithm** 
2. Define **IoU** (Intersection Over Union)
3. Define **NMS** (Non-Max Suppression) algorithm and how does it works
4. Read the dataframe from **object_detection_cnn/raw_data** directory
5. Plot the **image "raccoon-12.jpg" and its corresponding bounding boxes**
6. Implement a function which generates **2000 Regions** for each image then save this regions into folder  **object_detection_cnn** directory in subfolders **dataset/[racoon,no_racoon]** to annotate if regions contains racoons
7. Implement train & test datasets using TrainTestSplit from **Sklearn (20% Test)**
8. Based on the DenseNet pre-trained model construct a model which looks like
<img src="./image_sources/detection_model.png"/> 
9. Train the object detection model (PS : You have computed classes frequencies)
10. Prepare a pipeline for inferance. (**Try to see the difference between results with and without NMS**)
11. Use this images as test image <a href="https://www.nps.gov/guis/images/110617_Racoon_Odom_01.jpg"> link </a>
