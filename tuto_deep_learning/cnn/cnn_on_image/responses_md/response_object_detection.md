1. Object detection was performed using a technique called **image pyramid and sliding window**. This old method will rescale the image to different size to detect bigger objects using a sliding window. Image pyramid and sliding over the whole image has a high cost from object detection algorithm because it is **computationally inefficient (slow)**, it returns very **large number of candidaites regions (possible objects)** and it is **sensitive to hyperparameter choices (Chosen scales, ROI sizes and step size for sliding)**.
<div style="height:250px;text-align:center;">
<img src="https://gurus.pyimagesearch.com/wp-content/uploads/2015/06/sliding_window_car.gif" />
</div>

**Selective Search Region proposal algorithm**  seek to replace the traditional image pyramid and sliding window approach.
This method used in object detection. It is designed to be fast with a very high recall. It is based on computing hierarchical grouping of similar regions based on color, texture, size and shape compatibility. It is build on the idea of **SuperPixel Clustering**. It takes an image as the input and output bounding boxes corresponding to all patches in an image that are most likely to be objects. We note These region proposals can be noisy, overlapping and may not contain the object perfectly but amongst these region proposals, there will be a proposal which will be very close to the actual object in the image.
<div style="height:250px;text-align:center;">
<img src="https://learnopencv.com/wp-content/uploads/2017/09/hierarchical-segmentation-1.jpg" />
</div>

2. **Intersection over Union (IoU)**  is an evaluation metric used to measure the accuracy of an object detector on a particular dataset.
<div style="height:270px;text-align:center;">
<img style="height:250px;width:350px;float:left;" src="https://storage.googleapis.com/kaggle-media/competitions/rsna/IoU.jpg" />
<img style="height:250px;width:350px;float:right;" src="https://pyimagesearch.com/wp-content/uploads/2016/09/iou_examples.png" />   
</div>

In **IoU**, predicted bounding boxes that **heavily overlap** with the ground-truth bounding boxes have **higher scores** than those with **less overlap**. This makes **Intersection over Union an excellent metric for evaluating custom object detectors**.
**The main goal for this metric is to ensure that predicted bounding boxes are the closest possible to the ground truth boxes.**

3. **The purpose of non-max suppression is to select the best bounding box** for an object and **reject or “suppress”** all other bounding boxes. NMS is a class of algorithms to **select one entity (e.g. bounding boxes) out of many overlapping entities**. The selection criteria can be chosen to arrive at particular results. Most commonly, the criteria is some form of probability number along with some form of overlap measure (e.g. IOU).
    
</div>
<div style="height:270px;text-align:center;">
<img style="height:250px;width:250px;float:left;" src="https://miro.medium.com/max/419/1*8EoRC_Xu625eVAquP9ga5w.png" />
<img style="height:250px;width:450px;float:right;" src="https://software.intel.com/sites/default/files/managed/a6/17/image2.gif" /> 
</div>

4. See code below 
5. See code below
6. See code below
7. See code below
8. See code below
10. See code below
11. See code below 
12. See code below
