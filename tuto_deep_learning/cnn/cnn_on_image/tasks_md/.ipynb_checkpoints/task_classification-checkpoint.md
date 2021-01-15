### Task 

1. Construct a dataframe and **shuffle** (random_state=42) it using all images in **classification_cnn** which looks like
<div style="height:200px;text-align: center;"> 
<img  style="height:200px;width:500px;" src="./image_sources/classification_df.png" title="Classfication dataframe" />
</div>
2. Using LabelBinarinze from Sklearn, preprocess the class of the images.
3. Using the Train test split generate Train and Test set with **20%** of images for test and a **random_state=42**

4. Construct a Keras Data Generator using images filenames, images labels and batch size
5. Based on the DenseNet pre-trained model construct a model which looks like :
<img src="./image_sources/classif_model.png"/>

6. Train your model on your data and save the model history. Use 10 epochs, 3 update per epoch on the train and a batch size of 100 images and an Adam Optimizer with an initial learning rate of 1e-4.

7. Plot the variation of the history 
