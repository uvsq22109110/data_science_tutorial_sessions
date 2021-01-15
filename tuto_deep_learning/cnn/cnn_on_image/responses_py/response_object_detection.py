df_detection = pd.read_csv("object_detection_cnn/raw_data/dataframe_object_detection.csv")
print(df_detection.head())

def plot_img_bboxes(in_folder_image_path, bboxes, dir_img_src="./object_detection_cnn/raw_data"):
    
    img = io.imread(dir_img_src+"/"+in_folder_image_path)
    
    fig,ax1 = plt.subplots(1,figsize=(10,10))
    # loop over the bounding box indexes
    #print( boxes[boxIdxs[0]])   
    ax1.imshow(img)
    for box in bboxes:
        # draw the bounding box, label, and probability on the image
        start_x, start_y, end_x, end_y = box
        width_bbox = int(end_x-start_x)  
        height_bbox = int(end_y-start_y)
        rect = patches.Rectangle((start_x, start_y),width_bbox,height_bbox,linewidth=2,edgecolor='g',facecolor='none')
        ax1.add_patch(rect)
        text= "racoon"
        ax1.text(start_x,start_y, text)
    # show the output image *after* running NMS
    plt.show()
    
path_img = "raccoon-12.jpg"
img_df = df_detection[df_detection["filename"]==path_img]
bboxes = img_df[["xmin","ymin","xmax","ymax"]].values
plot_img_bboxes(path_img,bboxes)


def get_selection_search_proposed_regions(image):
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposed_regions= []
    for (x, y, w, h) in rects:
        proposed_regions.append((x, y, x + w, y + h))
    return np.array(proposed_regions)

def generate_dataset(source_dataframe, source_dir="./object_detection_cnn/raw_data/", max_region_to_treat=2000, max_postive_per_image=30, max_negative_per_image=10):

    total_positive,total_negative = 0, 0
    for file_name in source_dataframe["filename"]:

        try:
            #Selecting True regions
            true_regions =  source_dataframe[source_dataframe["filename"]==file_name][["xmin","ymin","xmax","ymax"]].values
            # load the input image from disk
            image = io.imread(source_dir+"/"+file_name)
            #Generate regions with selective search
            proposed_regions = get_selection_search_proposed_regions(image)
            
            #Counter of generated region per image
            positive_roi_per_image = 0
            negative_roi_per_image = 0
            
            # looping other max_region_to_treat selective search results
            for proposed_region in proposed_regions[:max_region_to_treat]:
                # unpack the proposed rectangle bounding box
                (prop_region_start_x, prop_region_start_y, prop_region_end_x, prop_region_end_y) = proposed_region
                # loop over the ground-truth bounding boxes
                for true_region in true_regions:

                    # compute the intersection over union between the two (How good is this proposed region)
                    iou = compute_iou(true_region, proposed_region)
                    (true_region_start_x, true_region_start_y, true_region_end_x, true_region_end_y) = true_region
                    
                    # initialize the ROI and output path
                    roi = None
                    output_path_region_image = None
                    
                    
                    
                    if iou > 0.75 and positive_roi_per_image <= max_postive_per_image:
                        positive_roi_per_image += 1
                        total_positive += 1

                        roi = image[prop_region_start_y:prop_region_end_y, prop_region_start_x:prop_region_end_x]
                        output_path_region_image = "./object_detection_cnn/dataset/racoon/"+str(total_positive)+".jpg"
                        # increment the positive counters

                    # Detect wether proposed region is smaller (Very small) than true region and detecting if true region is inside proposed region
                    regions_are_full_overlaping = prop_region_start_x >= true_region_start_x
                    regions_are_full_overlaping = regions_are_full_overlaping and prop_region_start_y >= true_region_start_y
                    regions_are_full_overlaping = regions_are_full_overlaping and prop_region_end_x <= true_region_end_x
                    regions_are_full_overlaping = regions_are_full_overlaping and prop_region_end_y <= true_region_end_y

                    if not regions_are_full_overlaping and iou < 0.05 and negative_roi_per_image <= max_negative_per_image:
                       
                        negative_roi_per_image += 1
                        total_negative += 1

                        roi = image[prop_region_start_y:prop_region_end_y, prop_region_start_x:prop_region_end_x]
                        output_path_region_image = "./object_detection_cnn/dataset/no_racoon/"+str(total_negative)+".jpg"

                    # check to see if both the ROI and output path are valid
                    if roi is not None and negative_roi_per_image is not None:
                        roi = skimage.transform.resize(roi,(224,224))
                        io.imsave(output_path_region_image, img_as_ubyte(roi))
                        
        except Exception as e:
            print(file_name, str(e))
        
    return total_positive,total_negative

directory = os.listdir("./object_detection_cnn/dataset/no_racoon/") 
if len(directory) == 0: 
    total_positive,total_negative = generate_dataset(df_detection)
    print("We have generated {} region image for racoon and {} region image for no_racoon".format(total_positive,total_negative))
else:
    print("Dataset already generated")
    try:
        print("We have generated {} region image for racoon and {} region image for no_racoon".format(total_positive,total_negative))
    except:
        pass;
# Constructing Train and Test

imgs = glob("./object_detection_cnn/dataset/*/*")
df_racoon = pd.DataFrame(imgs,columns=["path"])
df_racoon["class"] = df_racoon["path"].apply(lambda x : x.split("/")[-2])
df_racoon = df_racoon.sample(frac=1,random_state=42)
print(df_racoon.head())

nb_classes = df_racoon["class"].nunique()
#Generating Train Test
lb_object_detection = LabelBinarizer()
labels_encoded = lb_object_detection.fit_transform(df_racoon["class"])
labels_encoded = to_categorical(labels_encoded)
x_train, x_test, y_train, y_test = train_test_split(df_racoon["path"], labels_encoded,test_size=0.20, random_state=42)


class batch_generator(keras.utils.Sequence) :
  
    def __init__(self, image_paths, labels, batch_size) :
        
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
    
    def __len__(self) :
        return (np.ceil(len(self.image_paths) / float(self.batch_size))).astype(np.int)
  
  
    def __getitem__(self, idx) :
        #Generating batch data
        batch_x = self.image_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        preprocessed_batch_images = []
        for path in batch_x:
            img_keras = load_img(path,target_size=(224,224))
            img_keras = img_to_array(img_keras)
            img_keras = preprocess_input(img_keras)
            preprocessed_batch_images.append(img_keras)
    
            
        preprocessed_batch_images = np.array(preprocessed_batch_images, dtype="float32")
        batch_y_labels = np.array(batch_y_labels)
        return preprocessed_batch_images, batch_y_labels

#Constructing RCNN model classifier
denseNet = DenseNet121(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
#Fixing base cnn network weights
denseNet.trainable = False
#Adding Flat Layer
flat_layer = Flatten()(denseNet.output)
dense_layer_4 = Dense(1024, activation="relu", name="dense_layer_4")(flat_layer)
dropout_layer_2 = Dropout(0.5, name="dropout_layer_2")(dense_layer_4)
dense_layer_3 = Dense(512, activation="sigmoid", name="dense_layer_3")(dropout_layer_2)
dropout_layer_1 = Dropout(0.5,  name="dropout_layer_1")(dense_layer_3)
dense_layer_2 = Dense(256, activation="relu",  name="dense_layer_2")(dropout_layer_1)
dropout_layer_0 = Dropout(0.5,  name="dropout_layer_0")(dense_layer_2)
dense_layer_1 = Dense(128, activation="relu",  name="dense_layer_1")(dropout_layer_0)
class_dense_layer = Dense(nb_classes, activation="sigmoid", name="label_layer")(dense_layer_1)
object_detection_model = Model(inputs=denseNet.input, outputs=class_dense_layer)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
object_detection_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#Training The model  
object_detection_batch_size = 100 
object_detection_update_steps = 20
object_detection_epochs = 10


my_training_batch_generator = batch_generator(x_train,y_train, object_detection_batch_size)
my_validation_batch_generator = batch_generator(x_test,y_test, object_detection_batch_size)

object_detection_history = object_detection_model.fit_generator(my_training_batch_generator,
                steps_per_epoch = object_detection_update_steps,
                epochs = object_detection_epochs,
                verbose = 1,
                validation_data = my_validation_batch_generator)

dict_object_detection_history = object_detection_history.history
plot_history(dict_object_detection_history,object_detection_epochs)

def test_model(model, img_path, lb_to_use, max_region_to_test=2000, threshold_nms=0.3):

    #Preparing images
    image = io.imread(img_path)
    
    #Generate regions with selective search and preprocessing to prepare as input of the network
    preprocessed_batch_images = []
    proposed_regions = get_selection_search_proposed_regions(image)[:max_region_to_test]
    for proposed_region in proposed_regions:
        
        (prop_region_start_x, prop_region_start_y, prop_region_end_x, prop_region_end_y) = proposed_region
        roi = image[prop_region_start_y:prop_region_end_y, prop_region_start_x:prop_region_end_x]
        
        img_keras = skimage.transform.resize(roi,(224,224))
        img_keras = img_as_ubyte(img_keras)
        img_keras = preprocess_input(img_keras)
        preprocessed_batch_images.append(img_keras)
            
    preprocessed_batch_images = np.array(preprocessed_batch_images, dtype="float32")
    proba = model.predict(preprocessed_batch_images)

    #Getting pontential racoons
    labels = lb_to_use.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == "racoon")[0]

    # Selecting bouding boxes for potential racoons
    boxes = proposed_regions[idxs]
    proba = proba[idxs][:, 1] # Selecting probabilities to be racoon  [no_racoon_prob, racoon_prob]

    # Getting only ~100% racoons
    idxs = np.where(proba >= 0.9)
    boxes = boxes[idxs]
    proba = proba[idxs]
    
    fig,ax = plt.subplots(1,2,figsize=(12,6))

    ax[0].imshow(image)
    ax[0].set_title("Wihtout NMS")
    for (box, prob) in zip(boxes, proba):
        # draw the bounding box, label, and probability on the image
        start_x, start_y, end_x, end_y = box
        width_bbox = int(end_x-start_x)  
        height_bbox = int(end_y-start_y)
        rect = patches.Rectangle((start_x, start_y),width_bbox,height_bbox,linewidth=2,edgecolor='g',facecolor='none')
        ax[0].add_patch(rect)
        text= "racoon "+str(np.round(prob,2))
        ax[0].text(start_x,start_y, text)
    
    ax[1].imshow(image)
    ax[1].set_title("Wiht NMS")
    boxes_non_maxima_suppression = non_max_suppression(boxes, threshold_nms)
    for (box, prob) in zip(boxes_non_maxima_suppression, proba):
        # draw the bounding box, label, and probability on the image
        start_x, start_y, end_x, end_y = box
        width_bbox = int(end_x-start_x)  
        height_bbox = int(end_y-start_y)
        rect = patches.Rectangle((start_x, start_y),width_bbox,height_bbox,linewidth=2,edgecolor='g',facecolor='none')
        ax[1].add_patch(rect)
        text= "racoon"
        ax[1].text(start_x,start_y, text)
    
    plt.show()
   
test_model(object_detection_model, "./data_inferance/racoon.jpeg",lb_object_detection, max_region_to_test=2000, threshold_nms=0.3) 