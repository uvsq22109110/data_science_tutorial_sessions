# %load responses_py/response_classification.py
#Getting Images
paths = glob("./classification_cnn/*/*")
#Generating dataframe
df_classification = pd.DataFrame(paths,columns=["path"])
df_classification["class"] = df_classification["path"].apply(lambda x : x.split("/")[-2])
#Shuffling dataframe
df_classification = df_classification.sample(frac=1,random_state=42)


#nb_classes
nb_classes = df_classification["class"].nunique()

#Generating Train Test
lb_classification = LabelBinarizer()
labels_encoded = lb_classification.fit_transform(df_classification["class"])
x_train, x_test, y_train, y_test = train_test_split(df_classification["path"], labels_encoded,test_size=0.20, random_state=42)

#Printing dataframe
print(df_classification.head())


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

#Constructing classification model
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
class_dense_layer = Dense(nb_classes, activation="softmax", name="label_layer")(dense_layer_2)
classification_model = Model(inputs=denseNet.input, outputs=class_dense_layer)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
classification_model.compile(loss="CategoricalCrossentropy", optimizer=opt, metrics=["accuracy",tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


#Training The model  
classification_batch_size = 100 
classification_update_steps =  3
classification_epochs = 10

my_training_batch_generator = batch_generator(x_train,y_train, classification_batch_size)
my_validation_batch_generator = batch_generator(x_test,y_test, classification_batch_size)

classification_history = classification_model.fit_generator(my_training_batch_generator,
                steps_per_epoch = classification_update_steps,
                epochs = classification_epochs,
                verbose = 1,
                validation_data = my_validation_batch_generator)

dict_classification_history = classification_history.history
plot_history(dict_classification_history,classification_epochs)
