#Load DenseNet pre-trained model 
denseNet = DenseNet121(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
#Adding Flat Layer
flat_layer = Flatten()(denseNet.output)
#Building Siamse network to encode image in new dimension space
clustering_model = Model(inputs=denseNet.input, outputs = flat_layer)

#Showing Architecture of the Network
#clustering_model.summary()

#Getting data from clustering_cnn_embedding
images = glob("./clustering_cnn_embedding/*/*")
print(images[:2])

#Preprocessing images and passing through the network
preprocessed_images = []
for image in images:
    img_keras = load_img(image,target_size=(224,224))
    img_keras = img_to_array(img_keras)
    img_keras = preprocess_input(img_keras)
    preprocessed_images.append(img_keras)
    
preprocessed_images = np.array(preprocessed_images)
print(preprocessed_images.shape)
embedded_images = clustering_model.predict(preprocessed_images)

#Dimentionnality reduction using PCA
pca = PCA(n_components=2,random_state=42)
reduced_images_dimensions = pca.fit_transform(embedded_images)

#Ploting the embedded images
plot_embedding_images(reduced_images_dimensions,images)