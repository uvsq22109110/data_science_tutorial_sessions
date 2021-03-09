# retrieving data (paths)
link_dogs = glob("./data/classification_data/raw_data/all_dogs/*")
link_cats = glob("./data/classification_data/raw_data/all_cats/*")

#Processing image and storing into npy file 
def preprocess_data_links(links, path_to_save):

    for link in links :

        filename = ".".join(link.split("/")[-1].split(".")[:-1])
        img = imread(link)
        resized_img = resize(img, (128,64))
        fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

        with open(path_to_save+filename+'.npy', 'wb') as f:
            np.save(f, fd)
    
preprocess_data_links(link_dogs, './data/classification_data/processed_data/all_dogs/hog/')    
preprocess_data_links(link_cats, './data/classification_data/processed_data/all_cats/hog/')    


#Constructing DataFrame
link_dogs_hog_fd  = glob("./data/classification_data/processed_data/all_dogs/hog/*")
link_cats_hog_fd  = glob("./data/classification_data/processed_data/all_cats/hog/*")

dogs_hog_fd = np.array([np.load(link) for link in link_dogs_hog_fd])
class_dogs = np.ones((len(dogs_hog_fd)))
dogs_data = np.column_stack((dogs_hog_fd,class_dogs))

cats_hog_fd = np.array([np.load(link) for link in link_cats_hog_fd])
class_cats = np.ones((len(cats_hog_fd)))*2
cats_data = np.column_stack((cats_hog_fd,class_cats))

all_data = np.row_stack((dogs_data, cats_data))

cols = ["fd_"+str(i) for i in range(1,3781)]
cols.append("label")
df_classficiation = pd.DataFrame(all_data,columns=cols)
df_classficiation = df_classficiation.sample(frac=1,random_state=42)

#Split Data
x_data = df_classficiation[["fd_"+str(i) for i in range(1,3781)]]
y_data = df_classficiation["label"]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, stratify=y_data, random_state=42)
print(X_train.shape, X_test.shape)

#Model
clf_reg = LogisticRegression(random_state=42,max_iter=500)
clf_reg.fit(X_train,y_train)
clf_reg.score(X_train,y_train), clf_reg.score(X_test,y_test)

img = imread('./data/examples_data/dog.jpg')
zoomed_img = imread('./data/examples_data/dog_zoomed.jpg')
resized_img = resize(img, (128,64))
fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

print("orignal image prediction", clf_reg.predict([fd]))

resized_img = resize(zoomed_img, (128,64))
fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

print("zoomed image prediction", clf_reg.predict([fd]))


img = imread('./data/examples_data/dog_zoomed.jpg')
rotated_img_15 = rotate(img,15,resize=True)
rotated_img_45 = rotate(img,45,resize=True)

resized_img = resize(img, (128,64))
rotated_img_15 = resize(rotated_img_15, (128,64))
rotated_img_45 = resize(rotated_img_45, (128,64))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(resized_img, cmap=plt.cm.gray) 
ax1.set_title('Resized image')

ax2.imshow(rotated_img_15, cmap=plt.cm.gray) 
ax2.set_title('Rotated image 15°')

ax3.imshow(rotated_img_45, cmap=plt.cm.gray) 
ax3.set_title('Rotated image 45°')
plt.show()

fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

print("resized image prediction", clf_reg.predict([fd]))
fd, hog_image_normalized = hog(rotated_img_15, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
print("rotated image 15° prediction", clf_reg.predict([fd]))


fd, hog_image_normalized = hog(rotated_img_45, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
print("rotated image 45° prediction", clf_reg.predict([fd]))