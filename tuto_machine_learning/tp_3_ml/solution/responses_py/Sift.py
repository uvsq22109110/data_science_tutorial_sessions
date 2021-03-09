# retrieving data (paths)
link_dogs = glob("./data/classification_data/raw_data/all_dogs/*")
link_cats = glob("./data/classification_data/raw_data/all_cats/*")

def preprocess_sift_rgb_img(img):
    
    img_gray = rgb2gray(img)
    keypoints, descriptors = sift.detectAndCompute(img_gray,None)
    descriptors_h = list(descriptors.mean(0)[:128])
    descriptors_v = list(descriptors.mean(1)[:30])
    while(len(descriptors_h) < 128):
        descriptors_h.append(0)
    while(len(descriptors_v) < 30):
        descriptors_v.append(0)
    descriptors = descriptors_v+descriptors_h
    return np.array(descriptors)

#Processing image and storing into npy file 
def preprocess_data_links_sift(links, path_to_save):
    sift = cv2.xfeatures2d.SIFT_create(30)
    for link in links :

        filename = ".".join(link.split("/")[-1].split(".")[:-1])
        img = imread(link)
        descriptors = preprocess_sift_rgb_img(img)
        with open(path_to_save+filename+'.npy', 'wb') as f:
            np.save(f, descriptors)
    
preprocess_data_links_sift(link_dogs, './data/classification_data/processed_data/all_dogs/sift/')    
preprocess_data_links_sift(link_cats, './data/classification_data/processed_data/all_cats/sift/')    


link_dogs_sift_fd  = glob("./data/classification_data/processed_data/all_dogs/sift/*")
link_cats_sift_fd  = glob("./data/classification_data/processed_data/all_cats/sift/*")

dogs_sift_fd = np.array([np.load(link) for link in link_dogs_sift_fd])
class_dogs = np.ones((len(dogs_sift_fd)))
dogs_data = np.column_stack((dogs_sift_fd,class_dogs))

cats_sift_fd = np.array([np.load(link) for link in link_cats_sift_fd])
class_cats = np.ones((len(cats_sift_fd)))*2
cats_data = np.column_stack((cats_sift_fd,class_cats))

all_data = np.row_stack((dogs_data, cats_data))
cols = ["sift_descriptor_"+str(i) for i in range(1,159)]
cols.append("label")

df_classficiation = pd.DataFrame(all_data,columns=cols)
df_classficiation = df_classficiation.sample(frac=1,random_state=42)

#Split Data
x_data = df_classficiation[["sift_descriptor_"+str(i) for i in range(1,159)]]
y_data = df_classficiation["label"]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, stratify=y_data, random_state=42)
print(X_train.shape, X_test.shape)

#Model
clf_reg = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
clf_reg.fit(X_train,y_train)
clf_reg.score(X_train,y_train), clf_reg.score(X_test,y_test)

img = imread('./data/examples_data/dog.jpg')

img_descriptors = preprocess_sift_rgb_img(img)
zoomed_img_descriptors = preprocess_sift_rgb_img(zoomed_img)


print("orignal image prediction", clf_reg.predict([img_descriptors]))

img = imread('./data/examples_data/dog.jpg')
rotated_img_15 = skimage.img_as_ubyte(rotate(img,15))
rotated_img_45 = skimage.img_as_ubyte(rotate(img,45))
img_descriptors = preprocess_sift_rgb_img(img)
rotated_img_15_img_descriptors = preprocess_sift_rgb_img(rotated_img_15)
rotated_img_45_img_descriptors = preprocess_sift_rgb_img(rotated_img_45)

print("resized image prediction", clf_reg.predict([img_descriptors]))
print("rotated image 10° prediction", clf_reg.predict([rotated_img_15_img_descriptors]))
print("rotated image 45° prediction", clf_reg.predict([rotated_img_45_img_descriptors]))
