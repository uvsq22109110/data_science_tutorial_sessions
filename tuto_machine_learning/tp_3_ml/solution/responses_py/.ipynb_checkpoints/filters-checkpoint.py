
def rgb2gray(rgb):
    
    rgb = rgb/255
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return skimage.img_as_ubyte(gray)


dog_rgb = skimage.io.imread("./data/examples_data/dog.jpg")
dog_gray = rgb2gray(dog_rgb)
dog_sobel_h = sobel_h(dog_gray)
dog_sobel_v = sobel_v(dog_gray)
dog_sobel = sobel(dog_gray)
dog_canny = feature.canny(dog_gray,sigma=4)

fig, axes = plt.subplots(ncols=6, sharex=True, sharey=True,
                         figsize=(20, 8))
axes[0].imshow(dog_rgb)
axes[0].set_title('dog rgb')
axes[1].imshow(dog_gray,cmap=plt.cm.gray)
axes[1].set_title('dog gray')
axes[2].imshow(dog_sobel_h,cmap=plt.cm.gray)
axes[2].set_title('dog sobel horizontal')
axes[3].imshow(dog_sobel_v,cmap=plt.cm.gray)
axes[3].set_title('dog sobel vertical')
axes[4].imshow(dog_sobel,cmap=plt.cm.gray)
axes[4].set_title('dog sobel')
axes[5].imshow(dog_canny,cmap=plt.cm.gray)
axes[5].set_title('dog canny')
plt.show()



#reading the image
img = imread('./data/examples_data/dog.jpg')
resized_img = resize(img, (128,64))
#creating hog features 
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                     visualize=True, multichannel=True)
fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(resized_img, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients')

ax2.imshow(hog_image, cmap=plt.cm.gray) 
ax2.set_title('Histogram of Oriented Gradients')

ax3.imshow(hog_image_normalized, cmap=plt.cm.gray) 
ax3.set_title('Normalized Histogram of Oriented Gradients')

plt.show()



#reading the image
img = imread('./data/examples_data/car.jpg')
resized_img = resize(img, (128,64))
#creating hog features 
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                     visualize=True)
fd, hog_image_normalized = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True) 

ax1.imshow(resized_img, cmap=plt.cm.gray) 
ax1.set_title('Histogram of Oriented Gradients')

ax2.imshow(hog_image, cmap=plt.cm.gray) 
ax2.set_title('Histogram of Oriented Gradients')

ax3.imshow(hog_image_normalized, cmap=plt.cm.gray) 
ax3.set_title('Normalized Histogram of Oriented Gradients')

plt.show()
