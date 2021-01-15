import numpy as np 

def convolve2D(input_image, kernel, padding="valid", padding_value=0, stride=1):
    
    input_image = np.array(input_image)
    
    # Gather Shapes of Kernel + Image + Padding
    x_kernel_shape,y_kernel_shape = kernel.shape[:2]
    x_input_shape,y_input_shape = input_image.shape[:2]
    image_to_use = input_image.copy()
    padding_dim = 0
    # Apply Equal Padding to All Sides
    if padding == "same" :
        padding_dim = int(((x_input_shape-1)*stride + x_kernel_shape - x_input_shape)/2) #When Stide = 1 (x_kernel_shape -1 )/2
        
        padded_img_shape = list(input_image.shape)
        padded_img_shape[0] = x_input_shape+padding_dim*2
        padded_img_shape[1] = y_input_shape+padding_dim*2
        
        image_to_use = np.zeros(tuple(padded_img_shape)) + padding_value
        image_to_use[int(padding_dim):int(-1*padding_dim), int(padding_dim):int(-1*padding_dim)] = input_image.copy()
    
    # Using formula
    x_feature_map_shape = int(((x_input_shape - x_kernel_shape + 2 * padding_dim) / stride) + 1)
    y_feature_map_shape = int(((y_input_shape - y_kernel_shape + 2 * padding_dim) / stride) + 1)
    feature_map = np.zeros((x_feature_map_shape, y_feature_map_shape))
    for i in range(x_feature_map_shape):
        for j in range(y_feature_map_shape):
            feature_map[i][j] = np.sum(image_to_use[i:i+x_kernel_shape, j:j+y_kernel_shape]*kernel)

    return feature_map


img = PIL.Image.open("./data_inferance/img_test_conv.jpeg")
plt.imshow(img)
plt.show()

kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
img_conv = convolve2D(img,kernel,'same')
img_conv = PIL.Image.fromarray(img_conv)
plt.imshow(img_conv)
plt.show()