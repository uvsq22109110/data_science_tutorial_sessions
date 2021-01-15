import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow.keras.preprocessing.image import *

def plot_embedding_images(images_placement,images_paths):
    
    def getImage(path):
        return OffsetImage(load_img(path,target_size=(32,32)))
    
    fig, ax = plt.subplots(figsize=(15,10))
    ax.scatter(images_placement[:,0],images_placement[:,1]) 

    for (embedding,img_path) in zip(images_placement,images_paths):
        ab = AnnotationBbox(getImage(img_path), tuple(embedding))
        ax.add_artist(ab)
    plt.show()