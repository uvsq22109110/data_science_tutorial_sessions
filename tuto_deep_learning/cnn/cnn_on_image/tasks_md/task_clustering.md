### Task 

1. What is DenseNet121 ? Could you describe the theoritical principles of DenseNets ?
2. ImageNet (How many classes) and does it contain [Orange, Pizza, Pineapple] ? 
3. Load DenseNet121 pretrained model and show its description (Architecture, Layer succession, Trainable parameters..) 
4. What is the difference when you set a value for include_top [True, False] 
5. Implement neural network that uses the previous pretrained model and ends with a flatten layer which looks like :
<img src="./image_sources/clustering_model.png"/>
6. Use these netwrok to encode images located in **clustering_cnn_embedding**
7. Use some dimensionality reduction (2 components ) and  plot your images on a graph using the **exisiting helper function plot_embedding_images**