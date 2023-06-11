# Art Style Transfer

This project utilizes the neural style transfer technique to generate a new image by combining the content of one image with 
the style of another image. The technique involves using a convolutional neural network (CNN) model VGG19 pre-trained on 
ImageNet dataset to extract features from both the content and style images. These features are then used to create a new image 
that preserves the content structure of the content image while adopting the artistic style of the style image.

## How does it work

1. **Content Image**: The content image is the base image whose structure and content will be preserved in the final generated image.

2. **Style Image**: The style image provides the artistic style that will be applied to the content image. The style of the style image is defined by its textures, colors, and overall visual patterns.

3. **VGG19 Model**: The VGG19 model used in this project is a pre-trained convolutional neural network on ImageNet. It 
   extracts style and content features by passing a content image and a style image into convolutional layers. Low-level style features are obtained 
   from early layers, while high-level content features are obtained from a deeper layer.
   
   <img src="https://media.licdn.com/dms/image/C5612AQGvRB_z3dNS0w/article-inline_image-shrink_400_744/0/1595163783759?e=1691625600&v=beta&t=oCjtPxSfu0dzD-5EaUrpuQpqtAYGo0KvpZDYeIJIUUA" height="300">

4. **Content Loss**: The content loss measures the difference between the feature representations of the generated image and 
   the content image. By minimizing this loss, the generated image is encouraged to have similar content to the content image.
   
   [![\\ \mathcal{L}_c = \| F^l(I) - F^l(I_c) \|_F^2 = \sum_{i, j} \left( F^l(I)_{i,j} - F^l(I_c)_{i, j} \right)^2 \\ ](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathcal%7BL%7D_c%20%3D%20%5C%7C%20F%5El(I)%20-%20F%5El(I_c)%20%5C%7C_F%5E2%20%3D%20%5Csum_%7Bi%2C%20j%7D%20%5Cleft(%20F%5El(I)_%7Bi%2Cj%7D%20-%20F%5El(I_c)_%7Bi%2C%20j%7D%20%5Cright)%5E2%20%5C%5C%20)](#_)

5. **Style Loss**: The style loss quantifies the difference between the feature representations of the generated image and the 
   style image. It is computed by comparing the Gram matrices of the feature maps from the selected layers. The **Gram matrix** captures the style information by measuring the correlations between different feature maps.
   
   Style loss:
   
   [![\\ \mathcal{L}_s = \| G(F^l(I)) - G(F^l(I_s)) \|_F^2 = \sum_{i, j} \left( G(F^l(I))_{i, j} - G(F^l(I_s))_{i, j} \right)^2 \\ ](https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathcal%7BL%7D_s%20%3D%20%5C%7C%20G(F%5El(I))%20-%20G(F%5El(I_s))%20%5C%7C_F%5E2%20%3D%20%5Csum_%7Bi%2C%20j%7D%20%5Cleft(%20G(F%5El(I))_%7Bi%2C%20j%7D%20-%20G(F%5El(I_s))_%7Bi%2C%20j%7D%20%5Cright)%5E2%20%5C%5C%20)](#_)
   
   Gram matrix:
   
   [![\\ G^l_{cd} = \frac{\sum_{ij} F^l_{ijc}(x)F^l_{ijd}(x)}{IJ} \\  \\  \\ ](https://latex.codecogs.com/svg.latex?%5C%5C%20G%5El_%7Bcd%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bij%7D%20F%5El_%7Bijc%7D(x)F%5El_%7Bijd%7D(x)%7D%7BIJ%7D%20%5C%5C%20%20%5C%5C%20%20%5C%5C%20)](#_)

6. **Total Variation Loss**: The total variation loss promotes spatial smoothness in the generated image by minimizing the 
   high-frequency noise. It helps to produce visually appealing and coherent results.
   

7. **Optimization**: The optimization process aims to minimize the total loss, which is a weighted sum of the content loss, 
   style loss, and total variation loss. By iteratively updating the generated image using gradient descent, the algorithm gradually refines the image to find a balance between preserving content and adopting style.

8. **Generated Image**: The final output of the style transfer process is the generated image, which combines the content 
   structure of the content image with the artistic style of the style image.


<img src="https://github.com/jsmlau/art-style-transfer/blob/main/content_images/cat.jpg"  height="300"> <img src="https://github.com/jsmlau/art-style-transfer/blob/main/styles_images/starry_night.jpg?raw=true"  height="300"> <img src="https://github.com/jsmlau/art-style-transfer/blob/main/output_images/cat_starry_night.jpg?raw=true"  height="300">

<img src="https://github.com/jsmlau/art-style-transfer/blob/main/content_images/cat.jpg"  height="300"> <img src="https://github.com/jsmlau/art-style-transfer/blob/main/styles_images/composition_vii.jpg?raw=true"  height="300"> <img src="https://github.com/jsmlau/art-style-transfer/blob/main/output_images/cat_composition_vii.jpg?raw=true"  height="300">
