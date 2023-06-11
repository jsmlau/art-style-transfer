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
   from 
   early layers, while high-level content features are obtained from a deeper layer.

4. **Content Loss**: The content loss measures the difference between the feature representations of the generated image and 
   the content image. By minimizing this loss, the generated image is encouraged to have similar content to the content image.

5. **Style Loss**: The style loss quantifies the difference between the feature representations of the generated image and the 
   style image. It is computed by comparing the **Gram matrices** of the feature maps from the selected layers. The Gram matrix captures the style information by measuring the correlations between different feature maps.

6. **Total Variation Loss**: The total variation loss promotes spatial smoothness in the generated image by minimizing the 
   high-frequency noise. It helps to produce visually appealing and coherent results.

7. **Optimization**: The optimization process aims to minimize the total loss, which is a weighted sum of the content loss, 
   style loss, and total variation loss. By iteratively updating the generated image using gradient descent, the algorithm gradually refines the image to find a balance between preserving content and adopting style.

8. **Generated Image**: The final output of the style transfer process is the generated image, which combines the content 
   structure of the content image with the artistic style of the style image.



