import torch
import torch.optim as optim
from tqdm import tqdm

from style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from utils import preprocess, save_image
from model import VGG19


class ArtStyleTransfer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGG19().to(self.device).eval()
        
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss(self.device)
        self.total_variation_loss = TotalVariationLoss()

    def run_style_transfer(self,
                           content_weight: float = 1e3,
                           style_weight: float = 1e4,
                           style_weight_ratio = {'conv1_1': .2,
                                            'conv2_1': 0.2,
                                            'conv3_1': 0.2,
                                            'conv4_1': 0.2,
                                            'conv5_1': 0.2},
                           total_variation_weight: float = 1.,
                           content_source: str='styles_images/cat.jpg',
                           style_source: str='styles_images/the_scream.jpg',
                           image_size: int=400,
                           style_size: int=400,
                           fname: str=None):
        """
        Style Transfer Execution.

        Args:
            content_weight: The weight assigned to the content loss.
            style_weight: The weight assigned to the style loss.
            style_weight_ratio: A list of ratio of weight to be assigned to each layer in the style_layers.
            total_variation_weight: The weight of the total variation regularization term.
            content_source: The filename of the content image.
            style_source: The filename of the style image.
            image_size: The size of the smallest dimension in the image, which is used for the content loss and generated image.
            style_size: The size of the smallest dimension in the style image.
            fname: The file name to be saved as the styled content image.

        Returns: A tensor representing result image if testing is True.

        """
        # Extract features for content image
        content_img = preprocess(content_source,
                                 size=image_size).to(self.device)
        content_layer = self.model.content_layer
        content_features = self.model(content_img)
        content_target = content_features[
            content_layer].clone().requires_grad_(True).to(self.device)

        # Extract features for style image
        style_img = preprocess(style_source, size=style_size).to(self.device)
        style_layers = self.model.style_layers
        style_features = self.model(style_img)
        style_targets = {layer: self.style_loss.gram_matrix(style_features[layer])
            for layer in style_layers}

        # Initialize output image to content image or noise
        target_img = content_img.clone().requires_grad_(True).to(self.device)

        # Set up optimization hyperparameters
        initial_lr = 1e-3
        epochs = 10000

        optimizer = optim.Adam([target_img], lr=initial_lr)

        print('Building the style transfer model..')
        save_image(1, 2, [content_img, style_img], ["Content", "Style"],
                   f"styles_images/content_and_style.jpg")

        name = f"styles_images/{fname}.jpg" if fname else f"styles_images/img_{content_weight}_{style_weight}.jpg"

        for t in tqdm(range(epochs)):

            target_features = self.model(target_img)

            # loss = content loss + style loss + total variation loss
            loss_content = self.content_loss(content_weight,
                                             target_features[content_layer],
                                             content_target)
            loss_style = self.style_loss(style_weight,
                                         style_weight_ratio,
                                         target_features,
                                         style_targets,
                                         style_layers)
            loss_tv = self.total_variation_loss(target_img, total_variation_weight)
            
            loss = loss_content + loss_style + loss_tv

            # Update target img
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if t % 100 == 0:
                print(f"Total Loss: {loss}")
                save_image(1, 1, [target_img], fname=name)
