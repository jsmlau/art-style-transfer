from typing import List

import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as T
from PIL import Image


def preprocess(image_name: Image, size: int = 512):
    image = Image.open(image_name).convert('RGB')
    transform = T.Compose([
        T.Resize(size),  # scale imported image
        T.ToTensor(),  # transform it into a torch tensor
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
    ])

    # discard the transparent, content_weight channel (that's the :3) and add the batch dimension
    # use unsqueeze(0) to add dimension for the batch size
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def deprocess(img: Image):
    
    img = img.cpu().detach().clone()
    img = img.numpy().squeeze(0)
    img = img.transpose(1,2,0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)

    return img


def imshow(ax, tensor, title=None):
    image = deprocess(tensor)
    ax.axis('off')
    ax.imshow(image)
    if title:
        ax.set_title(title)


def save_image(rows: int,
               cols: int,
               images: List,
               titles: List[str]=None,
               fname: str = "styles_images/new_img"):

    num_img = rows * cols

    f, ax = plt.subplots(rows, cols)
    ax = [ax] if num_img == 1 else ax

    for i in range(num_img):
        if titles:
            imshow(ax[i], images[i], titles[i])
        else:
            imshow(ax[i], images[i])

    plt.savefig(f"{fname}", bbox_inches='tight')
    plt.close()
    print(f"\nImage saved to {fname}")
