from style_transfer import ArtStyleTransfer

def main():
    # Generate Images
    stylizer = ArtStyleTransfer()

    run_params1 = {
        'content_weight': 1,
        'style_weight': 1e7,
        'style_weight_ratio': {'conv1_1': 1.,
                          'conv2_1': 0.8,
                          'conv3_1': 0.5,
                          'conv4_1': 0.2,
                          'conv5_1': 0.1},
        'total_variation_weight': 1.,
        'content_source': 'content_images/cat.jpg',
        'style_source': 'styles_images/popart.jpg',
        'image_size': 400,
        'style_size': 400,
        'fname': f'output_images/cat_popart.jpg',
    }

    stylizer.run_style_transfer(**run_params1)


if __name__ == '__main__':
    main()
