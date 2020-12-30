import argparse
import torch
from dataLoader import test_transform
from utils import denormalize
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from utils import color_control
from adain import load_AdaIN


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer Testing Script')

    # Required parameters
    # Content images or folder that contains content images, mutually exclusive
    content_group = parser.add_mutually_exclusive_group(required=True)
    content_group.add_argument('-c', '--content', type=str, metavar='<file>', nargs='+', help='Content image file(s)')
    content_group.add_argument('-cd', '--content-dir', type=str, metavar='<dir>', help='Directory with content images')

    # Style images or folder that contains style images, also mutually exclusive
    style_group = parser.add_mutually_exclusive_group(required=True)
    style_group.add_argument('-s', '--style', type=str, metavar='<file>', nargs='+', help='Style image file(s)')
    style_group.add_argument('-sd', '--style-dir', type=str, metavar='<dir>', help='Directory with style images')

    parser.add_argument('-m', '--model', type=str, required=True, metavar='<pth>', help='Trained AdaIN Transfer model')

    # Optional arguments
    parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
    parser.add_argument('--content-size', type=int, metavar='<int>', default=512,
                        help='Size for resizing content images, 0 for original size, default=512')
    parser.add_argument('--style-size', type=int, metavar='<int>', default=512,
                        help='Size for resizing style images, 0 for original size, default=512')
    parser.add_argument('--crop', action='store_true',
                        help='Option for center crop')
    parser.add_argument('--ext', type=str, metavar='<ext>', default='.jpg',
                        help='Extension name for the generated images, default=.jpg')
    parser.add_argument('--output', type=str, metavar='<dir>', default='./outputs',
                        help='Directory to save output images, default=./outputs')

    # Advanced options
    parser.add_argument('--alpha', type=float, metavar='<float>', default=1.0,
                        help='Option for degree of stylization, should be between 0 and 1, default=1.0')
    # Preserve color of the content or have weights for the style images.
    advanced_group = parser.add_mutually_exclusive_group()
    advanced_group.add_argument('--preserve-color', action='store_true',
                                help='Preserving color in generated images')
    advanced_group.add_argument('--interpolation-weights', type=int, metavar='<int>', nargs='+',
                                help='Weights of style images for interpolation')
    args = parser.parse_args()

    # Enable GPU when it's avaiable
    device = torch.device('cuda' if args.cuda and torch.cuda.is_avaiable() else 'cpu')

    # Directory for saving output images
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    assert args.content or args.content_dir, "Please provide a content image or a directory that has content image(s)"
    # Path to content images
    if args.content:
        contents = [Path(c) for c in args.content]
    else:
        content_dir = Path(args.content_dir)
        contents = list(content_dir.glob('**/*.*'))

    assert args.style or args.style_dir, "Please provide a style image or a directory that has style image(s)"
    # Path to style images
    if args.style:
        styles = [Path(s) for s in args.style]
    else:
        style_dir = Path(args.style_dir)
        styles = list(style_dir.glob('**/*.*'))

    # If interpolation weights are provided, combine style images with weights
    if args.interpolation_weights:
        assert len(styles) == len(
            args.interpolation_weights), 'All style images should be weighted. {} images are given while {} weights are given.'.format(
            len(styles), len(args.interpolation_weights))
        interpolation = True
        weights_sum = sum(args.interpolation_weights)
        interpolation_weights = [w / weights_sum for w in args.interpolation_weights]  # Normalize
        interpolation_weights = torch.tensor(interpolation_weights)
        interpolation_weights = interpolation_weights.to(device)
    else:
        interpolation = False

    # Test transforms
    content_transform = test_transform(args.content_size, args.crop or interpolation)
    style_transform = test_transform(args.style_size, args.crop or interpolation)

    # Load the trained model
    model = load_AdaIN(args.model)
    model.to(device)
    model.decoder.eval()
    model.encoder.eval()
    model.eval()
    # If interpolation with N content imgs and M style imgs, output N imgs, else output N*M imgs
    total, iteration = len(contents)*(1 if interpolation else len(styles)), 0
    for content_path in contents:
        if interpolation:
            iteration += 1
            print('[{}/{}] Content: {}, Style: interpolation'.format(iteration, total, content_path))

            # Get style images and stack up them up as a batch
            style = torch.stack([style_transform(Image.open(str(img))) for img in styles]).to(device)
            # Get a content image
            content = content_transform(Image.open(str(content_path))).unsqueeze(0).to(device)

            # Generate a styled image with interpolated styles
            with torch.no_grad():
                output = model(content, style, interpolation_weights=interpolation_weights)
            output = output.cpu()

            # Save the generated image
            save_image(output, str(output_dir / '{}_interpolation{}'.format(content_path.stem, args.ext)))

        else:
            # Process one content img and one style img
            for style_path in styles:
                iteration += 1
                print('[{}/{}] Content image: {}, Style image: {}'.format(iteration, total, content_path, style_path))
                # One content image
                content = content_transform(Image.open(str(content_path)))
                content = content.unsqueeze(0).to(device)

                # One style image
                style = style_transform(Image.open(str(style_path))).to(device)
                # Change the color of the style image to that of content image
                if args.preserve_color:
                    style = color_control(style, content)
                style = style.to(device).unsqueeze(0)

                # Generate a styled picture
                with torch.no_grad():
                    output = model(content, style, alpha=args.alpha)
                output = denormalize(output[0],device).cpu()
                # Save the generated image
                save_image(output,
                           str(output_dir / '{}_stylized_{}{}'.format(content_path.stem, style_path.stem, args.ext)))

if __name__ == '__main__':
    main()