import torch
from torch.utils.data import DataLoader
from dataLoader import ImageDataset, InfiniteSampler, train_transform
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import denormalize
from adain import AdaIN, save_AdaIn
from pathlib import Path
from PIL import Image, ImageFile
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description='AdaIN Module Training Script')
    # Directories arguments
    parser.add_argument('-cd', '--content_dir', type=str, metavar='<dir>', required=True,
                        help='Directory to content images')
    parser.add_argument('-sd', '--style_dir', type=str, metavar='<dir>', required=True,
                        help='Directory to style images')

    # Optional arguments for training
    parser.add_argument('--continual', type=str, metavar='<.pth>', default=None,
                        help='File to save and load for continual training, default=disabled')

    parser.add_argument('--save_dir', type=str, metavar='<dir>', default='./experiments',
                        help='Directory to save trained models, default=./experiments')
    parser.add_argument('--log_dir', type=str, metavar='<dir>', default='./logs',
                        help='Directory to save logs, default=./logs')
    parser.add_argument('--log_image_every', type=int, metavar='<int>', default=200,
                        help='Interval for logging generated images, negative for disabling, default=200')
    parser.add_argument('--save_interval', type=int, metavar='<int>', default=10000,
                        help='Interval for saving model, default=10000')
    parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
    parser.add_argument('--n-threads', type=int, metavar='<int>', default=2,
                        help='Number of threads used for dataloader, default=2')

    # hyper-parameters
    parser.add_argument('--learning_rate', type=float, metavar='<float>', default=1e-4,
                        help='Learning rate, default=1e-4')
    parser.add_argument('--learning_rate_decay', type=float, metavar='<float>', default=5e-5,
                        help='Learning rate decay, default=5e-5')
    parser.add_argument('--max_iter', type=int, metavar='<int>', default=160000,
                        help='Maximum number of iteration, default=15000')
    parser.add_argument('--batch_size', type=int, metavar='<int>', default=8, help='Size of the batch, default=8')
    parser.add_argument('--content_weight', type=float, metavar='<float>', default=1.0,
                        help='Weight of content loss, default=1.0')
    parser.add_argument('--style_weight', type=float, metavar='<float>', default=10.0,
                        help='Weight of style loss, default=10.0')

    args = parser.parse_args()

    # For error handling
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Use GPU when it's available
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print("Using {} at the moment.".format(device))
    # Make new directory to save if not created already
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Make directory for logs if not created already
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Log writer
    writer = SummaryWriter(log_dir=str(log_dir))

    # Content dataset
    content_dataset = ImageDataset(args.content_dir, train_transform((512, 512), 256))
    print(len(content_dataset))
    content_iter = iter(DataLoader(content_dataset, batch_size=args.batch_size,
                                   sampler=InfiniteSampler(len(content_dataset)),
                                   num_workers=args.n_threads))

    # Style dataset
    style_dataset = ImageDataset(args.style_dir, train_transform((512, 512), 256))
    style_iter = iter(DataLoader(style_dataset, batch_size=args.batch_size,
                                 sampler=InfiniteSampler(len(style_dataset)),
                                 num_workers=args.n_threads))

    # AdaIN model
    model = AdaIN()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Continue training from what was left off
    start_iter = 0
    if args.continual:
        if os.path.exists(args.continual):
            state_dict = torch.load(args.continual)
            start_iter = state_dict['iter']
            model.decoder.load_state_dict(state_dict['decoder'])
            optimizer.load_state_dict(state_dict['optimizer'])

    # Start training
    model.to(device)
    model.train()
    for i in tqdm(range(start_iter, args.max_iter)):

        # Get images
        content_imgs = next(content_iter).to(device)
        style_imgs = next(style_iter).to(device)

        # Compute training loss
        out_img, loss_content, loss_style = model(content_imgs, style_imgs)
        loss_content = args.content_weight * loss_content
        loss_style = args.style_weight * loss_style
        loss = loss_content + loss_style

        # Optimize the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Change learning rate
        if model.training:
            exp_lr_scheduler.step()

        # Logging
        writer.add_scalar('Loss/Loss_content', loss_content.item(), i + 1)
        writer.add_scalar('Loss/Loss_style', loss_style.item(), i + 1)
        writer.add_scalar('Loss/Total Loss', loss.item(), i + 1)

        if args.log_image_every > 0 and ((i + 1) % args.log_image_every == 0 or i == 0 or (i + 1) == args.max_iter):
            content_img = denormalize(content_imgs[0], device)
            style_img = denormalize(style_imgs[0], device)
            out_img = denormalize(out_img[0], device)
            writer.add_image('Image/Content', content_img, i + 1)
            writer.add_image('Image/Style', style_img, i + 1)
            writer.add_image('Image/Generated', out_img, i + 1)

        # Save model
        if (i + 1) % args.save_interval == 0 or (i + 1) == args.max_iter:
            save_AdaIn(model, os.path.join(save_dir, 'iter_{}.pth'.format(i + 1)))
            # continual training
            if args.continual:
                decoder_dict = model.decoder.state_dict()
                optimizer_dict = optimizer.state_dict()
                torch.save({
                            'iter': i + 1,
                            'decoder': decoder_dict,
                            'optimizer': optimizer_dict
                            }, args.continual)
    writer.close()

if __name__ == '__main__':
    main()