import argparse
import os
import cv2
import monai
import pandas as pd
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from statistics import mean
from PIL import Image
from tqdm import tqdm


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a segmentation model using Segment Anything model.")

    parser.add_argument('--data_csv', type=str, required=True, help="Path to the CSV file with image annotations.")
    parser.add_argument('--image_root', type=str, required=True, help="Root directory for images.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the SAM model checkpoint.")
    parser.add_argument('--model_type', type=str, default="vit_l", help="Type of SAM model.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=5e-7, help="Learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="Directory to save outputs and model checkpoints.")

    return parser.parse_args()


class CustomDataset(Dataset):
    """Custom dataset class for loading and transforming images."""

    def __init__(self, df, image_root, device='cuda'):
        self.df = df
        self.image_root = image_root
        self.device = device
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and preprocess image and its ground truth."""
        k = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, k['in_file'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0)

        input_image = sam_model.preprocess(input_image_torch.to(self.device))
        original_image_size = image.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])

        gt_path = os.path.join(self.image_root, k['out_file'])
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        sample = {
            'image': input_image,
            'input_size': input_size,
            'original_image_size': original_image_size,
            'bbox': np.array([k['x_min'] * 224, k['y_min'] * 224, k['x_max'] * 224, k['y_max'] * 224]),
            'gt': gt
        }

        return sample


def train_model(args):
    """Train the SAM model."""
    # Load the SAM model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model.to(device='cuda')

    # Load dataset and dataloader
    df = pd.read_csv(args.data_csv)
    dataset = CustomDataset(df, image_root=args.image_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    loss_fn = monai.losses.DiceLoss()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_losses = []
        iou_scores = []

        sam_model.train()
        loader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', dynamic_ncols=True)

        for batch_idx, batch in enumerate(loader):
            input_images = batch['image'][0]
            input_sizes = batch['input_size']
            original_image_sizes = batch['original_image_size']
            prompt_box = batch['bbox'][0].numpy()
            boxes = ResizeLongestSide(sam_model.image_encoder.img_size).apply_boxes(prompt_box, (224, 224))
            boxes_torch = torch.as_tensor(boxes, dtype=torch.float, device='cuda')[None, :]

            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(input_images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch,
                    masks=None,
                )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_sizes, original_image_sizes).to('cuda')
            binary_masks = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_grayscales = batch['gt']
            gt_mask_resized = torch.nn.functional.interpolate(
                torch.as_tensor(gt_grayscales).unsqueeze(1).to('cuda'), size=upscaled_masks.shape[-2:], mode='nearest'
            )
            gt_binary_masks = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            iou_score = monai.metrics.compute_iou(binary_masks.detach().cpu(), gt_binary_masks.detach().cpu())
            loss = loss_fn(binary_masks, gt_binary_masks.to('cuda'))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            iou_scores.append(iou_score.item())
            loader.set_postfix(
                loss=f'{mean(epoch_losses):.6f}',
                last_batch=f'{loss.item():.6f}',
                mean_iou=f'{mean(iou_scores):.6f}',
                last_iou=f'{iou_score.item():.6f}',
                refresh=True
            )

        # Save model checkpoint
        torch.save(sam_model.state_dict(), os.path.join(args.output_dir, f'sam_model_epoch_{epoch + 1}.pth'))
        print(f'EPOCH: {epoch + 1}')
        print(f'Mean loss: {mean(epoch_losses)}')


if __name__ == "__main__":
    args = get_args()
    train_model(args)
