import argparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import torchvision
import os
import pytorch_lightning as pl
from transformers import DetrFeatureExtractor, DetrConfig, DetrForObjectDetection
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm.notebook import tqdm


# Custom COCO dataset class
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "coco_format_data_train.json" if train else "coco_format_data_valid.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


# Detr Model Definition using PyTorch Lightning
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101",
                                                            num_labels=1000,
                                                            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


# Utility functions for inference and evaluation
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = '10'  # Placeholder class label (should be replaced with actual class)
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def visualize_predictions(image, outputs, threshold=0.99, keep_highest_scoring_bbox=True):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    if keep_highest_scoring_bbox:
        keep = probas.max(-1).values.argmax()
        keep = torch.tensor([keep])
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_mape(result_df, valid_df):
    total_ape = 0
    num_samples = len(result_df)
    for index, row in result_df.iterrows():
        pred_box = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
        ground_truth_row = valid_df.loc[valid_df['in_file'] == row['in_file']]
        gt_box = [ground_truth_row['x_min'].values[0], ground_truth_row['y_min'].values[0],
                  ground_truth_row['x_max'].values[0], ground_truth_row['y_max'].values[0]]
        gt_box = [gt_box[0] * 224, gt_box[1] * 224, gt_box[2] * 224, gt_box[3] * 224]
        iou = calculate_iou(pred_box, gt_box)
        ape = np.abs(1 - iou)
        total_ape += ape
    mape = (total_ape / num_samples) * 100
    return mape


def train_model(args):
    # Load feature extractor and datasets
    feature_extractor = DetrFeatureExtractor.from_pretrained(args.model_name)
    train_dataset = CocoDetection(img_folder=f'{args.img_folder}/train', feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=f'{args.img_folder}/val', feature_extractor=feature_extractor, train=False)

    # Dataloader
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False,
                                num_workers=2)

    # Initialize model
    model = Detr(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay)

    # Train and validate
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, log_every_n_steps=50)
    trainer.fit(model, train_dataloader, val_dataloader)


def test_model(args):
    # Load feature extractor and datasets
    feature_extractor = DetrFeatureExtractor.from_pretrained(args.model_name)
    val_dataset = CocoDetection(img_folder=f'{args.img_folder}/val', feature_extractor=feature_extractor, train=False)

    # Dataloader
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False,
                                num_workers=2)

    # Initialize model
    model = Detr.load_from_checkpoint(args.checkpoint_path, lr=args.lr, lr_backbone=args.lr_backbone,
                                      weight_decay=args.weight_decay)

    # Load validation images and visualize predictions
    valid_df = pd.read_csv(f'{args.img_folder}/valid/labels.csv')
    result_df = pd.DataFrame()

    for image_file in tqdm(valid_df['in_file']):
        image = Image.open(os.path.join(args.img_folder, "val", image_file))
        encoding = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
        visualize_predictions(image, outputs)
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0].cpu(), image.size)
        for box in bboxes_scaled:
            result_df = result_df.append({
                'in_file': image_file,
                'x_min': box[0].item(),
                'y_min': box[1].item(),
                'x_max': box[2].item(),
                'y_max': box[3].item()
            }, ignore_index=True)

    mape = calculate_mape(result_df, valid_df)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


def main(args):
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    else:
        raise ValueError("Mode should be 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DETR model on a custom COCO dataset.")

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help="Mode of operation: 'train' to train the model, 'test' to test the model.")
    parser.add_argument('--img_folder', type=str, required=True,
                        help="Path to the folder containing images and annotations.")
    parser.add_argument('--checkpoint_path', type=str, help="Path to the model checkpoint file for testing.")
    parser.add_argument('--model_name', type=str, default="facebook/detr-resnet-101", help="Pre-trained model name.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help="Learning rate for the backbone model.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs.")

    args = parser.parse_args()
    main(args)
