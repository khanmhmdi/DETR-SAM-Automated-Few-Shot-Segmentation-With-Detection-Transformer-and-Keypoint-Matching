import os
import cv2
import numpy as np
import pandas as pd
import statistics
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM models on validation set")
    parser.add_argument('--checkpoints', nargs='+', help='Paths to SAM model checkpoints', required=True)
    parser.add_argument('--validation-csv', type=str, help='Path to validation CSV file', required=True)
    parser.add_argument('--train-csv', type=str, help='Path to training CSV file', required=True)
    parser.add_argument('--output', type=str, help='Path to save results', default='results.txt')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to use for inference')
    return parser.parse_args()


# Function to display annotations on the image
def show_anns(anns, axes=None):
    if not anns:
        return
    ax = axes if axes else plt.gca()
    ax.set_autoscale_on(False)
    # Sort annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # Overlay each annotation with a random color
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.5)))


# Function to display a mask with optional random color
def show_mask(mask, ax, random_color=False):
    color = np.random.random(3) if random_color else np.array([30 / 255, 144 / 255, 255 / 255])
    mask_image = np.dstack((mask * color[0], mask * color[1], mask * color[2], np.full_like(mask, 0.6)))
    ax.imshow(mask_image)


# Function to display keypoints on the image
def show_points(coords, labels, ax, marker_size=375):
    ax.scatter(coords[labels == 1][:, 0], coords[labels == 1][:, 1], color='green', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)
    ax.scatter(coords[labels == 0][:, 0], coords[labels == 0][:, 1], color='red', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)


# Function to display bounding box
def show_box(box, ax):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))

def sample_points_from_mask(mask, num_samples=5):
    mask = np.int8(mask)
    coords_true = np.column_stack((np.where(mask == 1)[1], np.where(mask == 1)[0]))
    coords_false = np.column_stack((np.where(mask == 0)[1], np.where(mask == 0)[0]))
    sampled_coords_true = coords_true[np.random.choice(coords_true.shape[0], size=num_samples, replace=False)]
    sampled_coords_false = coords_false[np.random.choice(coords_false.shape[0], size=num_samples, replace=False)]
    labels_true = np.ones((num_samples,), dtype=int)
    labels_false = np.zeros((num_samples,), dtype=int)
    sampled_coords = np.vstack([sampled_coords_true, sampled_coords_false])
    labels = np.concatenate([labels_true, labels_false])
    random_order = np.random.permutation(2 * num_samples)
    sampled_coords = sampled_coords[random_order]
    labels = labels[random_order]
    return sampled_coords, labels

def get_support_paths(label, train_df):
    specific_rows = train_df[train_df['in_file'].str.contains(label)]
    support_mask_paths = [i.replace('./', 'C:/Users/Khanmhmdi/Desktop/FSS1000/Files/FSS-1000/') for i in specific_rows['out_file'].tolist()]
    return support_mask_paths

def find_most_similar_mask(support_masks, sam_outputs):
    def flatten_and_normalize(mask):
        flat_mask = mask.flatten()
        norm = np.linalg.norm(flat_mask)
        return flat_mask if norm == 0 else flat_mask / norm

    def calculate_cosine_similarity(mask1, mask2):
        vec1 = flatten_and_normalize(mask1)
        vec2 = flatten_and_normalize(mask2)
        return cosine_similarity([vec1], [vec2])[0][0]

    def resize_to_shape(mask, target_shape):
        mask = mask.astype(np.uint8)
        return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    target_shape = support_masks[0].shape
    resized_sam_outputs = [resize_to_shape(pred_mask, target_shape) for pred_mask in sam_outputs]
    similarities = [calculate_cosine_similarity(support_masks[0], pred_mask) for pred_mask in resized_sam_outputs]
    most_similar_index = np.argmax(similarities)
    return most_similar_index

def get_matched_keypoints_coords(support_image_path, query_image_path, show_plot=False):
    support_image = cv2.imread(support_image_path, cv2.IMREAD_GRAYSCALE)
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    detector = cv2.SIFT_create()
    support_keypoints, support_descriptors = detector.detectAndCompute(support_image, None)
    query_keypoints, query_descriptors = detector.detectAndCompute(query_image, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(support_descriptors, query_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    support_coords = [support_keypoints[m.queryIdx].pt for m in good_matches]
    query_coords = [query_keypoints[m.trainIdx].pt for m in good_matches]
    if show_plot:
        img_matches = cv2.drawMatches(support_image, support_keypoints, query_image, query_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return support_coords, query_coords

def filter_coords_by_mask(coords, mask):
    return [[x, y] for x, y in coords if mask[x][y] > 0]

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

def main(args):
    checkpoints = [{"path": path, "model_type": path.split('_')[-1].split('.')[0]} for path in args.checkpoints]
    device = args.device

    df_val = pd.read_csv(args.validation_csv)
    df_train = pd.read_csv(args.train_csv)

    df_val['out_file'] = df_train['out_file']
    df_val['in_file'] = df_train['in_file']

    results = []

    for checkpoint_info in checkpoints:
        sam_checkpoint = checkpoint_info["path"]
        model_type = checkpoint_info["model_type"]

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        C = tqdm(df_val.iterrows(), desc=f'EVALUATING {model_type}', dynamic_ncols=True)
        iou_scores = []
        model_scores_point = []

        for index, row in C:
            image_path = row['in_file'].replace('./', 'C:/Users/Khanmhmdi/Desktop/FSS1000/Files/FSS-1000/')
            image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            predictor.set_image(image_array)
            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            input_box = np.array([x_min, y_min, x_max, y_max])
            ground_truth_path = row['out_file'].replace('./', 'C:/Users/Khanmhmdi/Desktop/FSS1000/Files/FSS-1000/')
            ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE) > 0

            masks, scores_point, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )

            label = row['in_file'].split('/')[1]
            support_masks = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0 for path in get_support_paths(label, df_train)]

            scores_point = [calculate_iou(mask, ground_truth_mask) for mask in masks]
            max_iou_index = scores_point.index(max(scores_point))

            input_points, input_labels = sample_points_from_mask(masks[max_iou_index], num_samples=5)
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            masks, scores_point, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box[None, :],
                multimask_output=True,
            )

            iou_score = calculate_iou(np.stack(masks), ground_truth_mask)
            iou_scores.append(iou_score)
            scores_point.sort()
            model_scores_point.append(scores_point[-1])
            C.set_postfix(mean_iou=f'{statistics.mean(iou_scores):.4f}', refresh=True)

        mean_iou = statistics.mean(iou_scores)
        results.append((sam_checkpoint, model_type, mean_iou))

    with open(args.output, "w") as file:
        for checkpoint, model_type, mean_iou in results:
            file.write(f"Checkpoint: {checkpoint}, Model Type: {model_type}, Mean IoU: {mean_iou}\n")

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
