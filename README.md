# DETR-SAM: Automated Few-Shot Segmentation

## Overview
DETR-SAM is a novel method that leverages the DEtection TRansformer (DETR) and keypoint matching to automate the few-shot segmentation (FSS) process. This approach aims to segment objects in images using minimal annotated examples, enhancing segmentation accuracy without relying on extensive manual prompts.

## Authors
- Mohamadreza Khanmohamadi
  - Department of Computer and Data Science, Shahid Beheshti University, Tehran, Iran
  - Email: m.khanmohamadi@mail.sbu.ac.ir
- Bahar Farahani
  - Cyberspace Research Institute, Shahid Beheshti University, Tehran, Iran
  - Email: bfarahani@sbu.ac.ir

## Abstract
Few-shot segmentation (FSS) facilitates segmenting objects in images with limited annotated examples. The Segment Anything Model (SAM) has been successful in FSS but typically requires manual prompts, which can lead to inaccuracies. Our proposed method, DETR-SAM, integrates DETR with keypoint matching to automatically generate prompts for SAM, significantly improving accuracy and reducing ambiguity in segmentation tasks.

## Key Contributions
- **Automated Prompt Generation**: Reduces the need for manual prompts, improving contextual understanding.
- **Enhanced Segmentation**: Combines bounding boxes from DETR and points from keypoint matching to achieve highly accurate segmentation without extensive fine-tuning.
- **Performance**: Achieved an 84.3 mean Intersection over Union (mIoU) score on the FSS-1000 dataset, demonstrating competitive performance against state-of-the-art models.

## Methodology
1. **DETR**: A single-stage object detection model that predicts bounding boxes and enhances prompt generation.
2. **Keypoint Matching**: Utilized to identify keypoints between support and query images, enhancing the SAM's attention to important regions.
3. **SAM Architecture**:
   - Image Encoder: Extracts features from input images using a Vision Transformer (ViT).
   - Prompt Encoder: Filters critical features based on human inputs.
   - Light Mask Decoder: Generates segmentation masks based on prompts and extracted features.

## Experimental Setup
- **Datasets**: Evaluated on the FSS-1000 dataset, containing 10,000 images across 1,000 classes with pixel-level annotations.
- **Evaluation Metric**: Mean Intersection over Union (mIoU) score used for performance assessment.

## Results
Comparison of DETR-SAM against baseline models on the FSS-1000 dataset:

| Model               | 1-shot | 5-shot |
|---------------------|--------|--------|
| OSLMS               | 70.3   | 73.0   |
| GNet                | 71.9   | 74.3   |
| FSS 1000            | 73.5   | 80.1   |
| DETR-SAM (H)       | 81.2   | 84.3   |

## Limitations
- The method may struggle in scenarios with illumination changes or deformations between the query and support images, necessitating further advancements in keypoint detection techniques.

## Conclusion
DETR-SAM demonstrates a promising advancement in few-shot segmentation, achieving competitive results while maintaining resilience against overfitting.

## References
For detailed reference and further reading, please refer to the original paper and related works cited within.
