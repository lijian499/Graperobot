# ERG-Mask Core Implementation

This directory provides the core reference implementation of ERG-Mask described in the manuscript:

**ERG-Mask: An edge–region collaborative instance segmentation model for greenhouse table-grape clusters**

## Code structure

- `resnet.py`  
  Modified ResNet-101 backbone used to extract the multi-level feature maps C1-C5.

- `fpn.py`  
  Top-down feature pyramid network for multi-scale feature fusion.

- `pan.py`  
  Bottom-up path aggregation network for enhancing multi-scale feature interaction.

- `edges.py`  
  Edge-prediction subnetwork, including multi-level side outputs, feature fusion, and the class-balanced edge loss.

- `anchors.py`  
  IoU-based k-means implementation used for anchor generation.

- `contour_reconstruction.py`  
  Polar-coordinate contour reconstruction module. It uses 16 radial rays with an angular interval of 22.5°, an edge threshold of 0.5, 8-connected component matching, missing-ray interpolation, and morphological closing to reconstruct grape-cluster instance masks.

## Pretrained backbone

The ResNet-101 backbone can be initialized using ImageNet-1K pretrained weights provided by `torchvision`.

## Dependencies

The main dependencies include:

- Python 3
- PyTorch
- torchvision
- NumPy
- OpenCV

## Dataset

The dataset used in this study is provided in the `dataset` directory of the ERG-Mask repository.
