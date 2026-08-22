# ERG-Mask Dataset

This repository contains dataset files associated with the ERG-Mask study, including grape-cluster images, corresponding annotation files, and annotation conversion code. These files provide the necessary dataset components for reproducing the ERG-Mask training and evaluation workflow reported in the study and are provided for academic research purposes.

The images folder contains grape-cluster images. The annotations folder contains the corresponding annotation files, including edge-label PNG files, label visualization files, and label-name information. The edge_json_decode folder contains files generated from the original LabelMe JSON annotations. The json_to_annotations.py script provides the code used to perform this conversion and generate the corresponding label images, label-name files, visualization images, and edge-label images.

Some of the data used in this study were provided by collaborators. We are currently communicating with the data provider and organizing the relevant data, while gradually expanding the publicly available dataset. The data and annotation files currently released can already be used to reproduce the ERG-Mask training and evaluation workflow.

For questions about the dataset, please contact the corresponding author: lijian499@163.com.
