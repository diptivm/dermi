# dermi
## Leveraging unlabeled data for skin cancer classification using the ISIC2019 dataset

We investigate the potential of learning from unlabeled
data for classification of skin lesion images, specifically for
the ISIC (International Skin Imaging Collaboration) 2019
dataset. While there has been significant work in fullysupervised approaches on skin lesion images including ISIC
datasets, and some success in transfer learning from natural images with domain-specific adaptations for skin data,
there has been limited work exploring semi-supervised and
self-supervised techniques in this domain. We experiment
with FixMatch as a semi-supervised approach and BarlowTwins as a self-supervised approach for leveraging unlabeled images and compare performance against a baseline trained only on labeled images. We demonstrate the
effectiveness of these approaches in improving generalization to a held-out test set and in dealing with severe class
imbalance, and identify promising future directions for each
approach.
