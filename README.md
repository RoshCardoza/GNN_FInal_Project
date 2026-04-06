# Low-Data Breast Lesion Segmentation with Synthetic Data Augmentation

This project studies **breast lesion segmentation in ultrasound images** under **limited-data settings** using the **BUSI (Breast Ultrasound Images Dataset)**. The core idea is to evaluate whether synthetic data can improve a U-Net segmentation model when only a small fraction of the training data is available.

The repository contains experiments for three training settings:

1. **Real-only baseline** – train U-Net on a small real subset
2. **Classical augmentation** – train U-Net with standard geometric augmentation
3. **Synthetic augmentation** – expand the low-data training set using synthetic samples

> **Important:** this project does **not** train on the complete BUSI dataset for the main experiments.  
> The code first creates a **70% / 15% / 15% train/validation/test split**, and then uses only **5%, 10%, and 25% of the training split** for the low-data experiments.

---

## Project Motivation

Medical image segmentation usually needs a large amount of labeled data, but expert annotations are expensive and limited. This project focuses on a realistic question:

**Can synthetic data help lesion segmentation when only a small amount of labeled training data is available?**

Instead of optimizing only for visual quality of generated images, the project evaluates synthetic data by its **downstream effect on segmentation performance**.

---

## Dataset

This project uses the **BUSI (Breast Ultrasound Images Dataset)**, a public breast ultrasound dataset containing three categories: **benign**, **malignant**, and **normal** images. The commonly cited BUSI release contains **780 images** collected from **600 female patients**, with **437 benign**, **210 malignant**, and **133 normal** images. citeturn900742search0turn900742search6turn900742search8

Dataset source:
- Kaggle BUSI dataset: citeturn900742search0
- Kaggle notebook reference used for dataset selection: citeturn164483view1

### How the dataset is used in this project

The notebooks do **not** use the full dataset as one training pool.

The code workflow is:
- load BUSI images and masks from category folders
- create a **stratified train/validation/test split**
- build **low-data subsets** from the training split only
- compare segmentation performance at:
  - **5% of the training split**
  - **10% of the training split**
  - **25% of the training split**

This is the main experimental setup and should be kept in mind when interpreting results.

---

## What the Code Actually Does

After reviewing the notebooks, the project has **two main experimental pipelines**.

### 1) Classical low-data segmentation pipeline
Implemented in:
- `src/create_synthetic_data_lesion.ipynb`
- `Lesion method.ipynb` (appears to be a consolidated or duplicated notebook version)

This notebook:
- loads grayscale ultrasound images and binary masks
- resizes images to **256 × 256**
- trains a **U-Net** segmentation model in PyTorch
- evaluates performance using:
  - **Binary cross-entropy with logits loss**
  - **Dice score**
  - **IoU**
- compares three data settings:
  - **Real only**
  - **Real + classical augmentation**
  - **Real + synthetic copy-paste augmentation**

### 2) GAN-based synthetic image pipeline
Implemented in:
- `src/create_synthetic_data_gan.ipynb`

This notebook includes:
- a **U-Net baseline** for segmentation under low-data settings
- a **GAN** trained on BUSI images to generate synthetic ultrasound images
- **pseudo-mask generation** by passing synthetic images through a trained segmentation model
- a combined dataset made from **real + synthetic image/mask pairs**

A key detail from the code is that the GAN image dataset is built only from **benign** and **malignant** image folders, not the normal class, while the copy-paste pipeline uses normal images as backgrounds and lesion images as sources.

---

## Experimental Design

### Data split
- **70%** training
- **15%** validation
- **15%** test

### Low-data regimes
The project intentionally simulates data scarcity by training only on small stratified subsets of the training split:
- **5% subset**
- **10% subset**
- **25% subset**

### Compared settings
For each low-data regime, the project compares:

#### A. Real-only baseline
Train a U-Net only on the selected real training subset.

#### B. Classical augmentation
Train a U-Net on the same subset with standard paired transformations applied to image and mask:
- horizontal flip
- vertical flip
- random rotation

#### C. Synthetic augmentation via lesion copy-paste
Synthetic lesion samples are created by:
- extracting lesion regions from lesion images using masks
- applying random geometric transformations
- pasting the lesion onto a normal background image
- creating a corresponding synthetic binary mask

These synthetic samples are then concatenated with the real low-data subset before training.

#### D. GAN-based augmentation
The GAN notebook explores a second synthetic-data strategy:
- train a GAN on BUSI ultrasound images
- generate synthetic ultrasound images
- create pseudo-masks using a trained U-Net
- combine synthetic image/mask pairs with real samples

This GAN workflow is more exploratory than the copy-paste pipeline and should be described as an experimental extension.

---

