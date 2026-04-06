# Low-Data Breast Lesion Segmentation with Synthetic Data Augmentation

This project studies **breast lesion segmentation in ultrasound images** under **low-data settings** using the **BUSI (Breast Ultrasound Images Dataset)**. The main goal is to evaluate whether **synthetic data augmentation** improves a **U-Net** segmentation model when only a small fraction of the training data is available.

The repository contains experiments for three training settings:

1. **Real-only baseline** – train U-Net on a small real subset  
2. **Classical augmentation** – train U-Net with standard geometric augmentation  
3. **Synthetic augmentation** – expand the low-data training set using synthetic lesion samples  

> This project is developed in the context of **Generative Neural Networks**, with a focus on **synthetic data augmentation** for medical image segmentation.  


---

## Project Motivation

Medical image segmentation usually requires a large amount of labeled data, but expert annotations are expensive and limited. This project investigates the question:

**Can synthetic data improve breast lesion segmentation when only a small amount of labeled training data is available?**

Rather than evaluating synthetic images only by visual appearance, the project measures their usefulness through **downstream segmentation performance**.

---

## Dataset

This project uses the **BUSI (Breast Ultrasound Images Dataset)**, a public breast ultrasound dataset with three categories:

- **benign**
- **malignant**
- **normal**

The commonly cited BUSI release contains:

- **780 images**
- **600 female patients**
- **437 benign**
- **210 malignant**
- **133 normal**

Dataset source:  
[Kaggle BUSI dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)

### How the dataset is used

The experiments do **not** train on the full dataset directly.

Instead, the workflow is:

- load BUSI images and masks from category folders
- create a **stratified train / validation / test split**
- construct **low-data subsets** from the training split only
- compare segmentation performance using:
  - **5% of the training split**
  - **10% of the training split**
  - **25% of the training split**

This low-data setup is the core experimental design of the project.

---

## Experimental Design

### Data split

- **70%** training
- **15%** validation
- **15%** test

### Compared settings

For each low-data regime, the project compares:

#### 1. Real-only baseline
Train a U-Net only on the selected real training subset.

#### 2. Classical augmentation
Train a U-Net on the same subset with paired image-mask transformations:

- horizontal flip
- vertical flip
- random rotation

#### 3. Synthetic augmentation via lesion copy-paste
Synthetic lesion samples are created by:

- extracting lesion regions from annotated lesion images
- applying random geometric transformations
- pasting lesions onto normal background ultrasound images
- generating corresponding synthetic binary masks

These synthetic samples are then concatenated with the real low-data subset before training.

#### 4. GAN-based augmentation (exploratory)
A second notebook explores a GAN-based strategy:

- train a GAN on BUSI ultrasound images
- generate synthetic ultrasound images
- create pseudo-masks using a trained U-Net
- combine synthetic image/mask pairs with real samples

This GAN workflow is included as an exploratory extension and is less stable than the copy-paste pipeline.

---

## Repository Structure

```text
GNN_FInal_Project/
├── src/
│   ├── create_synthetic_data_lesion.ipynb
│   └── create_synthetic_data_gan.ipynb
├── Lesion method.ipynb
└── README.md

