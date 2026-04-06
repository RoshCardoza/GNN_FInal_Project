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

This project uses the **BUSI (Breast Ultrasound Images Dataset)**, a public breast ultrasound dataset containing three categories: **benign**, **malignant**, and **normal** images. The commonly cited BUSI release contains **780 images** collected from **600 female patients**, with **437 benign**, **210 malignant**, and **133 normal** images. 

Dataset source:
- Kaggle BUSI dataset:https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

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

## Repository Structure

Based on the current GitHub repository structure, the main files are inside `src/`, which contains the two notebooks used for the project. 

```text
GNN_FInal_Project/
├── src/
│   ├── create_synthetic_data_gan.ipynb
│   └── create_synthetic_data_lesion.ipynb
├── Lesion method.ipynb
└── README.md
```

### File descriptions
- **`src/create_synthetic_data_lesion.ipynb`**  
  Main notebook for low-data segmentation experiments with real-only, classical augmentation, and synthetic copy-paste augmentation.

- **`src/create_synthetic_data_gan.ipynb`**  
  Experimental notebook for GAN-based synthetic image generation and pseudo-mask creation.

- **`Lesion method.ipynb`**  
  Appears to be another version of the lesion segmentation workflow.

---

## Model Details

### Segmentation model
The main segmentation network is a **U-Net** implemented in PyTorch.

The model is trained with:
- **input size:** 256 × 256
- **batch size:** 8
- **optimizer:** Adam
- **loss:** `BCEWithLogitsLoss`

The lesion notebook uses **30 epochs** by default for segmentation experiments, while the GAN notebook uses **20 epochs** for its U-Net experiments and **50 epochs** for GAN training.

### GAN model
The GAN notebook defines:
- a **Generator** network
- a **Discriminator** network
- adversarial training with **BCELoss**

The GAN is used to generate grayscale ultrasound-like images, which are later pseudo-labeled for segmentation training.

---

## Evaluation Metrics

The project evaluates segmentation quality using:
- **Dice Score**
- **Intersection over Union (IoU)**
- **Test loss**

The proposal also frames the project around understanding **when synthetic augmentation helps or harms**, rather than assuming that realistic-looking generated images automatically improve segmentation.

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/RoshCardoza/GNN_FInal_Project.git
cd GNN_FInal_Project
```

### 2. Download the BUSI dataset
Download the dataset from Kaggle and place it in a local folder such as:

```text
dataset/
├── benign/
├── malignant/
└── normal/
```

### 3. Update dataset paths in the notebooks
The notebooks currently use **hardcoded local Windows paths**, for example:

```python
DATA_DIR = r"C:\Users\ROSHAL CARDOZA\Desktop\WS25-26\Final_Project\GNN_FInal_Project\dataset"
```

Before running the notebooks, replace this with the path on your own machine.

### 4. Install dependencies
Typical dependencies used in the notebooks:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

### 5. Run the notebooks
Suggested order:

#### Option A — main segmentation study
Run:
- `src/create_synthetic_data_lesion.ipynb`

This is the best notebook to use for the main project story because it contains the clearest end-to-end low-data segmentation workflow.

#### Option B — GAN extension
Run:
- `src/create_synthetic_data_gan.ipynb`

Use this notebook if you want to reproduce the GAN-based synthetic data experiment.

---

## Key Findings the README Should Communicate

This project is **not** simply “breast ultrasound segmentation on BUSI.”  
It is more specifically:

> **An evaluation of synthetic data augmentation for breast lesion segmentation in low-data regimes using BUSI and a U-Net baseline.**

The most important design choice is that the experiments are run under **artificial label scarcity** by training on only **5%, 10%, and 25% of the training split**, not the full dataset.

That makes the project stronger academically, because the goal is not just segmentation performance, but understanding whether synthetic data helps when real labeled data is scarce.

---

## Limitations

A few limitations should be stated clearly:

1. **Not full-dataset training**  
   The experiments are designed around low-data subsets, so results should be interpreted as low-data findings rather than full-dataset BUSI performance.

2. **Notebook-based codebase**  
   The project is currently organized as notebooks rather than a modular Python package.

3. **Hardcoded local paths**  
   The code requires manual path updates before running on another machine.

4. **GAN pipeline is exploratory**  
   The GAN-based augmentation uses pseudo-masks generated by a trained segmentation model, so synthetic labels may contain propagated segmentation errors.

5. **Result reporting should be verified from final runs**  
   Some notebook summary values appear to be manually entered for comparison plotting, so final reported numbers should come from saved experiment outputs.

---

## Future Improvements

Possible next steps for the project:
- convert notebooks into reusable Python scripts
- save metrics automatically to CSV or JSON
- add full reproducibility with a `requirements.txt` file
- compare copy-paste augmentation and GAN augmentation more systematically
- test on additional segmentation architectures such as Attention U-Net or UNet++
- evaluate synthetic image quality separately from downstream segmentation utility

---

## Acknowledgments

- **BUSI dataset** for breast ultrasound images and lesion masks 
- Your project proposal for framing the study around **generative augmentation in low-data medical segmentation** 
- GitHub repository used for the implementation structure 

