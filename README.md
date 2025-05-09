# ❄️ Climate Modeling Using SAR Imagery

This repository contains the source code and jupyter notebook of the project on snow cover segmentation and climate analysis using Synthetic Aperture Radar (SAR) satellite imagery. The aim is to study snow distribution patterns and their correlation with climatic variables across seasons, leveraging deep learning and visual analysis techniques.

---

## 📂 Repository Structure

```
.
├── Snow_Masks.ipynb    # Main notebook with full analysis pipeline
├── unet.py                            # U-Net model architecture for snow segmentation
├── unet_train.py                      # Training script for the U-Net model
├── unet_inference.py                  # Inference script to predict snow masks
├── SAM.py                             # Script to run Segment Anything Model (SAM) on SAR images
├── plots/                             # PNG visualizations: activation maps & climate trends
├── README.md                          # Project documentation
```

---

## 🛰️ Project Summary

### ❄️ Objective:

To perform snow cover segmentation on SAR satellite images and analyze its correlation with climate variables like temperature, precipitation, humidity, and snowfall.

### 📌 Key Components:

* **Data**: Triplets of SAR image (from Sentinel1), RGB image, and the snow cover mask derived using Normalized Difference Fractional Snow Index (NDFSI). The dataset before curation was of size 4K+, but after curation reduced to 1085 images. Beside, Openmeteo Archive API is used to collect the climate parameters such as temperature, humidity, precipitation, snowfall, etc.
* **Segmentation Models**:

  * **U-Net**: Custom architecture trained from scratch on SAR snow masks.
  * **SAM (Segment Anything Model)**: For advanced segmentation benchmarking.
 
* **Feature Visualization**:

  * Activation maps generated using **CLIP**, **DINOv2**, and **DeepLabV3**.
  * t-SNE plots comparing RGB and SAR image embeddings using a custom CLIP-style model.

### 📈 Climate Analysis:

* SAR imagery is paired with meteorological variables from Open-Meteo data.
* Seasonal trend plots are generated (Winter, Spring, Fall).
* Linear regression fits on snow cover vs. climatic variables.
* Time series of snow pixel counts over months/years.

---

## 🧠 Learnings

1. **SAR Image Preprocessing**: Efficient binarization and adaptive edge detection methods were essential for feature extraction and segmentation quality.
2. **CLIP-style Embeddings**: A contrastive model effectively learned a separable latent space for RGB vs SAR images, as visualized with t-SNE.
3. **Temporal Snow Trends**: Correlating snow cover area with daily meteorological data gave insight into snow behavior across seasons.
4. **Model Interpretability**: Visualizing activation maps from models like DeepLabV3 and DINOv2 provided insight into what features are emphasized during inference.
5. **Pipeline Integration**: Full integration of preprocessing, segmentation, climate data pairing, and visualization into one reproducible pipeline.

---

## 📓 Getting Started

### 🔧 Requirements:

* Python ≥ 3.8
* PyTorch
* torchvision
* OpenCV
* pandas, matplotlib, seaborn
* scikit-learn
* PIL

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🚀 Running the Notebook:

The main notebook contains:

* Motivation, Objective, Related Work of multimodal learning, my learning, relation with the course DA623, some interesting insights and outcomes, and a few preliminary discussion on satellite imaging.
* Data pairing logic (SAR + weather CSVs)
* Snow segmentation performance
* Activation map visualization
* Time series and regression plots
* Seasonal snow behavior analysis

---

## 📸 Visualizations

* PNGs of activation maps from CLIP, DINOv2, and DeepLabV3 are saved in the `plots/` directory.
* t-SNE projection of embeddings distinguishes SAR and RGB modalities.
* Time series plots of snow cover across various months and regions.

<p align="center">
  <img src="plots/clip_activations_comparison.png" alt="CLIP Activation Map" width="500"/>
</p>

---


## 📌 TODO

* [ ] Compile the results in a manuscript and submit to IEEE Transactions on Geosciences and Remote Sensing.
* [ ] Propose a novel segmentation model to improve the accuracy of cross-domain segmentation

---

## 🧾 License

This project is released under the MIT License.

---

