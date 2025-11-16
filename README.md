# AI-Based Ordinal Classification of Rheumatoid Arthritis Severity

## **Abstract**
Rheumatoid arthritis (RA) is a chronic inflammatory autoimmune disorder that primarily affects the small peripheral joints of the hands and wrists. If left untreated, it progresses from mild joint inflammation to severe cartilage destruction, bone erosion, deformities, and disability. Hand radiography (X-ray imaging) is one of the most widely used diagnostic methods to assess joint damage and monitor disease progression. Convolutional Neural Networks (CNNs) have demonstrated strong performance in medical image classification tasks due to their ability to learn complex visual patterns.

This project develops a machine learning model that predicts RA severity stages (0–4) from hand X-ray images using **ordinal regression** instead of standard multi-class classification. The model uses **transfer learning** with EfficientNet-B0 for feature extraction and applies **CORAL** (COnsistent RAnk Logits) to enforce the natural ordering of disease stages. This ensures that misclassifications between adjacent stages are penalized less than large-stage errors, producing clinically meaningful predictions.

The model achieved the following evaluation results:

- **Accuracy:** 0.6751  
- **Quadratic Weighted Kappa (QWK):** 0.8306  
- **Mean Absolute Error (MAE):** 0.3732  
- **F1-score (macro):** 0.6848  

These results demonstrate strong ordinal agreement between predicted and true labels, indicating that ordinal deep learning can serve as a useful tool to assist clinicians in staging rheumatoid arthritis progression.

---

## **Repository Structure**
```
RA_Ordinal_Project/
├── demo.py                     # Single-image prediction script
├── src/
│   ├── dataset.py              # PyTorch Dataset class
│   ├── model.py                # EfficientNet-B0 + CORAL ordinal head
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script — QWK, MAE, F1, confusion matrix
│   └── utils.py                # (optional) helper functions
├── data/RA/                    # Dataset (NOT included in repo)
│   ├── train/
│   ├── val/
│   └── test/
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── predictions.csv
├── saved_models/               # Model checkpoints (ignored via .gitignore)
├── notebooks/                  # Optional EDA / prototyping notebooks
├── requirements.txt
└── README.md
```

---

## **How to Run the Project**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### **2. Train the Model**
```bash
python3 src/train.py
```

This trains EfficientNet-B0 + CORAL for 10 epochs and saves the model to:
```
saved_models/efficientnet_ordinal.pth
```

---

### **3. Evaluate the Model**
```bash
python3 src/evaluate.py
```

This generates:
- `results/confusion_matrix.png`
- `results/predictions.csv`
- Prints Accuracy, QWK, MAE, and F1-score

---

### **4. Run Single-Image Demo**
```bash
python3 demo.py --image data/RA/test/<class>/<filename>.png
```

Example:
```bash
python3 demo.py --image data/RA/test/2/9003316R.png
```

This displays:
- The input X-ray  
- Predicted RA stage (0–4)  
- Threshold probabilities  
- Visual plot of the image  

---

## **How Ordinal Regression Works**
Instead of predicting a single class, the model predicts **4 sigmoid thresholds**:

```
p(stage > 0)
p(stage > 1)
p(stage > 2)
p(stage > 3)
```

Final stage = number of threshold probabilities > 0.5  
(Using CORAL transformation)

This makes predictions more **clinically meaningful** than softmax classification.

---

## **Dataset (Not Included)**
You must download the RA/Knee dataset manually from Kaggle and place it under:

```
data/RA/train/
data/RA/val/
data/RA/test/
```

Each folder must contain subfolders `0/ 1/ 2/ 3/ 4/` with images.

---

## **Future Improvements**
- Train longer on GPU for higher accuracy  
- Add Grad-CAM heatmaps to show attention regions  
- Compare CORAL vs Softmax classification  
- Build a Streamlit web UI  
- Add uncertainty calibration  
