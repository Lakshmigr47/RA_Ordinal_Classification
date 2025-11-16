AI-Based Ordinal Classification of Rheumatoid Arthritis Severity
This project develops a machine learning model that classifies the severity stages (0–4) of rheumatoid arthritis (RA) from radiographic X-ray images using ordinal regression, transfer learning, and EfficientNet-B0.
The goal is to build a clinically meaningful model that understands the ordered progression of joint degeneration and penalizes larger staging errors more than smaller ones.

Aim
To design and implement a deep learning system capable of predicting rheumatoid arthritis severity from medical X-rays using ordinal logic (CORAL) and evaluate its performance using clinically relevant metrics such as Accuracy, Mean Absolute Error (MAE), and Quadratic Weighted Kappa (QWK).

Project Description
Uses a publicly available X-ray dataset (Kaggle RA/OA dataset)
Applies transfer learning with EfficientNet-B0 for feature extraction
Implements ordinal regression to model disease progression
Evaluates performance using Accuracy, QWK, MAE, F1-score
Aims to assist radiologists by providing consistent staging predictions

Tech Stack
Python
PyTorch
Torchvision
Coral-Pytorch (ordinal regression)
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn

Dataset (Not included)
This repository does not contain the dataset due to size and licensing restrictions.
Please download it manually from Kaggle and place it in: data/RA/

Folder Structure (Initial)
RA_Ordinal_Project/
│── data/
│── src/
│── notebooks/
│── saved_models/
│── results/
│── README.md
│── requirements.txt
