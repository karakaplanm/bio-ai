# 📊 Reports & Diagnostic Figures

This directory stores evaluation plots, confusion matrices, and Grad-CAM explainability heatmaps generated during model testing:

- `confusion_matrix.png`: Breakdown of True Positives, False Positives, True Negatives, False Negatives.
- `roc_curve.png`: Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC).
- `gradcam_vhl_explanation.png`: Visual heatmap highlighting clear cell cytoplasm and vascular morphology.

> **Note:** These figures are automatically generated when running `python main.py --mode eval` or completing the Google Colab notebook (`notebooks/VHL_Histopathology_Colab.ipynb`).

