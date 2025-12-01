Plant Nutrient Deficiency Prediction Project
------------------------------------------------

This project includes a machine learning workflow for predicting nutrient deficiency in plant leaves
based on chemical measurements such as chlorophyll, moisture, pH, and nitrate levels.

📊 1. Plant Leaf Nutrient Deficiency Analysis

This section includes:
- Simulated chemical measurement dataset
- Features: chlorophyll, moisture, pH, nitrate
- Labels: 0 = healthy leaf, 1 = nitrogen deficiency
- Train/Test split (75% - 25%)
- Classification using Random Forest
- Accuracy evaluation
- 2D scatter plot visualization of predictions

📁 Files included in this project:
1. plant_deficiency_classifier.py → ML training, testing & visualization
2. sample_leaf_data.csv → Example dataset
3. README.txt → This explanation file

🚀 Running the Project

Install required packages:
pip install scikit-learn matplotlib numpy pandas

Run the script:
python plant_deficiency_classifier.py
