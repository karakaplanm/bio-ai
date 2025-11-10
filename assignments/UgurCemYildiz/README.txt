Project Name - Malatya Customer Segmentation (Clustering)
Project Type - Unsupervised Learning / Clustering 
Industry - Retail / Marketing 
Name and Surname – Uğur Cem Yıldız

Project Summary
Project Description: This project focuses on developing an unsupervised machine learning model to segment customers in Malatya based on their behavioral and demographic data, using the custom dataset malatya_musteri_segmentasyonu.csv. The dataset includes customer demographics (age, gender, marital status, district) and financial behaviors (monthly income, spending score, car ownership, credit card count).
Objective: The primary goal of this project is to uncover natural groupings and hidden profiles within the customer base in a scenario where there are no pre-defined "correct answers" (labels). Using the K-Means clustering algorithm, customers will be grouped into distinct "clusters" (segments), and a profile will be created for each segment.

Key Project Details:
Dataset: malatya_musteri_segmentasyonu.csv (500 customer records).
Model: KMeans Clustering.
Analysis Method:
All numerical and categorical features in the dataset (e.g., yas, aylik_gelir_TL, harcama_skoru, cinsiyet, medeni_durum, semt) were used for the analysis.
A ColumnTransformer was used to allow the model to process different data types (numerical and categorical) simultaneously.
Numerical features were scaled using StandardScaler.
Categorical features were transformed using OneHotEncoder.
The "Elbow Method" was used to determine the optimal number of clusters (k).
The final model was trained with this optimal 'k' value, and a cluster label was assigned to each customer.
Evaluation: As there is no direct "accuracy" metric, the model's success was evaluated by visually and statistically analyzing the separation of the resulting clusters based on key business metrics like "Monthly Income" and "Spending Score."

Problem Statement
Approaching all customers with a single, uniform marketing strategy is inefficient. Different customer groups (e.g., "High Income, Low Spenders," "Low Income, High Spenders," "Young Savers") have different needs, motivations, and purchasing habits. However, these groups are not always predefined or labeled.
Objective: The objective of this project is to analyze the demographic and financial data of 500 customers in Malatya to segment them into meaningful groups in an "unsupervised" manner (i.e., without any pre-existing labels). The primary goal of the model is to discover these hidden structures and natural groupings in the data, providing a foundation for developing tailored marketing strategies, campaigns, and product recommendations for each segment.

Project Details:
Model: K-Means Clustering algorithm. This algorithm groups data points based on their distance to the center (centroid) of a predetermined number of 'k' clusters.

Key Analysis Questions:
How many natural groups do the customers fall into? (Finding optimal k with the Elbow Method).
Which features (Income, Age, or Spending Score?) are most influential in the segmentation?
How do these clusters appear on a "Monthly Income" vs. "Spending Score" axis?

Preprocessing: A Pipeline was used to ensure the model could fairly and accurately evaluate both numerical (income, age) and categorical (district, gender) data. This involved StandardScaler for numerical features and OneHotEncoder for categorical ones.
The significance of this project lies in its ability to provide a data-driven basis for optimizing marketing budgets, increasing customer loyalty, and making more efficient business decisions by offering tailored engagement to each customer group.
