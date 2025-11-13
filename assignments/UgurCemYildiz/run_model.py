import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# === PART 1: CREATE THE 1000-PERSON DATASET ===

# Define the options for each category
gender_options = ['Female', 'Male']
diet_options = ['Mediterranean', 'Western', 'Vegetarian']
smoking_options = ['Non-Smoker', 'Quit', 'Smoker']
activity_options = ['Low', 'Medium', 'High']
ethnicity_options = ['Caucasian', 'Asian', 'African', 'Hispanic']

# For consistent results every time we run it
np.random.seed(42) 
N = 1000 # Number of people

# Create the main DataFrame
df = pd.DataFrame({
    'Gender': np.random.choice(gender_options, N),
    'Diet_Type': np.random.choice(diet_options, N),
    'Smoking_Status': np.random.choice(smoking_options, N),
    'Physical_Activity': np.random.choice(activity_options, N),
    'Ethnicity': np.random.choice(ethnicity_options, N),
    'BMI': np.random.normal(loc=28, scale=4, size=N).round(1)
})

# 3. CALCULATE LIFESPAN (Based on factors)
# We start with a base lifespan and add/subtract years based on choices.
base_lifespan = 80 
lifespans = []

for i in range(N):
    lifespan = base_lifespan
    row = df.iloc[i]

    # Effects of the factors (Bonus/Penalty points)
    if row['Gender'] == 'Female': lifespan += 3
    
    # Diet Effect
    if row['Diet_Type'] == 'Mediterranean': lifespan += 7
    elif row['Diet_Type'] == 'Vegetarian': lifespan += 2
    elif row['Diet_Type'] == 'Western': lifespan -= 7

    # Smoking Effect (Strong negative impact)
    if row['Smoking_Status'] == 'Smoker': lifespan -= 12
    elif row['Smoking_Status'] == 'Quit': lifespan -= 3

    # BMI Effect (Ideal range 20-25)
    if row['BMI'] > 30: # Obese
        lifespan -= 8
    elif row['BMI'] < 18.5: # Underweight
        lifespan -= 4
    elif 20 < row['BMI'] < 25: # Ideal
        lifespan += 4

    # Activity Effect
    if row['Physical_Activity'] == 'High': lifespan += 6
    elif row['Physical_Activity'] == 'Low': lifespan -= 5
        
    # Genetic / Luck factor (a small random number)
    lifespan += np.random.normal(0, 4) 
    
    lifespans.append(int(lifespan))

# Add the calculated lifespan to the table
df['Lifespan'] = lifespans

# 4. Create the AI Target (This is what we want to predict)
# Target is 1 (Yes) if 100 or older, 0 (No) if not.
df['Long_Lived_100'] = np.where(df['Lifespan'] >= 100, 1, 0)


print(f"Dataset is ready. Number of 100+ year olds: {df['Long_Lived_100'].sum()}")
print("--------------------------------------------------\n")


# === PART 2: TRAINING THE AI MODEL ===

# 1. Define Inputs (X) and Target (y)
# We don't use 'Lifespan' as an input, because that's the answer (cheating!)
X = df[['Gender', 'Diet_Type', 'Smoking_Status', 'Physical_Activity', 'Ethnicity', 'BMI']]
y = df['Long_Lived_100']

# 2. Split data into Training (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define Data Preprocessing Steps
# Which columns are text (categorical) and which are numbers?
categorical_features = ['Gender', 'Diet_Type', 'Smoking_Status', 'Physical_Activity', 'Ethnicity']
numerical_features = ['BMI']

# Create a "transformer" to convert text data to numbers (One-Hot Encoding)
# 'passthrough' means 'leave the numerical columns (BMI) as they are'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Create the Model (Random Forest)
# A "Pipeline" automatically combines the preprocessing and the model.
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                     ])

# 5. Train the Model
print("AI Model is training...")
model.fit(X_train, y_train)
print("Model training complete.")

# Check the model's accuracy (Optional)
accuracy = model.score(X_test, y_test)
print(f"Model Test Accuracy: {accuracy:.2%}")
print("--------------------------------------------------\n")


# === PART 3: ANALYZE RESULTS (FEATURE IMPORTANCE) ===

# Get the trained "classifier" (RandomForest) part from the model
rf_model = model.named_steps['classifier']
# Get the "preprocessor" (the converter) part from the model
preprocessor = model.named_steps['preprocessor']

# Get the names of all the features after conversion
# e.g., 'cat__Gender_Male', 'cat__Diet_Type_Mediterranean', 'num__BMI'
feature_names = preprocessor.get_feature_names_out()

# Get the importance scores from the model
importances = rf_model.feature_importances_

# Put the names and scores into a nice table (DataFrame)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'ImportanceScore': importances
})

# Clean up the names to make them easier to read
importance_df['Feature'] = importance_df['Feature'].str.replace('num__', '')
importance_df['Feature'] = importance_df['Feature'].str.replace('cat__', '')

# Sort the table from most important to least important
importance_df = importance_df.sort_values(by='ImportanceScore', ascending=False)

print("--- MOST IMPORTANT FACTORS FOR 100+ LIFE (Model Report) ---")
print(importance_df.head(15)) # Show the top 15
print("\n")


# === PART 4: VISUALIZATION (THE GRAPH) ===

plt.figure(figsize=(10, 8)) # Set the graph size
sns.barplot(x='ImportanceScore', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('AI Model: Feature Importance for Living 100+ Years', fontsize=16)
plt.xlabel('Importance Score (How much the model used this factor)', fontsize=12)
plt.ylabel('Factors (Features)', fontsize=12)
plt.tight_layout() # Makes sure the labels fit
plt.show()
