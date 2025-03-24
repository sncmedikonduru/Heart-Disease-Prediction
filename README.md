# Heart Disease Prediction

## Problem Statement
Heart disease is a leading cause of mortality worldwide. Early detection of heart disease using machine learning can help in timely intervention and treatment. This project aims to develop predictive models that classify patients as having heart disease or not based on their medical attributes.

## Dataset
- **File:** `heart.csv`
### Features
- `age`: Age of the patient  
- `sex`: Sex (1 = Male, 0 = Female)  
- `cp`: Chest pain type (0-3)  
- `trestbps`: Resting blood pressure (mm Hg)  
- `chol`: Serum cholesterol (mg/dl)  
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)  
- `restecg`: Resting ECG results (0-2)  
- `thalach`: Maximum heart rate achieved  
- `exang`: Exercise-induced angina (1 = Yes, 0 = No)  
- `oldpeak`: ST depression induced by exercise  
- `slope`: Slope of the peak exercise ST segment  
- `ca`: Number of major vessels (0-4)  
- `thal`: Thalassemia type (0-3)  

## Technologies Used
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Machine Learning Models:**
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - XGBoost
  - Neural Network

## Exploratory Data Analysis (EDA)
- Visualized feature distributions and correlations.
- Checked for missing values and outliers.
- Analyzed feature importance and relationships.

## Model Implementation
Each machine learning model was trained and evaluated using performance metrics such as accuracy, precision, recall, and confusion matrix.

## Results
- **Best Performing Model:** Random Forest with 95% of Accuracy

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart_disease_prediction.ipynb
   ```
4. Run the cells to perform EDA and train the models.

## License
This project is open-source and free to use.
