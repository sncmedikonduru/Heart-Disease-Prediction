# Heart Disease Prediction Using Machine Learning

## Problem Statement
Heart disease is a leading cause of mortality worldwide. Early detection of heart disease using machine learning can help in timely intervention and treatment. This project develops predictive models to classify patients as having heart disease or not based on their medical attributes, using the dataset `heart.csv`.

## Dataset
File: heart.csv  
Available on [Kaggle](https://www.kaggle.com/datasets) 

### Features
- **age**: Age of the patient
- **sex**: Sex (1 = Male, 0 = Female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = Yes, 0 = No)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-4)
- **thal**: Thalassemia type (0-3)

## Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms and pair plots.
- Checked for missing values using `.isnull()` method and handled them appropriately.
- Analyzed feature importance using correlation matrices and statistical tests.
- Identified and handled outliers in the dataset.

### Machine Learning Models
- **Logistic Regression**: Used for binary classification.
- **Naive Bayes**: Assumes independence among features and is efficient for small datasets.
- **Support Vector Machine (SVM)**: Effective for high-dimensional spaces.
- **K-Nearest Neighbors (KNN)**: Non-parametric method that makes predictions based on the closest training examples.
- **Decision Tree**: Simple tree-based method for classification.
- **Random Forest**: An ensemble method to improve accuracy by using multiple decision trees.
- **XGBoost**: A highly efficient gradient boosting algorithm that often outperforms other models.
- **Neural Network**: Captures complex relationships in data, suitable for deep learning approaches.

## Results
- **Best Performing Model**: Random Forest with **95% Accuracy**.


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
