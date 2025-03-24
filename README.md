# Heart Disease Prediction Using Machine Learning

## Problem Statement
Heart disease is one of the leading causes of mortality worldwide, and early detection is critical to improving patient outcomes. This project aims to develop predictive models that classify individuals as having heart disease or not, based on their medical attributes. Machine learning techniques are employed to train models using historical patient data to identify patterns that may indicate the presence of heart disease.

The dataset used in this project is `heart.csv`, which contains a set of medical features associated with heart disease.

## Dataset
The dataset `heart.csv` is sourced from [Kaggle](https://www.kaggle.com/datasets) and includes the following features:

### Features
- **age**: Age of the patient
- **sex**: Sex of the patient (1 = Male, 0 = Female)
- **cp**: Chest pain type (0-3) - categorizes the type of chest pain.
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved during exercise
- **exang**: Exercise-induced angina (1 = Yes, 0 = No)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-4)
- **thal**: Thalassemia type (0-3)

These features were selected as they are known to be predictive of heart disease based on medical literature.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - **NumPy**: For numerical computations and array handling.
  - **Pandas**: For data manipulation and preprocessing.
  - **Matplotlib & Seaborn**: For data visualization.
  - **Scikit-learn**: For machine learning algorithms and model evaluation.

## Exploratory Data Analysis (EDA)
- **Data Cleaning**: Checked for missing values using `.isnull()` and handled them through imputation or removal.
- **Outlier Detection**: Used statistical methods like Z-scores and boxplots to identify and handle outliers that could skew model performance.
- **Feature Distribution Visualization**: Employed histograms, boxplots, and pair plots to visualize distributions and relationships between features.
- **Feature Correlation**: Applied a correlation matrix to evaluate multicollinearity and selected the most relevant features for model training.
- **Class Distribution**: Analyzed the target variable (presence of heart disease) to ensure a balanced dataset, or applied techniques like SMOTE for class balancing if necessary.

### Machine Learning Models
The following models were implemented and evaluated:

- **Logistic Regression**: A fundamental binary classifier that is used to model the relationship between a dependent binary variable and independent features.
- **Naive Bayes**: A probabilistic classifier based on Bayes’ Theorem, effective for categorical features, assuming independence among features.
- **Support Vector Machine (SVM)**: A powerful model that finds the optimal hyperplane to separate classes in high-dimensional space.
- **K-Nearest Neighbors (KNN)**: A simple, non-parametric algorithm that predicts the class based on the majority label of its nearest neighbors.
- **Decision Tree**: A tree-based method that makes decisions by splitting the data at each node based on feature values.
- **Random Forest**: An ensemble method that creates multiple decision trees and combines their predictions, improving accuracy and reducing overfitting.
- **XGBoost**: A gradient boosting model known for its performance and efficiency in handling large datasets, with hyperparameter tuning.
- **Neural Network**: A deep learning model capable of capturing non-linear relationships and interactions between features.

## Results
- The **Random Forest** model performed the best, achieving **95% Accuracy**, demonstrating its ability to capture complex relationships between features.

### Final Model
- **Best Performing Model**: Random Forest with 95% accuracy, followed by Logistic Regression and Naive Bayes.


## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart_disease_prediction.ipynb
   ```
4. Run the cells to perform EDA and train the models.

## License
This project is open-source and free to use.
