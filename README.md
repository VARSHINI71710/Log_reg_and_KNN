Overview

This repository contains a series of practice projects in machine learning using Python and popular libraries such as scikit-learn, pandas, NumPy, Matplotlib, and Seaborn.
The projects cover classification, regression, and evaluation of models, helping learners understand the complete ML workflow from data preprocessing to model evaluation.
_____________
General Workflow

For each project, the following steps are following
Data Loading

Read the dataset (CSV/Excel) using pandas.

  Inspect data for missing values and basic statistics.

Feature Selection

  Choose relevant features (X) and target (y).

  Separate numerical and categorical features if needed.

Data Splitting

  Split dataset into training and test sets using train_test_split.

  Typical split: 80% training, 20% testing.

Feature Scaling

Standardize numerical features using StandardScaler or similar.

Important for distance-based models (k-NN) and gradient-based models (Logistic Regression).

Model Training

  Choose a machine learning algorithm:

  classification: Logistic Regression, k-NN, Decision Trees, etc.

  Regression: Linear Regression, k-NN Regression, Ridge/Lasso, etc.

  Fit the model on training data.

Cross-Validation (Optional)

  Use k-fold cross-validation to tune hyperparameters like k in k-NN.

  Evaluate mean performance (accuracy/RMSE) on training data.

Prediction

  Predict on test data.

  For classification: predicted labels and probabilities.

  For regression: predicted numerical values.

  Model Evaluation

Classification metrics:

  Accuracy, Precision, Recall, F1-score, ROC-AUC.

  Confusion Matrix and ROC Curve.

  Regression metrics:

  RMSE, R² score, Mean Absolute Error (MAE).

Feature Importance

  Analyze which features impact predictions the most.

  For Logistic Regression: check coefficients.

  For tree-based models: check feature importances.

Visualization

  Plot confusion matrices, ROC curves, and feature importance for better understanding.
_______________
Example Practice Projects

Flowers Classification — k-NN

  Dataset: flowers.csv

  Predict flower species using sepal/petal measurements.

  Evaluate accuracy and confusion matrix.

  Find the best k using cross-validation.

Airbnb Price Prediction — k-NN Regression

  Dataset: airbnb.csv

  Predict rental price using size, distance, rating, reviews.

  Use 5-fold cross-validation to find best k.

  Evaluate test RMSE and R².

Loan Default Prediction — Logistic Regression

  Dataset: loan_default.csv

  Predict whether a customer will default.

  Scale numeric features and fit logistic regression.

  Report accuracy, precision, recall, F1-score, ROC-AUC.

  Identify features that most increase default probability.
___________________
Tools & Libraries

Python 3.12.5

pandas

NumPy

scikit-learn

Matplotlib

Seaborn
__________________
Notes

Always scale numeric features when using distance-based or gradient-based models.

Use cross-validation for hyperparameter tuning to prevent overfitting.

Visualizations help interpret model performance and feature impact.

Some output graphs:

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/f7ed5fb3-9ce8-4380-a62c-b5af37178a41" />

<img width="496" height="470" alt="image" src="https://github.com/user-attachments/assets/5bd528fa-ebaf-4b0d-8913-c417d34bfae6" />

<img width="496" height="470" alt="image" src="https://github.com/user-attachments/assets/78eb215c-923c-44ec-88b8-fc43fd9cf2e0" />


