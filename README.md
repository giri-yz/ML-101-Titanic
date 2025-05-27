
# Titanic Survival Prediction - Step-by-Step ML 101 Project

## Project Overview
This project predicts whether passengers survived the Titanic disaster using machine learning.  
We used the Titanic dataset with passenger info and survival labels to build a classification model.

---

## Step-by-Step Workflow and Explanation

### 1. Data Exploration  
**What we did:**  
- Loaded dataset and checked basic info with `df.info()` and `df.head()`  
- Observed data types, missing values, and column meanings  

```python
import pandas as pd

df = pd.read_csv("train.csv")
print(df.info())
print(df.head())
```

**Output:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
```

**Why we use this:**  
- To understand the dataset structure and data types  
- To identify missing values and plan data cleaning  
- To decide which features to keep or modify

---

### 2. Data Cleaning  
**What we did:**  
- Filled missing `Age` values with median  
- Dropped columns `Ticket`, `Name`, and `Cabin`  
- Filled missing `Embarked` values with the most common port

```python
df["Age"].fillna(df["Age"].median(), inplace=True)
df.drop(columns=["Ticket", "Name", "Cabin"], inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
```

**Why we use this:**  
- Missing data can disrupt model training, so filling with median prevents errors and keeps data distribution realistic  
- Dropping `Ticket`, `Name`, and `Cabin` removes irrelevant or too sparse features to reduce noise  
- Filling missing `Embarked` values ensures no missing categorical data which could cause errors

---

### 3. Encoding Categorical Variables  
**What we did:**  
- Converted `Sex` and `Embarked` into numeric columns using label encoding and one-hot encoding

```python
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
```

**Output:**

```python
print(df[["Sex", "Embarked_Q", "Embarked_S"]].head())
```

```
   Sex  Embarked_Q  Embarked_S
0    0           0           1
1    1           0           0
2    1           0           1
3    1           0           1
4    0           0           1
```

**Why we use this:**  
- ML models need numeric input, so we encode categorical data to numbers  
- One-hot encoding prevents the model from assuming any order in categories  
- Label encoding is used when only two categories exist (`Sex`)

---

### 4. Feature Selection  
**What we did:**  
- Selected relevant features and target variable

```python
X = df.drop("Survived", axis=1)
y = df["Survived"]
```

**Why we use this:**  
- Separating features (`X`) from target (`y`) is necessary for supervised learning  
- Dropping the target column from features avoids data leakage

---

### 5. Train-Test Split  
**What we did:**  
- Split data into training and testing sets (80%/20%)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why we use this:**  
- To evaluate the model’s ability to generalize on unseen data  
- Training on 80% and testing on 20% is a common practice for balanced evaluation  

---

### 6. Model Training - Logistic Regression  
**What we did:**  
- Trained Logistic Regression model with increased max iterations

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**Why we use this:**  
- Logistic Regression is a simple, interpretable baseline model for binary classification  
- Increasing `max_iter` helps the solver converge if it doesn’t within default iterations  

---

### 7. Model Prediction  
**What we did:**  
- Predicted on test data

```python
y_pred = model.predict(X_test)
```

**Why we use this:**  
- To generate predicted labels on the test set for evaluation  

---

### 8. Model Evaluation  
**What we did:**  
- Evaluated accuracy, precision, recall, and F1-score

```python
from sklearn.metrics import classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
```

**Output Example:**

```
Accuracy: 0.78
              precision    recall  f1-score   support

           0       0.85      0.78      0.81       109
           1       0.69      0.78      0.73        69

    accuracy                           0.78       178
   macro avg       0.77      0.78      0.77       178
weighted avg       0.79      0.78      0.78       178
```

**Why we use this:**  
- Accuracy measures overall correctness of the model  
- Precision and recall balance false positives and false negatives, important in imbalanced data  
- F1-score combines precision and recall into a single metric for performance comparison

---

## Results Summary  
- Achieved about **78% accuracy** on test data  
- Balanced precision and recall for both survived and non-survived classes  
- Logistic Regression serves as a strong baseline model

---

## Next Steps  
- Try more advanced models like Random Forest or Gradient Boosting  
- Add feature engineering such as family size or title extraction  
- Use hyperparameter tuning (e.g., GridSearchCV) for optimization  
- Explore techniques to handle class imbalance if needed

---

## How to Run This Project  
1. Install required libraries:  
   `pip install pandas numpy scikit-learn`  
2. Download the Titanic dataset and place `train.csv` in your working directory  
3. Run the provided Jupyter notebook or Python script step-by-step

---
