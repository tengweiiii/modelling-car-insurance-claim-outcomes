# 2.0 Data Prepocessing
Before training models, we need to handle the missing values.

## 2.1 Create a working copy
We should preserve the original dataset in case we need to revisit it.
```
# Create a copy of the original dataset
car_insurance_clean = car_insurance.copy()
```

## 2.2 Handle missing values
We have two numerical features with missing values:
* `credit_score`
* `annual_mileage`

Let's use median imputation.

**Why median instead of mean?**

The median is less sensitive to extreme values(outliers).

For example, annual mileage has a minimum of 2,000 and a maximum of 22,000, so using median is a reasonable starting point.

```
# Fill missing values using the median
car_insurance_clean['credit_score'] = car_insurance_clean['credit_score'].fillna(
  car_insurance_clean['credit_score'].median()
)

car_insurance_clean['annual_mileage'] = car_insurance_clean['annual_mileage'].fillna(
  car_insurance_clean['annual_mileage'].median()
)
```

## 2.3 Verify the missing values
```
# Check missing values again
car_insurance_clean.isnull().sum()
```
We expect all columns to have zero missing values.

**Why are we doing this?**

Statsmodel cannot fit a Logistic Regression model using observations with missing predictor values.

Imputation allows us to retin all 10,000 customer records.

Output:
| Column               | Missing Values |
|----------------------|---------------:|
| index                | 0              |
| id                   | 0              |
| age                  | 0              |
| gender               | 0              |
| driving_experience   | 0              |
| education            | 0              |
| income               | 0              |
| credit_score         | 0              |
| vehicle_ownership    | 0              |
| vehicle_year         | 0              |
| married              | 0              |
| children             | 0              |
| postal_code          | 0              |
| annual_mileage       | 0              |
| vehicle_type         | 0              |
| speeding_violations  | 0              |
| duis                 | 0              |
| past_accidents       | 0              |
| outcome              | 0              |

## 2.4 Identify the candidate features
Now we need to identify all the features that we will test.

Remember:
* Exclude `id` because it is only an identifier
* Exclude `outcome` because it is our target variable
* Include every other column, including `gender`

Based on the dataset, we expect 16 candidate features.

```
# Identify all predictor features
features = car_insurance_clean.drop(
  columns=['id', 'outcome']
).columns.tolist()

print(features)
print("Number of features: ", len(features))
```
Output:
```
['age', 'gender', 'driving_experience', 'education', 'income', 'credit_score', 'vehicle_ownership', 'vehicle_year', 'married', 'children', 'postal_code', 'annual_mileage', 'vehicle_type', 'speeding_violations', 'duis', 'past_accidents']
Number of features:  16
```

## Interpretation of Data Preprocessing
The data preprocessing stage focused on preparing the dataset for Logistic Regression modelling.

The original dataset contained 10,000 customer records, with missing values found in two numerical features: `credit_score` (982 missing values) and `annual_mileage` (957 missing values). These missing values were replaced using the median of their respective columns to retain all customer records while reducing the influence of extreme values.

After handling the missing values, all columns were checked again to ensure there were no remaining missing values.

Next, 16 candidate features were identified for modelling. The `id` column was excluded because it is only a unique customer identifier, while `outcome` was excluded from the predictor features because it is the target variable. All other features, including `gender`, were retained for evaluation.

As a result, the dataset is now prepared with 10,000 customer records, no missing values, and 16 candidate features ready for individual Logistic Regression models.

**Key takeaway**: Data preprocessing ensures that the dataset is complete and properly prepared for modelling, allowing us to compare the predictive performance of each feature fairly.
