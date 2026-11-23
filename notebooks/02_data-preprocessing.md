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
