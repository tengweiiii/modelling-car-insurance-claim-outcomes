# 3.0 Build individual Logistic Regression models
Our objective is to determine which single feature produces the highest prediction accuracy.

## 3.1 Understand the modelling approach
We will build 16 separate Logistic Regression models.

For example:

**Model 1:**
$$
outcome = f(age)
$$

**Model 2:**
$$
outcome = f(gender)
$$

**Model 3:**
$$
outcome = f(driving\_experience)
$$

Each model uses:
* One independent variable (predictor)
* One dependent variable (target: `outcome`)

We will use the `logit()` function from Statsmodels.

## 3.2 Understand the Logistic Regression formula
The general formula is:
```
logit('outcome ~ age', data=car_insurance_clean)
```
Let's break it down:
| Component | Meaning |
|---|---|
| `outcome` | Dependent variable (what we predict) |
| `~` | Explained by |
| `age` | Independent variable (predictor) |
| `data` | Dataset used for modelling |

In simple terms, we are asking:

Can a customer's age predict whether they will make an insurance claim?

For categorical variables, we can use `C()` to explicitly tell Statsmodels to tret the feature as categorical

For example:
```
logit('outcome ~ C(vehicle_type)', data=car_insurance_clean)
```

This is useful because `vehicle_type` contains text categories such as `sedan` and `sports car`.

## 3.3 Build a Logistic Regression model for each feature
Instead of manually creating 16 models, we can use a `for` loop.

This makes our code shorter, easier to maintain, and more efficient.

Let's first understand the process:
1. Loop through every feature in features.
2. Construct a Logistic Regression formula.
3. Fit the model using the dataset.
4. Generate predicted probabilities.
5. Convert probabilities into binary predictions.
6. Calculate the accuracy score.
7. Store the results for comparison.

### 3.3.1 Create an empty dictonary
We need somewhere to store the accuracy of each model.
```
# Create an empty dictonary to store model accuracy
accuracies = {}
```

### 3.3.2 Train individual models
```
# Loop through each candidate feature
for feature in features:

  # Create the Logistic Regression formula
  formula = f'outcome ~ {feature}'

  # Fit the Logistic Regression model
  model = logit(formula, data=car_insurance_clean).fit()

  # Generate predicted probabilities
  predictions = model.predict(car_insurance_clean)

  # Convert probabilities into binary predictions
  predicted_classes = (prediction >= 0.5).astype(int)

  # Calculate accuracy
  accuracy = (predicted_classes == car_insurance_clean['outcome']).mean()

  # Store the accuracy
  accuracies[feature] = accuracy
```

**Let's understand some important code**
**1. Constructing the formula**
   ```
   formula = f'outcome ~ {feature}'
   ```
   The f-string allows us to dynamically insert each feature into the formula.

    For example:
    * First iteration: `outcome ~ age`
    * Second iteration: `outcome ~ gender`
    * Third iteration: `outcome ~ driving_experience`
  
**2. Generating predictions**
   ```
   predictions = model.predict(car_insurance_clean)
   ```
   Logistic Regression returns probabilities between 0 and 1.

**3. Converting probabilities into classes**
   ```
   predicted_classes = (predictions >= 0.5).astype(int)
   ```
   We use 0.5 as the classification threshold.

   | Probability | Predicted Class |
   |---:|---:|
   | 0.85 | 1 |
   | 0.25 | 0 |
   | 0.63 | 1 |

**5. Calculating accuracy**
   ```
   accuracy = (predicted_classes == car_insurance_clean['outcome']).mean()
   ```
   Accuracy measures the proportion of correct predictions.
  
   $$
   \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
   $$
  
   For example, if the model correctly predicts 8,000 out of 10,000 customers:
  
   $$
   \text{Accuracy} = \frac{8000}{10000} = 0.80
   $$
  
  This means the model correctly predicts **80% of the customers**.
   
**6. Store the accuracy**
   ```
   accuracies[feature] = accuracy
   ```
   This stores each feature and its corresponding accuracy in our dictionary.
