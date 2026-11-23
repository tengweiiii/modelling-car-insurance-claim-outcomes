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
  predicted_classes = (predictions >= 0.5).astype(int)

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

Output:
```
Optimization terminated successfully.
         Current function value: 0.511794
         Iterations 6
Optimization terminated successfully.
         Current function value: 0.615951
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.467092
         Iterations 8
Optimization terminated successfully.
         Current function value: 0.603742
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.531499
         Iterations 6
Optimization terminated successfully.
         Current function value: 0.572649
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.552412
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.572668
         Iterations 6
Optimization terminated successfully.
         Current function value: 0.586659
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.595431
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.617345
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.605605
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.621700
         Iterations 5
Optimization terminated successfully.
         Current function value: 0.558922
         Iterations 7
Optimization terminated successfully.
         Current function value: 0.598699
         Iterations 6
Optimization terminated successfully.
         Current function value: 0.549220
         Iterations 7
```

## 3.4 Compare the model accuracy
Once all 16 models have been trained, let's convert our dictionary into a DataFrame.
```
# Convert accuracy dictoniary into a DataFrame
accuracy_df = pd.DataFrame(
  list(accuracies.items()),
  columns=['features', 'accuracy']
)

# Sort models by accuracy in descending order
accuracy_df = accuracy_df.sort_values(
  by='accuracy',
  ascending=False
)

# Display the results
accuracy_df
```
Output:
| index | features            | accuracy |
|------:|---------------------|---------:|
| 2     | driving_experience  | 0.7771   |
| 0     | age                 | 0.7747   |
| 4     | income              | 0.7425   |
| 6     | vehicle_ownership   | 0.7351   |
| 5     | credit_score        | 0.7053   |
| 11    | annual_mileage      | 0.6904   |
| 1     | gender              | 0.6867   |
| 3     | education           | 0.6867   |
| 7     | vehicle_year        | 0.6867   |
| 8     | married             | 0.6867   |
| 9     | children            | 0.6867   |
| 10    | postal_code         | 0.6867   |
| 12    | vehicle_type        | 0.6867   |
| 13    | speeding_violations | 0.6867   |
| 14    | duis                | 0.6867   |
| 15    | past_accidents      | 0.6867   |

## Important modelling consideration
In the code above, we train and evaluate each model using the same 10,000 records.

This measures training accuracy, not how well the model generalizes to unseen customers.

For this exercise, we are following the straightforward approach to identify the highest-accuracy feature. 

In a real production project, we should use a **train-test split** or **cross-validation** to obtain a more reliable performance estimate.

## 3.5 Interpretation of Logistic Regression Modelling
A total of 16 individual Logistic Regression models were trained, with each model using one feature to predict the insurance claim outcome (`outcome`). Predicted probabilities were converted into binary classes using a classification threshold of 0.5, and the accuracy of each model was calculated to evaluate its predictive performance.

The results showed that `driving_experience` achieved the highest accuracy of 77.71%, followed closely by `age` at 77.47%. This indicates that **driving experience** and **age** provide the strongest predictive performance among the individual features.

On the other hand, several features achieved an accuracy of 68.67%, which was equivalent to the majority-class baseline. This suggests that these features did not improve prediction accuracy when used individually.

Overall, the modelling process allows us to compare the predictive performance of each feature and identify `driving_experience` as the most suitable single feature for predicting car insurance claim outcomes.

**Key takeaway**: By training and evaluating individual Logistic Regression models, we identified `driving_experience` as the best-performing feature, with an accuracy of 77.71%.
