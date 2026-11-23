# 4.0 Model Evaluation and Comparison
## 4.2 Compare model accuracy
The results show that `driving_experience` achieved the highest accuracy, followed closely by `age`.

**Key Observation**
| Feature | Accuracy | Interpretation |
|---|---:|---|
| driving_experience | 77.71% | Best-performing predictor |
| age | 77.47% | Very close to the best model |
| income | 74.25% | Relatively strong predictor |
| vehicle_ownership | 73.51% | Relatively strong predictor |
| credit_score | 70.53% | Moderate predictive performance |
| annual_mileage | 69.04% | Slightly above baseline |
| Remaining features | 68.67% | Same as baseline |

## 4.2 Interpretation of the results
**1. Driving experience achieved the highest accuracy.**

`driving_experience` achieved an accuracy of 77.71%, making it the best single feature for predicting insurance claim outcomes.

This suggests that a customer's driving experience provides useful information about their likelihood of making an insurance claim.

**2. Age was another strong predictor.**

Age achieved 77.47% accuracy, only 0.24 percentage points below driving experience.

This suggests that age is also strongly associated with insurance claim outcomes.

**3. Income and vehicle ownership showed relatively good performance.**

Income achieved 74.25%, while vehicle ownership achieved 73.51%.

These features may provide additional information about differences in customer characteristics and claim behaviour.

**4. Several features performed at the baseline accuracy.**

Gender, education, vehicle year, marital status, children, postal code, vehicle type, speeding violations, DUIs and past accidents all achieved 68.67%.

This is the same accuracy as the majority-class baseline, meaning these individual models did not improve overall accuracy over simply predicting that every customer would make no claim.

Note that this does not necessarily mean these features have no relationship with insurance claims. It means they did not improve accuracy under the current modelling approach and classification threshold.

## 4.3 Identify the best feature
```
# Identify the row with the highest accuracy
best_model = accuracy_df.iloc[0]

# Create the required DataFrame
best_feature_df = pd.DataFrame({
    'best_feature': [best_model['features']],
    'best_accuracy': [best_model['accuracy']]
})

# Display the result
best_feature_df
```
Output:
| index | best_feature       | best_accuracy |
|------:|--------------------|--------------:|
| 0     | driving_experience | 0.7771        |

## Final project interpretation
`driving_experience` was identified as the best single predictor of car insurance claim outcomes, achieving the highest Logistic Regression accuracy of 77.71% among the 16 candidate features. 

Age was the second-best predictor with an accuracy of 77.47%, followed by income at 74.25% and vehicle ownership at 73.51%. 

Several other features achieved 68.67% accuracy, which was equivalent to the majority-class baseline. 

These results suggest that driving experience provides the strongest predictive performance when used individually to predict whether a customer will make an insurance claim.
