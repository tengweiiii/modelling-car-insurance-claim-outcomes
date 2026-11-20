# 1.0 Data Exploration
## 1.1 Import the required libraries
```
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
```
| Library	| Purpose |
| -------- | -------- |
| pandas	| Data manipulation and analysis|
| numpy	| Numerical operations|
| logit	| Build Logistic Regression models using statistical formulas|

## 1.2 Load the dataset
```
# Load the dataset
car_insurance = pd.read_csv('car_insurance.csv')

# Display the first five rows
car_insurance.head()
```
* `pd.read_csv()` read the csv file and stores the data as a panda DataFrame.
* `head()` displays the first five rows so we can understand the structure.

## 1.3 Inspect the dataset
### Check the number of rows and columns
```
# Check the number of rows and columns
car_insurance.shape
```
Output:
```
(10000, 18)
```

### Check column names and data types
```
# Check column names and data types
car_insurance.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 18 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   id                   10000 non-null  int64  
 1   age                  10000 non-null  int64  
 2   gender               10000 non-null  int64  
 3   driving_experience   10000 non-null  object 
 4   education            10000 non-null  object 
 5   income               10000 non-null  object 
 6   credit_score         9018 non-null   float64
 7   vehicle_ownership    10000 non-null  float64
 8   vehicle_year         10000 non-null  object 
 9   married              10000 non-null  float64
 10  children             10000 non-null  float64
 11  postal_code          10000 non-null  int64  
 12  annual_mileage       9043 non-null   float64
 13  vehicle_type         10000 non-null  object 
 14  speeding_violations  10000 non-null  int64  
 15  duis                 10000 non-null  int64  
 16  past_accidents       10000 non-null  int64  
 17  outcome              10000 non-null  float64
dtypes: float64(6), int64(7), object(5)
memory usage: 1.4+ MB
```

### Check the statistical summaries
```
# Check the statistical summaries
car_insurance.describe()
```
Output: 
| Statistic | index | id | age | gender | credit_score | vehicle_ownership | married | children | postal_code | annual_mileage | speeding_violations | duis | past_accidents | outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 10000 | 10000 | 10000 | 9018 | 10000 | 10000 | 10000 | 10000 | 9043 | 10000 | 10000 | 10000 | 10000 | 10000 |
| mean | 500521.9068 | 1.4895 | 0.499 | 0.5158128096 | 0.697 | 0.4982 | 0.6888 | 19864.5484 | 11697.0032069004 | 1.4829 | 0.2392 | 1.0563 | 0.3133 |
| std | 290030.7687578594 | 1.0252779474 | 0.5000240018 | 0.1376880893 | 0.4595781992 | 0.5000217618 | 0.4630075567 | 18915.6138547288 | 2818.4345283018 | 2.2419657069 | 0.5549902354 | 1.6524537423 | 0.4638584121 |
| min | 101 | 0 | 0 | 0.0533575455 | 0 | 0 | 0 | 10238 | 2000 | 0 | 0 | 0 | 0 |
| 25% | 249638.5 | 1 | 0 | 0.4171913163 | 0 | 0 | 0 | 10238 | 10000 | 0 | 0 | 0 | 0 |
| 50% | 501777 | 1 | 0 | 0.5250327586 | 1 | 0 | 1 | 10238 | 12000 | 0 | 0 | 0 | 0 |
| 75% | 753974.5 | 2 | 1 | 0.6183116647 | 1 | 1 | 1 | 32765 | 14000 | 2 | 0 | 2 | 1 |
| max | 999976 | 3 | 1 | 0.9608188278 | 1 | 1 | 1 | 92101 | 22000 | 22 | 6 | 15 | 1 |

### Check missing values
```
# Check missing values
car_insurance.isnull().sum()
```
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
| credit_score         | 982            |
| vehicle_ownership    | 0              |
| vehicle_year         | 0              |
| married              | 0              |
| children             | 0              |
| postal_code          | 0              |
| annual_mileage       | 957            |
| vehicle_type         | 0              |
| speeding_violations  | 0              |
| duis                 | 0              |
| past_accidents       | 0              |
| outcome              | 0              |
