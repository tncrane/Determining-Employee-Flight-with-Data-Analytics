# **Capstone project: Providing data-driven suggestions for HR**

### Understand the business scenario and problem

The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?

Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.


```python
## Imported various Python packages.
## Data exploration and visualization packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

## Model building and statistical analysis packages.
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## Model metrics packages.
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree
from sklearn import metrics

## Saving models.
import pickle
```


```python
## The Salifort HR dataset will just be called Sali. Head function was used for a first glance at the dataset.
df0 = pd.read_csv("HR_capstone_dataset.csv")
sali = df0


# Display first few rows of the dataframe
sali.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



### Gather basic information about the data


```python
## Info function was used to gather more basic information about the Salifort dataset. Based on there being 14,999 entries in 
## every column, there appears to be no missing data. We can verify that later. Note that cells were reran after the project 
## was complete, and so the duplicates have already been dropped, so we have 11,991 entries instead of the original 14,999.
sali.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11991 entries, 0 to 11999
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     11991 non-null  float64
     1   last_evaluation        11991 non-null  float64
     2   number_project         11991 non-null  int64  
     3   average_monthly_hours  11991 non-null  int64  
     4   time_spent_company     11991 non-null  int64  
     5   work_accident          11991 non-null  int64  
     6   left                   11991 non-null  int64  
     7   promotion_last_5years  11991 non-null  int64  
     8   department             11991 non-null  object 
     9   salary                 11991 non-null  object 
    dtypes: float64(2), int64(6), object(2)
    memory usage: 1.3+ MB


### Gather descriptive statistics about the data


```python
## Describe function was used to gather information on mean, median, standard deviation, and more on the dataset.
sali.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.612834</td>
      <td>0.716102</td>
      <td>3.803054</td>
      <td>201.050337</td>
      <td>3.498233</td>
      <td>0.144610</td>
      <td>0.238083</td>
      <td>0.021268</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.248631</td>
      <td>0.171169</td>
      <td>1.232592</td>
      <td>49.943099</td>
      <td>1.460136</td>
      <td>0.351719</td>
      <td>0.425924</td>
      <td>0.144281</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.440000</td>
      <td>0.560000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.640000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.870000</td>
      <td>5.000000</td>
      <td>245.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Rename columns


```python
## Columns function was used to look at the column names again. It looks like 'Department' and 'Work_accident' will have to be
## reformmatted to the otherwise uniform snake_case and lowercase format in all other columns. This makes data exploring and
## general coding much easier. 
sali.columns

```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_monthly_hours', 'time_spent_company', 'work_accident', 'left',
           'promotion_last_5years', 'department', 'salary'],
          dtype='object')




```python
## Renamed 'Department' and 'Work_accident' to their snake_case, lowercase counterparts. Also corrected spelling on 
## 'average_montly_hours' and made 'time_spend_company' past tense. 
sali = sali.rename(columns = {'Work_accident': 'work_accident',
                             'time_spend_company': 'time_spent_company',
                             'Department': 'department', 
                             'average_montly_hours': 'average_monthly_hours'})
## Double checked the column names to affirm that all title changes took place.
sali.columns
```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_monthly_hours', 'time_spent_company', 'work_accident', 'left',
           'promotion_last_5years', 'department', 'salary'],
          dtype='object')



### Check missing values


```python
## Checked for missing or null values. There are zero null values, and we know from before there are zero missing values.
sali.isnull().sum()

```




    satisfaction_level       0
    last_evaluation          0
    number_project           0
    average_monthly_hours    0
    time_spent_company       0
    work_accident            0
    left                     0
    promotion_last_5years    0
    department               0
    salary                   0
    dtype: int64



### Check duplicates


```python
## Checked Sali dataset for duplicate values. There are over 3,000 dupilcate values! Duplicates cannot be part of our
## statistical or machine learning models. 
sali.duplicated().sum()
```




    3008




```python
## Inspected the duplicate values a little further. 
duplicates = sali.duplicated()
print(sali[duplicates])

```

           satisfaction_level  last_evaluation  number_project  \
    396                  0.46             0.57               2   
    866                  0.41             0.46               2   
    1317                 0.37             0.51               2   
    1368                 0.41             0.52               2   
    1461                 0.42             0.53               2   
    ...                   ...              ...             ...   
    14994                0.40             0.57               2   
    14995                0.37             0.48               2   
    14996                0.37             0.53               2   
    14997                0.11             0.96               6   
    14998                0.37             0.52               2   
    
           average_monthly_hours  time_spent_company  work_accident  left  \
    396                      139                   3              0     1   
    866                      128                   3              0     1   
    1317                     127                   3              0     1   
    1368                     132                   3              0     1   
    1461                     142                   3              0     1   
    ...                      ...                 ...            ...   ...   
    14994                    151                   3              0     1   
    14995                    160                   3              0     1   
    14996                    143                   3              0     1   
    14997                    280                   4              0     1   
    14998                    158                   3              0     1   
    
           promotion_last_5years  department  salary  
    396                        0       sales     low  
    866                        0  accounting     low  
    1317                       0       sales  medium  
    1368                       0       RandD     low  
    1461                       0       sales     low  
    ...                      ...         ...     ...  
    14994                      0     support     low  
    14995                      0     support     low  
    14996                      0     support     low  
    14997                      0     support     low  
    14998                      0     support     low  
    
    [3008 rows x 10 columns]



```python
## Dropped the duplicate values of the Sali dataset. Then we confirmed the dataset has no duplicate values.
sali.drop_duplicates(keep = 'first', inplace = True)
sali.duplicated().sum()

```




    0



### Check outliers


```python
## Made a box plot of employee tenure at Salifort. This helps with initial understanding of the data, especially in terms of its 
## distribution. The largest concentration of tenure is between 3 and 4 years, with many more being in between 2 and 5 years.
## This seems like a typical rangeo of tenure for a company.
plt.figure(figsize = (5, 5))
plt.title('Tenure Distribution in Salifort', fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
sns.boxplot(x = sali['time_spent_company'])
plt.show

```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_18_1.png)



```python
### Used quantile function to find the first and third quartile values as well as the interquartile range.
percentile25 = sali['time_spent_company'].quantile(0.25)
percentile75 = sali['time_spent_company'].quantile(0.75)
innerq = percentile75 - percentile25

## Calculated the upper and lower limits using some formulas and our 1st and 3rd quartile values. 
upper_limit = percentile75 + 1.5 * innerq
lower_limit = percentile25 - 1.5 * innerq
print('upper:', upper_limit)
print('lower:', lower_limit)

## Defined outliers so that we can exclude them from models sensitive to outliers, if needed. Outliers are part of the data, 
## and caution should be used in excluding any data from the original set, but to get the best model possible, we need to keep
## our options open. 
outliers = sali[(sali['time_spent_company'] > upper_limit) | (sali['time_spent_company'] < lower_limit)]
print('Rows in Salifort data containing outliers:', len(outliers))
```

    upper: 5.5
    lower: 1.5
    Rows in Salifort data containing outliers: 824



```python
## Calculated the percentage of people who left the company versus the amount of employees in dataset. 83% of employees have 
## been retained while ~17% have left the company. 
print(sali['left'].value_counts())
print(sali['left'].value_counts(normalize = True))

```

    0    10000
    1     1991
    Name: left, dtype: int64
    0    0.833959
    1    0.166041
    Name: left, dtype: float64


### Data visualizations

Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.


```python
## A simple line chart shows satisfaction levels meandering about for the part time employees until about 160 monthly
## hours, reaching and staying at their relative peak until about 270 monthly hours, after which there is a drastic drop off.
## Note that average_monthly_hours is not a chronological variable, but this simple chart is a good start to understand some 
## basic relationships. Therefore, employees seem the most satisfied when they are between 160 and 270 monthly working hours. 
sns.lineplot(x='average_monthly_hours', y='satisfaction_level', data = sali)
plt.title('Satisfaction Level with Monthly Hours', fontsize='14')
plt.show()

```


![png](output_23_0.png)



```python
## One histogram shows the predictable reality that lower satisfaction levels, especially the absolute lowest, are strongly
## correlated with employees leaving Salifort. For some of the lower values, there are actually more employees leaving than 
## staying, despite the overall percentages of staying employees being much higher than those who left (83% vs 17%). The other 
## histogram shows retained and associates who left per department. The three departments with the largest sheer numbers of 
## employees who left are sales, technical, and support. However, these deparments also have by far the most employees overall,
## and the histogram bars do not seem particularly high in favor of 'leaving' employees, despite one's initial assumptions about
## sales positions. ;)
sns.histplot(x = 'satisfaction_level', data = sali, hue = 'left', bins = 10, alpha = .7)
plt.title("Satisfaction Levels Effect on Employee Retention")
plt.show()

plt.figure(figsize = (12, 6))
ax = sns.histplot(data = sali, x = 'department', hue = 'left', multiple = 'dodge', shrink = 1)
plt.title("Retention Per Department")
plt.show()
```


![png](output_24_0.png)



![png](output_24_1.png)



```python
## Here is a scatterplot to more effectively show the variables of average_monthly_hours and satisfaction_level. Notice the 3
## major concentrations of employees who left the company: the very dense cluster in the lower left portion, the more spread 
## out cocentration in the upper right portion, and the elongated concentration in the lower right portion. One might surmise 
## that these clusters of 'left' employees represent underutilized/temporary employees, highly successful individuals who 
## inevitably leave for greener pastures, and overworked/burnt out employees, respectively. These are simply assumptions, but
## are true to a degree in many organizations.
plt.figure(figsize=(20, 10))
sns.scatterplot(data=sali, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.5)
plt.legend(labels=['left', 'stayed'])
plt.title('Workload Effect on Satisfaction Level', fontsize='14');

```


![png](output_25_0.png)



```python
## Made a copy of the Sali dataset changed the 'salary' variable to a numeric one, and split the categorical 'department' 
## variable into several binary columns. This allows the two variables with string outputs to be mathematically included into 
## the upcoming machine learning model algorithms. 
sali_copy = sali.copy()
sali_copy['salary'] = (sali_copy['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes)
sali_copy = pd.get_dummies(sali_copy, drop_first = False)

sali_copy.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>time_spent_company</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## We defined our predictor variables and our outcome variable as X and y, respectively. X will be every other variable in the 
## dataset since none of them are irrelvant to the outcome variable. y of course is the variable 'left' (the condition of an 
## employee leaving Salifort Motors. Then, we used the train_test_split function to split the data into testing and training 
## sets.
X = sali_copy.drop('left', axis = 1)
y = sali_copy['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)

## We are building a tree-based ensemble model using XGBoost. It has many advantages such as avoiding overfitting bias, each
## subsequent learner adapts from the errors of previous learners, does not have to be scaled or normalized, can handle numeric 
## and categorical features, is resiliant to outliers, and can handle multicollinearity amongst features. We first had to set 
## our paramters for the various, randomly generated learners or 'trees' the variety of maximum depths or decisions a given tree
## can make, a variety of learning rates that assign lower and higher values to subsequent learners (trees), and various end 
## node conditions using the parameter min_child_weight. 
cv_params = {'max_depth': [2, 4, 6, 8, 10],
            'min_child_weight': [3, 5, 10, 15],
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 150]}

## Then we specified the type of model to be used. On the next coding line we specified the desired scoring metrics, and on the 
## final line we instantiated the model. 
xgb = XGBClassifier(objective = 'binary:logistic', random_state = 0)

scoring = {'accuracy', 'precision', 'recall', 'f1'}

xgbcv = GridSearchCV(xgb, cv_params, scoring = scoring, cv = 5, refit = 'f1')

```


```python
%%time
## The model was fitted and then ran on the X and y training data. We also used the magic command %%time to get a the amount of
## time the model took to train. 
xgbcv = xgbcv.fit(X_train, y_train)
xgbcv
```

    CPU times: user 17min 21s, sys: 5.77 s, total: 17min 26s
    Wall time: 8min 44s





    GridSearchCV(cv=5, error_score=nan,
                 estimator=XGBClassifier(base_score=None, booster=None,
                                         callbacks=None, colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=None,
                                         early_stopping_rounds=None,
                                         enable_categorical=False, eval_metric=None,
                                         gamma=None, gpu_id=None, grow_policy=None,
                                         importance_type=None,
                                         interaction_constraints=None,
                                         learning_rate=None, max...
                                         objective='binary:logistic',
                                         predictor=None, random_state=0,
                                         reg_alpha=None, ...),
                 iid='deprecated', n_jobs=None,
                 param_grid={'learning_rate': [0.1, 0.2, 0.3],
                             'max_depth': [2, 4, 6, 8, 10],
                             'min_child_weight': [3, 5, 10, 15],
                             'n_estimators': [50, 100, 150]},
                 pre_dispatch='2*n_jobs', refit='f1', return_train_score=False,
                 scoring={'recall', 'precision', 'accuracy', 'f1'}, verbose=0)




```python

```


```python
## The model was pickled for future use. This saves lots of time for us since the model will not have to be refitted. 
path = '/home/jovyan/work/'
with open(path + 'sali_xgb.pickle', 'wb') as to_write:
    pickle.dump(xgbcv, to_write) 


```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-85e09a64f986> in <module>
          2 path = '/home/jovyan/work/'
          3 with open(path + 'sali_xgb.pickle', 'wb') as to_write:
    ----> 4     pickle.dump(xgbcv, to_write)
          5 


    NameError: name 'pickle' is not defined



```python
## Model was tested on the X and y test data using the predict function. The print function was used to give an easy to follow 
## list of important metrics of the model. Note that these metrics are our scoring parameters from before.
xgbcv_preds = xgbcv.predict(X_test)
print('f1: ', f1_score(y_test, xgbcv_preds))
print('precision: ', precision_score(y_test, xgbcv_preds))
print('recall', recall_score(y_test, xgbcv_preds))
print('accuracy: ', accuracy_score(y_test, xgbcv_preds))


```

    f1:  0.9514963880288957
    precision:  0.9787685774946921
    recall 0.9257028112449799
    accuracy:  0.9843228819212808



```python
## A confusion matrix was created to visualize the performance of the model. The model did surprisingly well on the test data, 
## with only 47 total entries being falsely predicted out of about 3,000. 
cm = metrics.confusion_matrix(y_test, xgbcv_preds)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = xgbcv.classes_)
display.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f88229ecdd0>




![png](output_32_1.png)



```python
## Used the plot_importance function to visualize which variables were most predictive of the outcome variable 'left', which we 
## can recall represents employees who have left the Salifort company. Note that the top three predictors are average monthly
## hours worked, satisfaction level, and last evaluation score, with number of projects and tenure with the company also being
## considerable predictors. This disproves my initial hypothesis of satisfaction level being the overwhelming favorite amongst
## the predictors. Satisfaction levels are simply a representation of how the employee feels, which in turn is due to numerous 
## other issues, many of which are already part of this model such as monthly hours and tenure with the company. The point here
## is that satisfaction level, while being a strong predictor, is not a good root cause for a business to attempt to improve,
## since it depends on many other factors. 
plot_importance(xgbcv.best_estimator_)
plt.figure(figsize = (20, 10))
plt.show()

```


![png](output_33_0.png)



    <Figure size 1440x720 with 0 Axes>



```python
## Asked for the best score amongst the models. 
xgbcv.best_score_
```




    0.9473512998200164




```python
## Found out which parameters had the best performance in the XGBoost model. 
xgbcv.best_params_
```




    {'learning_rate': 0.1,
     'max_depth': 6,
     'min_child_weight': 3,
     'n_estimators': 100}


