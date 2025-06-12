---
title: Multivariate Statistics
subtitle: Foundations of Statistical Analysis in Python
abstract: This notebook explores multivariate relationships through linear regression analysis, highlighting its strengths and limitations. Practical examples and visualizations are provided to help users understand and apply these statistical concepts effectively.
author:
  - name: Karol Flisikowski
    affiliations: 
      - Gdansk University of Technology
      - Chongqing Technology and Business University
    orcid: 0000-0002-4160-1297
    email: karol@ctbu.edu.cn
date: 2025-05-25
---

## Goals of this lecture

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measurement of the relationship between distributions using **linear, regression analysis**.

## Importing relevant libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss
```


```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```


```python
import pandas as pd
df_estate = pd.read_csv("real_estate.csv")
df_estate.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>house age</th>
      <th>distance to the nearest MRT station</th>
      <th>number of convenience stores</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>house price of unit area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
    </tr>
  </tbody>
</table>
</div>



## Describing *multivariate* data with regression models

- So far, we've been focusing on *univariate and bivariate data*: analysis.
- What if we want to describe how *two or more than two distributions* relate to each other?

1. Let's simplify variables' names:


```python
df_estate = df_estate.rename(columns={
    'house age': 'house_age_years',
    'house price of unit area': 'price_twd_msq',
    'number of convenience stores': 'n_convenience',
    'distance to the nearest MRT station': 'dist_to_mrt_m'
})

df_estate.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>house_age_years</th>
      <th>dist_to_mrt_m</th>
      <th>n_convenience</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_twd_msq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
    </tr>
  </tbody>
</table>
</div>



We can also perform binning for "house_age_years":


```python
df_estate['house_age_cat'] = pd.cut(
    df_estate['house_age_years'],
    bins=[0, 15, 30, 45],
    include_lowest=True,
    right=False
)
df_estate.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>house_age_years</th>
      <th>dist_to_mrt_m</th>
      <th>n_convenience</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_twd_msq</th>
      <th>house_age_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
      <td>[30, 45)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
      <td>[15, 30)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
      <td>[0, 15)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
      <td>[0, 15)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
      <td>[0, 15)</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_dict = {
    pd.Interval(left=0, right=15, closed='left'): '0-15',
    pd.Interval(left=15, right=30, closed='left'): '15-30',
    pd.Interval(left=30, right=45, closed='left'): '30-45'
}

df_estate['house_age_cat_str'] = df_estate['house_age_cat'].map(cat_dict)
df_estate['house_age_cat_str'] = df_estate['house_age_cat_str'].astype('category')
df_estate.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>house_age_years</th>
      <th>dist_to_mrt_m</th>
      <th>n_convenience</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_twd_msq</th>
      <th>house_age_cat</th>
      <th>house_age_cat_str</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
      <td>[30, 45)</td>
      <td>30-45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
      <td>[15, 30)</td>
      <td>15-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
      <td>[0, 15)</td>
      <td>0-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
      <td>[0, 15)</td>
      <td>0-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
      <td>[0, 15)</td>
      <td>0-15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_estate.house_age_cat_str.dtype
```




    CategoricalDtype(categories=['0-15', '15-30', '30-45'], ordered=True, categories_dtype=object)




```python
df_estate.isna().any()
```




    No                   False
    house_age_years      False
    dist_to_mrt_m        False
    n_convenience        False
    latitude             False
    longitude            False
    price_twd_msq        False
    house_age_cat        False
    house_age_cat_str    False
    dtype: bool



## Descriptive Statistics

Prepare a heatmap with correlation coefficients on it:


```python
corr_matrix = df_estate.iloc[:, :6].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()
```


    
![png](Exercise10_files/Exercise10_15_0.png)
    


Draw a scatter plot of n_convenience vs. price_twd_msq:


```python
# Your code here
sns.scatterplot(data=df_estate , x="n_convenience", y="price_twd_msq")
plt.title("Convenience Stores vs Price")
plt.show()
```


    
![png](Exercise10_files/Exercise10_17_0.png)
    


Draw a scatter plot of house_age_years vs. price_twd_msq:


```python
# Your code here
sns.scatterplot(data=df_estate , x="house_age_years", y="price_twd_msq")
plt.title("House Age vs Price")
plt.show()
```


    
![png](Exercise10_files/Exercise10_19_0.png)
    


Draw a scatter plot of distance to nearest MRT station vs. price_twd_msq:


```python
# Your code here
sns.scatterplot(data=df_estate, x="dist_to_mrt_m", y="price_twd_msq")
plt.title("Distance to MRT vs. Price per m²")
plt.show()
```


    
![png](Exercise10_files/Exercise10_21_0.png)
    


Plot a histogram of price_twd_msq with 10 bins, facet the plot so each house age group gets its own panel:


```python
# Your code here
df_estate["house_age_group"] = pd.cut(df_estate["house_age_years"], bins=[0, 5, 10, 20, 30, 100], labels=["0–5", "5–10", "10–20", "20–30", "30+"])

filtered = df_estate[df_estate["house_age_years"].isin([5, 10, 15, 20])]
sns.displot(data=filtered, x="price_twd_msq", bins=10, col="house_age_years", col_wrap=4)
plt.show()
```


    
![png](Exercise10_files/Exercise10_23_0.png)
    


Summarize to calculate the mean, sd, median etc. house price/area by house age:


```python

# Your code here
df_estate.groupby("house_age_years")["price_twd_msq"].agg(["mean", "std", "median", "min", "max", "count"])

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>median</th>
      <th>min</th>
      <th>max</th>
      <th>count</th>
    </tr>
    <tr>
      <th>house_age_years</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>54.135294</td>
      <td>11.325466</td>
      <td>52.20</td>
      <td>37.9</td>
      <td>73.6</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>50.700000</td>
      <td>NaN</td>
      <td>50.70</td>
      <td>50.7</td>
      <td>50.7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1.1</th>
      <td>49.780000</td>
      <td>3.511695</td>
      <td>49.00</td>
      <td>45.1</td>
      <td>54.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>48.700000</td>
      <td>1.414214</td>
      <td>48.70</td>
      <td>47.7</td>
      <td>49.7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1.7</th>
      <td>50.400000</td>
      <td>NaN</td>
      <td>50.40</td>
      <td>50.4</td>
      <td>50.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40.9</th>
      <td>54.350000</td>
      <td>18.879751</td>
      <td>54.35</td>
      <td>41.0</td>
      <td>67.7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>41.3</th>
      <td>47.900000</td>
      <td>18.101934</td>
      <td>47.90</td>
      <td>35.1</td>
      <td>60.7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>41.4</th>
      <td>63.300000</td>
      <td>NaN</td>
      <td>63.30</td>
      <td>63.3</td>
      <td>63.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42.7</th>
      <td>35.300000</td>
      <td>NaN</td>
      <td>35.30</td>
      <td>35.3</td>
      <td>35.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43.8</th>
      <td>42.700000</td>
      <td>NaN</td>
      <td>42.70</td>
      <td>42.7</td>
      <td>42.7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>236 rows × 6 columns</p>
</div>



## Simple model

Run a linear regression of price_twd_msq vs. best, but only 1 predictor:


```python
import statsmodels.api as sm

X = df_estate[['dist_to_mrt_m']]
y = df_estate['price_twd_msq']

X = sm.add_constant(X)

model1 = sm.OLS(y, X).fit()

print(model1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          price_twd_msq   R-squared:                       0.454
    Model:                            OLS   Adj. R-squared:                  0.452
    Method:                 Least Squares   F-statistic:                     342.2
    Date:                Thu, 12 Jun 2025   Prob (F-statistic):           4.64e-56
    Time:                        17:24:07   Log-Likelihood:                -1542.5
    No. Observations:                 414   AIC:                             3089.
    Df Residuals:                     412   BIC:                             3097.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const            45.8514      0.653     70.258      0.000      44.569      47.134
    dist_to_mrt_m    -0.0073      0.000    -18.500      0.000      -0.008      -0.006
    ==============================================================================
    Omnibus:                      140.820   Durbin-Watson:                   2.151
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              988.283
    Skew:                           1.263   Prob(JB):                    2.49e-215
    Kurtosis:                      10.135   Cond. No.                     2.19e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.19e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

What do the above results mean? Write down the model and interpret it.

Discuss model accuracy.

Model:

price_twd_msq=45.85−0.0073×dist_to_mrt_m

Each extra meter from the MRT lowers the price per m² by 0.0073 TWD.

Key Results:

R² = 0.454: Distance explains ~45% of price variation.

P-values < 0.001: Both coefficients are statistically significant.

F-statistic = 342.2: Model is significant overall.

Durbin-Watson = 2.15: No autocorrelation in residuals.

Normality tests: Residuals are skewed and heavy-tailed.

Condition number = 2190: Possible numerical issues (but not due to multicollinearity in this case).

Conclusion:
This simple model shows a clear, significant negative relationship between distance to MRT and housing price. Accuracy is moderate; more predictors would improve performance.

## Model diagnostics

### 4 Diagnostic plots


```python
fig = plt.figure(figsize=(12, 10))
sm.graphics.plot_regress_exog(model1, 'dist_to_mrt_m', fig=fig)
plt.show()
```


    
![png](Exercise10_files/Exercise10_32_0.png)
    


The four plots show standard diagnostic plots for a linear regression model

### Outliers and high levarage points:


```python
fig, ax = plt.subplots(figsize=(8, 6))
sm.graphics.influence_plot(model1, ax=ax, criterion="cooks")
plt.title("Influence Plot (Outliers and High Leverage Points)")
plt.show()
```


    
![png](Exercise10_files/Exercise10_35_0.png)
    


Discussion: This multiple regression model significantly explains about 50.5% of the property price variation. All included predictors (distance to MRT and house age categories) are highly significant and negatively impact price as expected.


## Multiple Regression Model

### Test and training set 

We begin by splitting the dataset into two parts, training set and testing set. In this example we will randomly take 75% row in this dataset and put it into the training set, and other 25% row in the testing set:


```python
encode_dict = {True: 1, False: 0}

house_age_0_15 = df_estate['house_age_cat_str'] == '0-15'
house_age_15_30 = df_estate['house_age_cat_str'] == '15-30'
house_age_30_45 = df_estate['house_age_cat_str'] == '30-45'

df_estate['house_age_0_15'] = house_age_0_15.map(encode_dict)
df_estate['house_age_15_30'] = house_age_15_30.map(encode_dict)
df_estate['house_age_30_45'] = house_age_30_45.map(encode_dict)

df_estate.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No</th>
      <th>house_age_years</th>
      <th>dist_to_mrt_m</th>
      <th>n_convenience</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price_twd_msq</th>
      <th>house_age_cat</th>
      <th>house_age_cat_str</th>
      <th>house_age_group</th>
      <th>house_age_0_15</th>
      <th>house_age_15_30</th>
      <th>house_age_30_45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
      <td>[30, 45)</td>
      <td>30-45</td>
      <td>30+</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
      <td>[15, 30)</td>
      <td>15-30</td>
      <td>10–20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
      <td>[0, 15)</td>
      <td>0-15</td>
      <td>10–20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
      <td>[0, 15)</td>
      <td>0-15</td>
      <td>10–20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
      <td>[0, 15)</td>
      <td>0-15</td>
      <td>0–5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

# 75% training, 25% testing, random_state=12 for reproducibility
train, test = train_test_split(df_estate, train_size=0.75, random_state=12)
```

Now we have our training set and testing set. 

### Variable selection methods

Generally, selecting variables for linear regression is a debatable topic.

There are many methods for variable selecting, namely, forward stepwise selection, backward stepwise selection, etc, some are valid, some are heavily criticized.

I recommend this document: <https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/26/lecture-26.pdf> and Gung's comment: <https://stats.stackexchange.com/questions/20836/algorithms-for-automatic-model-selection/20856#20856> if you want to learn more about variable selection process.

[**If our goal is prediction**]{.ul}, it is safer to include all predictors in our model, removing variables without knowing the science behind it usually does more harm than good!!!

We begin to create our multiple linear regression model:


```python
import statsmodels.formula.api as smf
model2 = smf.ols('price_twd_msq ~ dist_to_mrt_m + house_age_0_15 + house_age_30_45', data = df_estate)
result2 = model2.fit()
result2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>price_twd_msq</td>  <th>  R-squared:         </th> <td>   0.485</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.482</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   128.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 12 Jun 2025</td> <th>  Prob (F-statistic):</th> <td>7.84e-59</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:24:09</td>     <th>  Log-Likelihood:    </th> <td> -1530.2</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   414</td>      <th>  AIC:               </th> <td>   3068.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   410</td>      <th>  BIC:               </th> <td>   3084.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   43.4096</td> <td>    1.052</td> <td>   41.275</td> <td> 0.000</td> <td>   41.342</td> <td>   45.477</td>
</tr>
<tr>
  <th>dist_to_mrt_m</th>   <td>   -0.0070</td> <td>    0.000</td> <td>  -17.889</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.006</td>
</tr>
<tr>
  <th>house_age_0_15</th>  <td>    4.8450</td> <td>    1.143</td> <td>    4.239</td> <td> 0.000</td> <td>    2.598</td> <td>    7.092</td>
</tr>
<tr>
  <th>house_age_30_45</th> <td>   -0.1016</td> <td>    1.355</td> <td>   -0.075</td> <td> 0.940</td> <td>   -2.765</td> <td>    2.562</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>145.540</td> <th>  Durbin-Watson:     </th> <td>   2.124</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1077.318</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.296</td>  <th>  Prob(JB):          </th> <td>1.16e-234</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.466</td>  <th>  Cond. No.          </th> <td>6.17e+03</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.17e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



What about distance to mrt? Please plot its scatterplot with the dependent variable and verify, if any transformation is needed:


```python
sns.scatterplot(data=df_estate, x="dist_to_mrt_m", y="price_twd_msq")
plt.title("Distance to MRT vs. House Price per sqm")
plt.show()
```


    
![png](Exercise10_files/Exercise10_44_0.png)
    



```python
df_estate["log_distance_to_mrt"] = np.log(df_estate["dist_to_mrt_m"])
model3 = smf.ols("price_twd_msq ~ log_distance_to_mrt", data=df_estate).fit()
print(model3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          price_twd_msq   R-squared:                       0.539
    Model:                            OLS   Adj. R-squared:                  0.538
    Method:                 Least Squares   F-statistic:                     482.2
    Date:                Thu, 12 Jun 2025   Prob (F-statistic):           2.52e-71
    Time:                        17:24:09   Log-Likelihood:                -1507.3
    No. Observations:                 414   AIC:                             3019.
    Df Residuals:                     412   BIC:                             3027.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              95.0169      2.637     36.034      0.000      89.833     100.200
    log_distance_to_mrt    -8.9235      0.406    -21.959      0.000      -9.722      -8.125
    ==============================================================================
    Omnibus:                      178.772   Durbin-Watson:                   2.109
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1764.720
    Skew:                           1.566   Prob(JB):                         0.00
    Kurtosis:                      12.617   Cond. No.                         38.5
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Discuss the results...
The cross-validated performance shows an MSE of 32.73 and RMSE of 5.72. This means, on average, the model's price predictions are about 5.72 units off the actual value on unseen data. This indicates its predictive accuracy and generalization ability.



```python
mse_result1 = model1.mse_resid
rse_result1 = np.sqrt(mse_result1)
print('The residual standard error for the above model is:',np.round(mse_result1,3))
```

    The residual standard error for the above model is: 101.375
    


```python
mse_result2 = result2.mse_resid
rse_result2 = np.sqrt(mse_result2)
print('The residual standard error for the above model is:',np.round(rse_result2,3))
```

    The residual standard error for the above model is: 9.796
    

Looking at model summary, we see that variables .... are insignificant, so let's estimate the model without those variables:


```python
model_new = smf.ols("price_twd_msq ~ log_distance_to_mrt", data=df_estate).fit()
print(model_new.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          price_twd_msq   R-squared:                       0.539
    Model:                            OLS   Adj. R-squared:                  0.538
    Method:                 Least Squares   F-statistic:                     482.2
    Date:                Thu, 12 Jun 2025   Prob (F-statistic):           2.52e-71
    Time:                        17:24:09   Log-Likelihood:                -1507.3
    No. Observations:                 414   AIC:                             3019.
    Df Residuals:                     412   BIC:                             3027.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              95.0169      2.637     36.034      0.000      89.833     100.200
    log_distance_to_mrt    -8.9235      0.406    -21.959      0.000      -9.722      -8.125
    ==============================================================================
    Omnibus:                      178.772   Durbin-Watson:                   2.109
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1764.720
    Skew:                           1.566   Prob(JB):                         0.00
    Kurtosis:                      12.617   Cond. No.                         38.5
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

### Evaluating multi-collinearity

There are many standards researchers apply for deciding whether a VIF is too large. In some domains, a VIF over 2 is worthy of suspicion. Others set the bar higher, at 5 or 10. Others still will say you shouldn't pay attention to these at all. Ultimately, the main thing to consider is that small effects are more likely to be "drowned out" by higher VIFs, but this may just be a natural, unavoidable fact with your model.


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = df_estate[['dist_to_mrt_m', 'house_age_0_15', 'house_age_30_45']].copy()
X_vif = X_vif.fillna(0)

X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif_data)
```

               feature       VIF
    0            const  4.772153
    1    dist_to_mrt_m  1.061497
    2   house_age_0_15  1.399276
    3  house_age_30_45  1.400308
    

Discuss the results... This pair plot visually displays relationships between all numerical variables and their individual distributions. It's a quick way to identify potential correlations and understand data patterns. This plot is essential for initial exploratory data analysis.



Finally we test our best model on test dataset (change, if any transformation on dist_to_mrt_m was needed):


```python
X_test = test[['dist_to_mrt_m', 'house_age_0_15', 'house_age_30_45']].copy()
X_test = X_test.fillna(0)
X_test = sm.add_constant(X_test)

y_test = test['price_twd_msq']

y_pred = result2.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")
```

    Test RMSE: 8.38
    

Interpret results... This correlation heatmap shows linear relationships between all variables. Price_twd_msq is strongly negatively correlated with dist_to_mrt_m and positively with n_convenience. We also observe moderate correlations among independent variables like dist_to_mrt_m and n_convenience. This plot effectively highlights key predictors and potential multicollinearity for model building.


## Variable selection using best subset regression

*Best subset and stepwise (forward, backward, both) techniques of variable selection can be used to come up with the best linear regression model for the dependent variable medv.*


```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

X = df_estate[['dist_to_mrt_m', 'n_convenience', 'house_age_0_15', 'house_age_15_30', 'house_age_30_45']]
y = df_estate['price_twd_msq']

lr = LinearRegression()

sfs_forward = SequentialFeatureSelector(lr, n_features_to_select='auto', direction='forward', cv=5)
sfs_forward.fit(X, y)
print("Forward selection support:", sfs_forward.get_support())
print("Selected features (forward):", X.columns[sfs_forward.get_support()].tolist())

sfs_backward = SequentialFeatureSelector(lr, n_features_to_select='auto', direction='backward', cv=5)
sfs_backward.fit(X, y)
print("Backward selection support:", sfs_backward.get_support())
print("Selected features (backward):", X.columns[sfs_backward.get_support()].tolist())
```

    Forward selection support: [ True  True False False False]
    Selected features (forward): ['dist_to_mrt_m', 'n_convenience']
    Backward selection support: [ True  True False False  True]
    Selected features (backward): ['dist_to_mrt_m', 'n_convenience', 'house_age_30_45']
    

### Comparing competing models


```python
import statsmodels.api as sm

features_forward = X.columns[sfs_forward.get_support()].tolist()
X_forward = df_estate[features_forward]
X_forward = sm.add_constant(X_forward)
model_forward = sm.OLS(y, X_forward).fit()
print("AIC (forward selection):", model_forward.aic)

features_backward = X.columns[sfs_backward.get_support()].tolist()
X_backward = df_estate[features_backward]
X_backward = sm.add_constant(X_backward)
model_backward = sm.OLS(y, X_backward).fit()
print("AIC (backward selection):", model_backward.aic)

print(model_forward.summary())
```

    AIC (forward selection): 3057.2813425866216
    AIC (backward selection): 3047.991777087278
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          price_twd_msq   R-squared:                       0.497
    Model:                            OLS   Adj. R-squared:                  0.494
    Method:                 Least Squares   F-statistic:                     202.7
    Date:                Thu, 12 Jun 2025   Prob (F-statistic):           5.61e-62
    Time:                        17:24:09   Log-Likelihood:                -1525.6
    No. Observations:                 414   AIC:                             3057.
    Df Residuals:                     411   BIC:                             3069.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const            39.1229      1.300     30.106      0.000      36.568      41.677
    dist_to_mrt_m    -0.0056      0.000    -11.799      0.000      -0.007      -0.005
    n_convenience     1.1976      0.203      5.912      0.000       0.799       1.596
    ==============================================================================
    Omnibus:                      191.943   Durbin-Watson:                   2.126
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2159.977
    Skew:                           1.671   Prob(JB):                         0.00
    Kurtosis:                      13.679   Cond. No.                     4.58e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.58e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

From Best subset regression and stepwise selection (forward, backward, both), we see that the models selected by forward and backward selection may include different sets of predictors, depending on their contribution to model fit. 

By comparing AIC values, the model with the lowest AIC is preferred, as it balances model complexity and goodness of fit.

In this case, the summary output for the best model (e.g., forward selection) shows which variables are most important for predicting price_twd_msq. This approach helps identify the most relevant predictors and avoid overfitting by excluding unnecessary variables.

Run model diagnostics for the BEST model:


```python
def forward_selection_with_diagnostics(X, y, alpha=0.05):
    included = []
    remaining = list(X.columns)
    
    while remaining:
        best_pval = float('inf')
        best_feature = None
        for feature in remaining:
            X_ = sm.add_constant(X[included + [feature]], has_constant='add')
            model = sm.OLS(y, X_).fit()
            pval = model.pvalues[feature]
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
        if best_pval < alpha:
            included.append(best_feature)
            remaining.remove(best_feature)
        else:
            break

    X_final = sm.add_constant(X[included], has_constant='add')
    model = sm.OLS(y, X_final).fit()

    residuals = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(8, 5))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.show()

    sm.qqplot(residuals, line='45', fit=True)
    plt.title("Normal Q-Q")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=np.sqrt(np.abs(residuals)))
    plt.xlabel("Fitted values")
    plt.ylabel("√|Residuals|")
    plt.title("Scale-Location")
    plt.show()

    sm.graphics.influence_plot(model, criterion="cooks",size=20)
    plt.title("Influence Plot")
    plt.show()

    return model

X_clean = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
model = forward_selection_with_diagnostics(X_clean, y)

print(model.summary())
```


    
![png](Exercise10_files/Exercise10_64_0.png)
    



    
![png](Exercise10_files/Exercise10_64_1.png)
    



    
![png](Exercise10_files/Exercise10_64_2.png)
    



    
![png](Exercise10_files/Exercise10_64_3.png)
    


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          price_twd_msq   R-squared:                       0.536
    Model:                            OLS   Adj. R-squared:                  0.533
    Method:                 Least Squares   F-statistic:                     158.1
    Date:                Thu, 12 Jun 2025   Prob (F-statistic):           4.50e-68
    Time:                        17:24:10   Log-Likelihood:                -1508.6
    No. Observations:                 414   AIC:                             3025.
    Df Residuals:                     410   BIC:                             3041.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    const             35.6792      1.377     25.906      0.000      32.972      38.387
    dist_to_mrt_m     -0.0052      0.000    -11.225      0.000      -0.006      -0.004
    n_convenience      1.3115      0.196      6.705      0.000       0.927       1.696
    house_age_0_15     5.5012      0.928      5.927      0.000       3.677       7.326
    ==============================================================================
    Omnibus:                      205.216   Durbin-Watson:                   2.089
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2594.066
    Skew:                           1.784   Prob(JB):                         0.00
    Kurtosis:                      14.733   Cond. No.                     5.33e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.33e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

Finally, we can check the Out-of-sample Prediction or test error (MSPE):


```python
X_test = test[features_forward].copy()
X_test = X_test.fillna(0)
X_test = sm.add_constant(X_test)

y_test = test['price_twd_msq']

y_pred = model_forward.predict(X_test)

mspe = np.mean((y_test - y_pred) ** 2)
print(f"Test MSPE (out-of-sample): {mspe:.2f}")
```

    Test MSPE (out-of-sample): 64.80
    

## Cross Validation

In Python, for cross-validation of regression models is usually done with cross_val_score from sklearn.model_selection.

To get the raw cross-validation estimate of prediction error (e.g., mean squared error), use:


```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

X = df_estate[['dist_to_mrt_m', 'house_age_0_15', 'house_age_30_45']]
y = df_estate['price_twd_msq']

model = LinearRegression()

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

cv_mse = -cv_scores.mean()
cv_rmse = np.sqrt(cv_mse)

print(f"Cross-validated MSE: {cv_mse:.2f}")
print(f"Cross-validated RMSE: {cv_rmse:.2f}")
```

    Cross-validated MSE: 95.90
    Cross-validated RMSE: 9.79
    

# Summary

1. Do you understand all numerical measures printed in the SUMMARY of the regression report?
2. Why do we need a cross-validation?
3. What are the diagnostic plots telling us?
4. How to compare similar, but competing models?
5. What is VIF telling us?
6. How to choose best set of predictors for the model?

## 1. Key measures include:

R-squared: Proportion of variance explained by the model.

Adjusted R-squared: Adjusted for number of predictors; useful when comparing models.

F-statistic & p-value: Tests overall significance of the regression.

Coefficients: Show the effect size of each predictor.

Standard Error & t-values: Used for significance testing of coefficients.

P-values: Help assess if individual predictors are statistically significant.

Confidence Intervals: Estimate the range in which the true coefficient lies.

## 2. 
To assess how well a model generalizes to unseen data. It helps detect overfitting and gives a more robust estimate of prediction error compared to a single train-test split.

## 3. 
They help assess model assumptions:

Residuals vs Fitted: Checks linearity and equal variance (homoscedasticity).

Q-Q Plot: Assesses normality of residuals.

Scale-Location: Checks homoscedasticity again.

Leverage/Influence plots: Identifies outliers or influential observations.

## 4. 
Compare Adjusted R-squared.

Use metrics like AIC, BIC.

Perform cross-validation and compare MSPE.

Analyze residual plots for patterns.

Compare VIFs to detect multicollinearity.

## 5. 
Variance Inflation Factor (VIF) quantifies multicollinearity. A VIF > 10 suggests a high correlation between predictors, which can inflate variances of coefficient estimates and make them unstable.

## 6. 
Stepwise selection (forward/backward).

Information criteria like AIC/BIC.

Cross-validation to test model generalizability.

Domain knowledge to prioritize relevant features.
