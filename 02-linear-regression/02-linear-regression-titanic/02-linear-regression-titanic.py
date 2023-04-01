#%%
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: ai
#     language: python
#     name: python3
# ---

# ## Foundations: Clean Data
#
# Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition.
#
# This dataset contains information about 891 people who were on board the ship when departed on April 15th, 1912. As noted in the description on Kaggle's website, some people aboard the ship were more likely to survive the wreck than others. There were not enough lifeboats for everybody so women, children, and the upper-class were prioritized. Using the information about these 891 passengers, the challenge is to build a model to predict which people would survive based on the following fields:
#
# - **Name** (str) - Name of the passenger
# - **Pclass** (int) - Ticket class
# - **Sex** (str) - Sex of the passenger
# - **Age** (float) - Age in years
# - **SibSp** (int) - Number of siblings and spouses aboard
# - **Parch** (int) - Number of parents and children aboard
# - **Ticket** (str) - Ticket number
# - **Fare** (float) - Passenger fare
# - **Cabin** (str) - Cabin number
# - **Embarked** (str) - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
#
# ![Clean Data](clean_data.png)

# ### Read in Data

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# %matplotlib inline

#%%
titanic = pd.read_csv('titanic.csv')
titanic.head()

#%%
# ### Clean continuous variables

# #### Fill missing for `Age`

titanic.isnull().sum()

titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic.head(10)

# #### Combine `SibSp` & `Parch`

for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )

titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']

# #### Drop unnnecessary variables

titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

titanic.head()

# ### Clean categorical variables

# #### Fill in missing & create indicator for `Cabin`

titanic.isnull().sum()

titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()

titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head()

# #### Convert `Sex` to numeric

# +
gender_num = {'male': 0, 'female': 1}

titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.head()
# -

# #### Drop unnecessary variables

titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
titanic.head()

# ### Write out cleaned data

titanic.to_csv('../../../titanic_cleaned.csv', index=False)


