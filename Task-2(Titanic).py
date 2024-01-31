import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/content/train.csv')
data.head(25)

data.tail(25)

data.info()

data.describe()

import seaborn as sns
plt.figure(figsize=(6,4))
sns.countplot(x='Survived',data=data,color='red')
plt.title('Distribution of Survival')
plt.show()


counts=data.groupby(['Pclass','Survived']).size().unstack()
counts.plot(kind='hist')


surv_of_diff_age=data.groupby('Age')['Survived'].mean()
print(surv_of_diff_age)

plt.bar(surv_of_diff_age.index,surv_of_diff_age.values,color='green')
plt.xlabel('Ages')
plt.ylabel('Survived')
plt.title('Survival rate of different ages of people')

sns.barplot(x='Sex',y='Survived',data=data,color='Blue')

sns.barplot(x='Pclass',y='Survived',data=data,color='yellow')
