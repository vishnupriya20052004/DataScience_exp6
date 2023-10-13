# Feature_Transformation
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:
![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/df7d2928-63bc-4806-bbbc-cf8a78fd0cbb)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/a0841636-3131-4d5a-ad5b-62e608a6e075)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/87af985e-5858-4135-ab4b-00774933b564)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/31cfc53c-3de8-4199-86ee-3279546ee33d)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/195d62d2-7f35-49ba-9594-8b018caa76d8)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/3e250d9f-782c-4601-a59f-9a8c473925c3)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/4a50de36-fd2d-4709-8dc8-9000af2740d3)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/fbc64f79-ad41-4547-91cd-95615bc4015f)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/a9031d55-df4c-4b5d-bb61-23bb1498124f)


![image](https://github.com/vishnupriya20052004/DataScience_exp6/assets/133640291/63dcce65-3c69-4380-b550-fa427a7dc545)


# RESULT:
Thus feature transformation is done for the given dataset.
