import laspy
import numpy as np
import pandas as pd
df=pd.read_csv('DSM_cal.csv')
print(df['avg_height_default'])
print(df['avg_height_single'])
print(df['z_rtk'])
RMSE1_default=0
RMSE1_single=0
RMSE2_default=0
RMSE2_single=0
temp_default=[]
temp_single=[]
n=0
m=0
for i in range(len(df['x'])):
    if df['avg_height_default'][i]!=-1:
        n=n+1
        temp_default.append((df['avg_height_default'][i]-df['z_rtk'][i])**2)
    if df['avg_height_single'][i]!=-1:
        m=m+1
        temp_single.append((df['avg_height_single'][i]-df['z_rtk'][i])**2)
RMSE1_default=np.sqrt((np.sum(temp_default)/n))
RMSE1_single=np.sqrt((np.sum(temp_single)/m))
# n=0
# temp_default=[]
# temp_single=[]
# for i in range(len(f['x'])):
#     if df['avg_height_single'][i]!=-1 and df['avg_height_deault']!=-1:
#         temp_default.append(df[''])
print(RMSE1_default)
print(RMSE1_single)