#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:44:30 2024

@author: Liam

"""

#%% Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#%% User Inputs

yrmin = 1979
yrmax = 2021

if True:
    # Traditional PDO region
    lat1=20
    lat2=65
    lon1=120
    lon2=260
    

    

#%% Load Data

vh = np.load('vh_'+str(yrmin)+'_'+str(yrmax) + '_' + str(lon1) + '_' + str(lon2) + '_' + str(lat2)+'.npy')



# Step 4: Define your response variable (time series you want to fit)
# Let's assume you have a pandas DataFrame 'responsed_df' containing the time series
rean = pd.read_csv('../data/era5_pwt.csv')
#fileter years
rean = rean[(rean['Year']>=yrmin) & (rean['Year']<=yrmax)]
# convert to numpy
rean_arr = rean['pwt_ann_500hpa'].to_numpy()
# remove trend
rean_vec = rean_arr - rean_arr.mean()
response_variable = rean_vec

#%% Decide what num_PC shoudl be

pc = list(range(5,25))
r_sqr_pc = []
r_sqr_adj_pc = []
rmse_pc = []
for num_PC in range(5,25):
    
    predictors = vh[:num_PC,:].T
    
    # Add a constant to the predictors matrix for the intercept term
    predictors_with_constant = sm.add_constant(predictors)
    
    model = sm.OLS(response_variable, predictors_with_constant)
    results = model.fit()
    
    coefficients = results.params
    PWT_model = []
    for i in range(len(rean_vec)):

        dsum = coefficients[0]
        for j in range(num_PC):
            dsum+= coefficients[j+1] * predictors[i,j]

        PWT_model.append(dsum)
        
    #rmse_pc.append(mean_squared_error(rean_vec,PWT_model))
    r_sqr_pc.append(results.rsquared)
    r_sqr_adj_pc.append(results.rsquared_adj)
    
fig,ax = plt.subplots()
ax.set_title('Neccisary PC Number Determination')
ax.plot(pc,r_sqr_pc,label='R Squared')
ax.plot(pc,r_sqr_adj_pc,label='R Squared (adjusted)')
#ax.plot(pc,rmse_pc,label='RMSE')
ax.set_xticklabels(pc)
ax.set_xticks(np.arange(min(pc),max(pc),1.0))
plt.grid()
ax.legend()
ax.set_xlabel('Number of PCs Used in Regression')

# include the two chosen number of PCs
ax.plot(pc[9],r_sqr_pc[9],'r.')
ax.plot(pc[9],r_sqr_adj_pc[9],'r.')
#ax.plot(pc[9],rmse_pc[9],'r.')
ax.plot(pc[16],r_sqr_pc[16],'r.')
ax.plot(pc[16],r_sqr_adj_pc[16],'r.')
#ax.plot(pc[16],rmse_pc[16],'r.')


# get num_PC
num_PC = pc[r_sqr_adj_pc.index(max(r_sqr_adj_pc))]
print("Using "+str(num_PC)+" PCs")

num_PC = 14


#%% Run all

predictors = vh[:num_PC,:].T
    
# Add a constant to the predictors matrix for the intercept term
predictors_with_constant = sm.add_constant(predictors)

model = sm.OLS(response_variable, predictors_with_constant)
results = model.fit()

coefficients = results.params
PWT_model_all = []
for i in range(len(rean_vec)):

    dsum = coefficients[0]
    for j in range(num_PC):
        dsum+= coefficients[j+1] * predictors[i,j]

    PWT_model_all.append(dsum)
    


#%% Leave one out analysis

#num_PC = 20

fig2,ax2 = plt.subplots(figsize=(8,6))


PC_leavout = list(range(num_PC))
vh_num= vh[:num_PC,:].T
r_sqr = []
rmse = []
for i in range(num_PC):
    vec = list(range(num_PC))
    vec = vec[vec!=i]
    
    predictors = vh_num
    predictors = np.delete(predictors,i,1)
    
    # Add a constant to the predictors matrix for the intercept term
    predictors_with_constant = sm.add_constant(predictors)
    
    model = sm.OLS(response_variable, predictors_with_constant)
    results = model.fit()
    
    coefficients = results.params

    
    r_sqr.append(results.rsquared)
    
    PWT_model = []
    for r in range(len(rean_vec)):

        dsum = coefficients[0]
        for j in range(num_PC-1):
            dsum+= coefficients[j+1] * predictors[r,j]

        PWT_model.append(dsum)
        
    ax2.plot(np.linspace(1979,2021,2021-1979+1),PWT_model-rean_vec,label='PC #'+str(i))

    
    rmse.append(mean_squared_error(rean_vec,PWT_model))
    
    #print(results.summary())

ax2.plot(np.linspace(1979,2021,2021-1979+1),PWT_model_all-rean_vec,'k',linewidth=2,label='All')
ax2.legend(ncol=2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Error Relative to Reanalysis PWT')
ax2.set_title('Leave-One-Out Analysis: Error')
plt.grid()

fig,ax = plt.subplots()
ax.set_title('Leave-One-Out Analysis: R-squared')
ax2 = ax.twinx()
ax.plot(PC_leavout,r_sqr,'k-')
#ax2.plot(PC_leavout,rmse,'r--')
plt.gca().invert_yaxis()
ax.set_xlabel('Left-Out PC')
ax.set_ylabel('R-Squared')

# get num_PC
weakest = PC_leavout[r_sqr.index(min(r_sqr))]
print("Leavout highlists PC #"+str(weakest))


