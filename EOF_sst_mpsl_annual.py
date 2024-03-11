#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:03:20 2024

@author: Liam
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.io import loadmat
from scipy.interpolate import griddata
import scipy.signal as sig
import xarray as xr
from cartopy import config
import cartopy.crs as ccrs
import pandas as pd
import netCDF4 as nc
from cartopy import config
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# land mask package
from global_land_mask import globe
import statsmodels.api as sm

#%% Load data

# path to data
#dpath = '../data/'

# min and max year
yrmin = 1940
yrmax = 2021

# toggle diagnostic plots on/off
plot_diag = True

# toggle remove annual mean on/off
rem_ann = True

# number of PCs to run in model
num_PC = 20

# spatical domain
if False:
    #  Global domain
    lat1=-88
    lat2=88
    lon1=0
    lon2=359

if False:
    # Traditional PDO region
    lat1=20
    lat2=65
    lon1=120
    lon2=260
    
if True:
    # Expanded PDO region
    lat1=30
    lat2=75
    lon1=120
    lon2=250
    
# central longitude for Robinson maps
cent_lon = -160.  

#%% Import data

# open dataset
ds = xr.open_dataset('../data/era5_mslp_sst.nc') 

# Define the new grid
new_lon = xr.DataArray(
    data=np.arange(0, 359, 1),  # 1-degree resolution in longitude
    dims=("lon",)
)
new_lat = xr.DataArray(
    data=np.arange(-90, 89, 1),  # 1-degree resolution in latitude
    dims=("lat",)
)

sst_raw = ds.sst
sp_raw = ds.msl

sst_ann = sst_raw.groupby('time.year').mean(dim='time')
sp_ann = sp_raw.groupby('time.year').mean(dim='time')

sst = sst_ann.interp(longitude=new_lon, latitude=new_lat, method='linear')
sp = sp_ann.interp(longitude=new_lon, latitude=new_lat, method='linear')

lat = sst.latitude.to_numpy()
lon = sst.longitude.to_numpy()

#%% Remove annual Cycle

# extract latitude and longitude
latx=len(lat) 
lonx=len(lon)

#%% Move to anomally space

sst_globmean = sst.mean(axis=0)
sp_globmean = sp.mean(axis=0)

for i in range(len(sst.year)):
    
    sst[i,:,:] = sst[i,:,:] - sst_globmean
    sp[i,:,:] = sp[i,:,:] - sp_globmean
    

    
#%% Remove annual anomaly

if rem_ann:

    sst_annmean = sst.mean(axis=(1,2))
    sp_annmean = sp.mean(axis=(1,2))
    
    for i in range(len(sst.year)):
        
        sst[i,:,:] = sst[i,:,:] - sst_annmean[i]
        sp[i,:,:] = sp[i,:,:] - sp_annmean[i]
        
#%% Divide by standard deviation

sst = sst/np.std(sst)
sp = sp/np.std(sp)

#%% Select Subset for EOF Analysis

# Specify year range, print for user confirmation
yr1= str(yrmin)+'-01-01'
yr2= str(yrmax)+'-12-31'
years = np.linspace(float(yrmin),float(yrmax+1),num = (yrmax-yrmin+1))
print('size years',np.shape(years),yr1,yr2)

# select all data

sst0 = sst[sst.year>=yrmin,:,:]
sst0 = sst0[sst0.year<=yrmax,:,:]

sp0 = sp[sp.year>=yrmin,:,:]
sp0 = sp0[sp0.year<=yrmax,:,:]

ssta = sst[sst.year>=yrmin,:,:]
ssta = ssta[ssta.year<=yrmax,:,:]
ssta = ssta[:,ssta.latitude>=lat1,:]
ssta = ssta[:,ssta.latitude<=lat2,:]
ssta = ssta[:,:,ssta.longitude>=lon1]
ssta = ssta[:,:,ssta.longitude<=lon2]

spa = sp[sp.year>=yrmin,:,:]
spa = spa[spa.year<=yrmax,:,:]
spa = spa[:,spa.latitude>=lat1,:]
spa = spa[:,spa.latitude<=lat2,:]
spa = spa[:,:,spa.longitude>=lon1]
spa = spa[:,:,spa.longitude<=lon2]

# make arrays of lat and lon
lat_sh=ssta.lat
lon_sh=ssta.lon

#%% Test Plot

if plot_diag:

    # pick arbitrary year
    i = 10

    # Plot
    plt.figure()
    plt.contourf(lon_sh,lat_sh,spa[i,:,:]) 
    plt.title('SP anomaly map: '+str(years[i]))
    plt.colorbar()
    
    # Plot
    plt.figure()
    plt.contourf(lon_sh,lat_sh,ssta[i,:,:]) 
    plt.title('SST anomaly map: '+str(years[i]))
    plt.colorbar()
    
#%% Arange Data in a form useful for EOF Analysis and apply land mask

# set mnx, which defines the number of timesteps
mnx = len(years)

# Convert to numpy
sstx = xr.DataArray.to_numpy(ssta)
spx = xr.DataArray.to_numpy(spa)

# convert into format for xarray
sstx = np.reshape(sstx,(mnx,sstx.shape[1]*sstx.shape[2]))
spx = np.reshape(spx,(mnx,spx.shape[1]*spx.shape[2]))

# start by computing a nan mask
nan_mask = np.isnan(sstx[0,:])

# now remove NaNs
sstx = sstx[:,~nan_mask]

# combine into one array
combx = np.concatenate((sstx,spx),axis=1)

# temporary reset of sstx
sstx = np.NaN

#%% Now we can check autocorrelation and estimate decrees of freedom

# Ok, now I think we have an array sstz that is [months,space]  Where only land regions are kept
# First let's estimate the number of degrees of freedom in the sample by estimating a grand autocorrelation
def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=np.nanmean(x)
    var=np.nanvar(x)
    if var == 0.0:
        var=1.0e-6
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    #corr=[ np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    return np.array(corr)
xcor = np.zeros([2])

nt, ns = np.shape(combx)
print('nt',nt,'ns',ns)
type(ns)

lags = [0,1]
for i in range(ns):
    xx=np.squeeze(combx[:,i])
    xcor1 = autocorr2(xx,lags)
    xcor = xcor + xcor1

print('xx',xx[0:4],xx[-4:])    
xcor = xcor/float(ns)
print('xcor ',xcor)
acorr = xcor[1]
dof_sst = mnx*(1. - acorr**2)/(1. + acorr**2)
print('DOF = ',dof_sst,',   DOF/mnx = ',dof_sst/mnx,',  mnx = ',mnx)

#%% EOF Analysis

mmax = np.max(combx)
u, s, vh = np.linalg.svd(combx.T,full_matrices=False)

#%% Plot of EOF results

print('u shape',np.shape(u))
print('vh shape',np.shape(vh))
print('s shape',np.shape(s))
type(u)
type(vh)
type(s)
spectrum = s*s.T
spectrum = spectrum/sum(spectrum)
plt.figure(figsize=(12, 4), dpi=100)
yerror = spectrum*np.sqrt(2/dof_sst)
index = np.linspace(0,24,25)
    
plt.errorbar(index[0:12],spectrum[0:12],yerror[0:12],capsize=5)
plt.ylabel('Fraction of Variance')
plt.xlabel('EOF number')
plt.title('Eigenvalue Spectrum - ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1)  )       
#  We need to construct the EOF map by regressing the pc onto the original data
pcmx=4  # were going to consider the first 4 eofs

ts = vh[0:pcmx,:]  # hope this is time series of first eof, YES looks right, has autocorrelation

for pci in range(0,pcmx):

    plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(years,ts[pci,:])
    plt.title('Timeseries of PC-' + str(pci+1) +' - ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1)  )
    plt.grid()

#%% Regress onto original data

# Make empty regression vector
regm_sst = np.empty([pcmx+1,latx*lonx])
regm_sp  = np.empty([pcmx+1,latx*lonx])

# Loop through each PC
for pci in range(0,pcmx):

    # extract the t vector for this PC
    t=ts[pci,:]

    # normalize predictor to have one standard deviation
    t = t/np.std(t)  

    # extract original data, and convert to 2D matrix
    sstb = xr.DataArray.to_numpy(sst0)
    sstb=np.reshape(sstb,(mnx,latx*lonx))
    spb = xr.DataArray.to_numpy(sp0)
    spb=np.reshape(spb,(mnx,latx*lonx))

    # compute regression (t*ssb)/mnx
    reg_sst = np.matmul(t,sstb)/mnx
    reg_sp = np.matmul(t,spb)/mnx

    # save this regression
    regm_sst[pci,:] = reg_sst
    regm_sp[pci,:] = reg_sp
    
# reshape for plotting 
regm_sst = np.reshape(regm_sst,(pcmx+1,latx,lonx))    
regm_sp  = np.reshape(regm_sp, (pcmx+1,latx,lonx))

#%% Plot regressions

# contour plot some regressions

# set colormaps
col_mapc = 'RdYlBu_r'
col_mapr = 'RdYlBu_r'

# set up lon
lonp = np.empty(len(lon)+1)
lonp[0:len(lon)] = lon
lonp[len(lon)]= lon[len(lon)-1]+lon[1]-lon[0]

# Loop through each PC
for ip in range(0,pcmx):

    # find appropreate regression for this PC
    regm1_sst = regm_sst[ip,:,:]
    regm1_sp  = regm_sp[ip,:,:]
    
    # here I am finding the maximum value, then I will fix the contours and colorbar to be constant across months
    if ip == 0:  # same contour interval for all plots
        rmax_sst = np.nanmax(np.abs(regm1_sst))
        rmax_sp = np.nanmax(np.abs(regm1_sp))
        rmax = max([rmax_sst,rmax_sp])
        nconts=60
        contr = np.linspace(-rmax,rmax,nconts+1)
        print(',  rmax ',rmax, ' - ',np.max(contr))

    # toggle True/False to plot maps of regressions
    if True:
        
        # here we extend the array by one, so there is no gap in the contours
        prcpmp_sst = np.empty([latx,lonx+1])  
        prcpm_sst=regm1_sst
        prcpmp_sst[:,0:len(lon)]= prcpm_sst
        prcpmp_sst[:,len(lon+1)]= prcpm_sst[:,0]
        prcpmp_sp = np.empty([latx,lonx+1])  
        prcpm_sp=regm1_sp
        prcpmp_sp[:,0:len(lon)]= prcpm_sp
        prcpmp_sp[:,len(lon+1)]= prcpm_sp[:,0]
        
        
        # # make plot
        # fix,ax = plt.subplots(2,1,figsize=(12,9),dpi=100)
        
        # # set up axes
        # ax[0] = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))
        # ax[1] = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))
        
        # # plot data
        # ax[0].contourf(lonp, lat, prcpmp_sst, contr, cmap=col_mapr,
        #              transform=ccrs.PlateCarree())
        # ax[1].contourf(lonp, lat, prcpmp_sp, contr, cmap=col_mapr,
        #              transform=ccrs.PlateCarree())
        
        # gl0 = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
        #           linewidth=1.5, color='gray', alpha=0.5, linestyle='-')
        # gl1 = ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
        #           linewidth=1.5, color='gray', alpha=0.5, linestyle='-')
        
        # ax[0].coastlines()
        # ax[1].coastlines()
        
        # make figure
        plt.figure(figsize=(12, 4), dpi=100)

        # set up axes
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))

        # Plot data
        plt.contourf(lonp, lat, prcpmp_sst, contr, cmap=col_mapr,
              transform=ccrs.PlateCarree())

        # add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1.5, color='gray', alpha=0.5, linestyle='-')

        # housekeeping
        ax.coastlines()
        plt.colorbar()
        plt.title('SST Regression for EOF ' + str(ip+1) + ' \N{DEGREE SIGN}C ' + str(yrmin) + '-' + str(yrmax) + '  ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1) )
        plt.show()
        
        # make figure
        plt.figure(figsize=(12, 4), dpi=100)

        # set up axes
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))

        # Plot data
        plt.contourf(lonp, lat, prcpmp_sp, contr, cmap=col_mapr,
              transform=ccrs.PlateCarree())

        # add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1.5, color='gray', alpha=0.5, linestyle='-')

        # housekeeping
        ax.coastlines()
        plt.colorbar()
        plt.title('SP Regression for EOF ' + str(ip+1) + ' \N{DEGREE SIGN}C ' + str(yrmin) + '-' + str(yrmax) + '  ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1) )
        plt.show()
        
#%% Make Regression model

# Step 4: Define your response variable (time series you want to fit)
# Let's assume you have a pandas DataFrame 'response_df' containing the time series
df = pd.read_csv('../data/all_annual.csv')
df = df[(df['Year']>=yrmin) & (df['Year']<=yrmax)]
response_variable = df['d18O'].to_numpy()

# Step 5: Define your predictors (principal components)
# Let's assume you have a numpy array 'principal_components' containing the principal components
predictors = vh[:num_PC,:].T  # Selecting the first 10 principal components
#ts[pci,:]

# Add a constant to the predictors matrix for the intercept term
predictors_with_constant = sm.add_constant(predictors)

# Step 5 (continued): Fit the regression model
model = sm.OLS(response_variable, predictors_with_constant)
results = model.fit()

# Step 5 (continued): Print regression results
print(results.summary())


coefficients = results.params

d18O_model = []
for i in range(len(years)):

    dsum = coefficients[0]
    for j in range(num_PC):
        dsum+=coefficients[j] * predictors[i,j]

    d18O_model.append(dsum)


#%% Make plot of model results

fig,axs = plt.subplots(dpi=100)
axs.plot(df['d18O'],d18O_model,'.')
axs.plot([-36,-25],[-36,-25])
axs.set_xlabel('d18O Observed')
axs.set_ylabel('d18O Model')
