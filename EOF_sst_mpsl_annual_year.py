

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
import matplotlib.cm as cm



#%% User Inputs

# path to data
#dpath = '../data/'

# min and max year
yrmin = 1980
yrmax = 2021

# toggle  plots on/off
plot_diag = False
output_plot = True


# toggle remove annual mean on/off
rem_ann = True
rem_trend = True

# number of PCs to run in model
num_PC = 21

# spatical domain
if False:
    #  Global domain
    lat1=-88
    lat2=88
    lon1=0
    lon2=359

if True:
    # Traditional PDO region
    lat1=20
    lat2=65
    lon1=120
    lon2=260
    
if False:
    # Expanded PDO region
    lat1=25
    lat2=70
    lon1=130
    lon2=250
    
if False:
    # Expanded PDO region
    lat1=0
    lat2=88
    lon1=0
    lon2=360
    
# central longitude for Robinson maps
cent_lon = -160.  

names = ['SST - Jan','550hPa Height - Jan',
         'SST - Feb','550hPa Height - Feb',
         'SST - Mar','550hPa Height - Mar',
         'SST - Apr','550hPa Height - Apr',
         'SST - May','550hPa Height - May',
         'SST - Jun','550hPa Height - Jun',
         'SST - Jul','550hPa Height - Jul',
         'SST - Aug','550hPa Height - Aug',
         'SST - Sep','550hPa Height - Sep',
         'SST - Oct','550hPa Height - Oct',
         'SST - Nov','550hPa Height - Nov',
         'SST - Dec','550hPa Height - Dec']

# names = ['MSLP - Jan','MSLP - Feb','MSLP - Mar','MSLP - Apr','MSLP - May',
# 'MSLP - Jun','MSLP - Jul','MSLP - Aug','MSLP - Sep','MSLP - Oct','MSLP - Nov','MSLP - Dec']

# names = ['SST - Jan',
#          'SST - Feb',
#          'SST - Mar',
#          'SST - Apr',
#          'SST - May',
#          'SST - Jun',
#          'SST - Jul',
#          'SST - Aug',
#          'SST - Sep',
#          'SST - Oct',
#          'SST - Nov',
#          'SST - Dec',]


#%% Import data

# open dataset
ds = xr.open_dataset('../data/era5_mslp_sst.nc') 

ds2 = xr.open_dataset('../data/era5_500hpa.nc')

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
#sp_raw = ds.msl
sp_raw = ds2.z

# print("starting regrid sst")
# sst = sst_raw.interp(longitude=new_lon, latitude=new_lat, method='linear')
# print("starting regrid mslp")
# sp = sp_raw.interp(longitude=new_lon, latitude=new_lat, method='linear')

print("starting regrid sst")
sst = sst_raw.coarsen(latitude=4,boundary='trim').mean().coarsen(longitude=4).mean()
print("starting regrid mslp")
sp = sp_raw.coarsen(latitude=4,boundary='trim').mean().coarsen(longitude=4).mean()
#sst=sst_raw
#sp=sp_raw


lat = sst.latitude.to_numpy()
lon = sst.longitude.to_numpy()
time = sst.time.to_numpy()

print("starting sst to numpy")
sst = xr.DataArray.to_numpy(sst)
print("starting mslp to numpy")
sp = xr.DataArray.to_numpy(sp)

# regrid
rfac = 4
i,a,b = np.shape(sst)

print("Loading complete")

#%% Move to anomally space

# sst = sst_np
# sp = sp_np

shape = np.shape(sst)

#%% Remove annual Cycle

# extract latitude and longitude
latx=len(lat) 
lonx=len(lon)

# extract shape of sstx
sts = np.shape(sst)
yrmx = int(sts[0]/12)

# Reshape
sst = np.reshape(sst,(yrmx,12,latx,lonx))
sp = np.reshape(sp,(yrmx,12,latx,lonx))


month_list = []

sst_glob = np.nanmean(sst,axis=(1,2,3))
sp_glob = np.nanmean(sp,axis=(1,2,3))

if rem_trend:
    for i in range(len(sst)):
        for j in range(12):
        
            sst[i,j,:,:] = sst[i,j,:,:]-sst_glob[i]
            sp[i,j,:,:] = sp[i,j,:,:]-sp_glob[i]
    
for m in range(12):

    month_list.append(sst[:,m,:,:])
    month_list.append(sp[:,m,:,:])

#%% Divide by standard dev, adjust for latitude

cn = np.sqrt(np.cos(lat*np.pi/180))

# Specify year range, print for user confirmation
year = np.linspace(1940,2023,2023-1940+1)


# filter
tfila = np.logical_and(year>=yrmin,year<=yrmax)
lonfila = np.logical_and(lon>=lon1,lon<=lon2)
latfila = np.logical_and(lat>=lat1,lat<=lat2)

y = time.astype('datetime64[Y]').astype(int) + 1970

list0 = []
lista = []

#%% Run through 
for i in range(len(month_list)):
    
    print(str(i)+" of 24")
    
    a = month_list[i]
    
    # remove point average
    amean = np.nanmean(a,axis=0)
    a = a-amean
    
    # filter by cosine
    for l in range(len(lat)):
        a[:,l,:] = a[:,l,:] * cn[i]
       
    # normalize
    a = a/np.nanstd(a)
    print("Normalized")
    
    # Select Subset for EOF Analysis
    list0.append(a[tfila,:,:])
    print("Filtered All")
    
    a = a[tfila,:,:]
    print("filtered Years")
    a = a[:,latfila,:]
    print("filtered lats")
    a = a[:,:,lonfila]
    print("filtered lons")
    
    # set mnx, which defines the number of timesteps
    mnx = len(a)
    
    lat_nan = lat[latfila]
    lon_nan = lon[lonfila]

    # first convert lon to -180 to 180 degrees format
    lon_sh_corr=[]
    for i in lon_nan:
        if i<180:
            lon_sh_corr.append(i)
        else:
            lon_sh_corr.append((360-i)*-1)
    
    # # now loop htrough and check each point with global land mask package
    icnt = 0
    jcnt = 0 
    landcnt=0
    for i in lat_nan:
        jcnt=0
        for j in lon_sh_corr:
            if ~globe.is_ocean(i,j):

                for k in range(mnx):
                    a[k,icnt,jcnt] = np.nan
                landcnt +=1
            jcnt+=1
        icnt+=1
    print(" Checked land mask")
    
    if plot_diag:
        # Plot
        plt.figure()
        plt.contourf(lon_nan,lat_nan,a[10,:,:]) 
        plt.colorbar()
        plt.show()
    
    a = np.reshape(a,(mnx,a.shape[1]*a.shape[2]))
    
    # start by computing a nan mask
    nan_mask = np.isnan(a[0,:])
    
    # now remove NaNs
    a = a[:,~nan_mask]
    print("removed nans")
    

    
    lista.append(a)
    

#%% Arange Data in a form useful for EOF Analysis 


# combine into one array

combx = lista[0]
for i in range(1,len(lista)):
    combx = np.concatenate((combx,lista[i]),axis=1)
#combx = ssta



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
years = np.linspace(yrmin,yrmax,yrmax-yrmin+1)
    
plt.errorbar(index[0:12],spectrum[0:12],yerror[0:12],capsize=5)
plt.ylabel('Fraction of Variance')
plt.xlabel('EOF number')
plt.title('Eigenvalue Spectrum - ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1)  )       
#  We need to construct the EOF map by regressing the pc onto the original data
pcmx=8  # were going to consider the first 4 eofs

ts = vh[0:pcmx,:]  # hope this is time series of first eof, YES looks right, has autocorrelation

for pci in range(0,pcmx):

    plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(years,ts[pci,:])
    plt.title('Timeseries of PC-' + str(pci+1) +' - ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
              + str(lat1)  )
    plt.grid()

#%% Regress onto original data

# Make empty regression vector
regm_list = []

# loop through each map
for i in range(len(list0)):
    
    sstb = list0[i]
    sstb=np.reshape(sstb,(mnx,latx*lonx))
    regm  = np.empty([pcmx+1,latx*lonx])

    # Loop through each PC
    for pci in range(0,pcmx):

        # extract the t vector for this PC
        t=ts[pci,:]

        # normalize predictor to have one standard deviation
        t = t/np.std(t)  
        
        # compute regression (t*ssb)/mnx
        reg = np.matmul(t,sstb)/mnx
        
    
        # save this regression
        regm[pci,:] = reg
    
    # reshape for plotting 
    regm = np.reshape(regm,(pcmx+1,latx,lonx))    
    
    regm_list.append(regm)
    
#%% Plot regressions (w/ subplots)

if output_plot:
    
    # set colormaps
    #col_mapc = 'RdYlBu_r'
    col_mapr = 'RdYlBu_r'
    col_mapr = 'RdBu'
    
    # set up lon
    lonp = np.empty(len(lon)+1)
    lonp[0:len(lon)] = lon
    lonp[len(lon)]= lon[len(lon)-1]+lon[1]-lon[0]
    
    # Loop through each PC
    rmax=0
    ip=0
    for i in range(len(regm_list)):
        regm = regm_list[i]
    
    
        # find appropreate regression for this PC
        regm1 = regm[ip,:,:]
        
        # here I am finding the maximum value, then I will fix the contours and colorbar to be constant across months
        if np.nanmax(np.abs(regm1)) >rmax:  # same contour interval for all plots
            rmax = np.nanmax(np.abs(regm1))
            print(rmax)
            
    rmax = 1.5
    
    nconts=60
    contr = np.linspace(-rmax,rmax,nconts+1)
    print(',  rmax ',rmax, ' - ',np.max(contr))

    # Loop through each PC
    for ip in range(0,pcmx):
        

        
        
        
        fig, ax = plt.subplots(ncols=2,nrows=12,dpi=200,figsize=(12,40), subplot_kw={'projection': ccrs.Robinson(central_longitude=cent_lon)})

        
        for i in range(len(regm_list)):
            
            print(i)
            
            regm = regm_list[i]
            
            row = int(np.floor(i/2))
            col = int(np.remainder(i,2))
    
        
            # find appropreate regression for this PC
            regm1 = regm[ip,:,:]
                    
            # toggle True/False to plot maps of regressions
            if True:
                
                # here we extend the array by one, so there is no gap in the contours
                prcpmp_sst = np.empty([latx,lonx+1])  
                prcpm_sst=regm1
                prcpmp_sst[:,0:len(lon)]= prcpm_sst
                prcpmp_sst[:,len(lon+1)]= prcpm_sst[:,0]
                
                # make figure
                #fig,ax = plt.subplots(figsize=(12, 4), dpi=100)
                #fig = plt.figure(figsize=(8, 5),dpi=200)
                
                #ax.append(fig.add_subplot(12, 2, i+1, projection=ccrs.Robinson(central_longitude=cent_lon)))
        
                # set up axes
               # ax = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))
        
                # Plot data
                cs = ax[row,col].contourf(lonp, lat, prcpmp_sst, contr, cmap=col_mapr,
                      transform=ccrs.PlateCarree())
                #cs.cmap.set_under('b')
                #cs.cmap.set_under('r')
        
                # add gridlines
                gl = ax[row,col].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1.5, color='gray', alpha=0.5, linestyle='-')
        
                # housekeeping
                ax[row,col].coastlines()
                fig.colorbar(cs)
                ax[row,col].title.set_text(names[i]+'( EOF ' + str(ip)+')')
        
        #fig.colorbar(cax=ax,ax = cbar_ax)
        fig.savefig('../figures/EOF'+str(ip)+'.png')
                

#%5 Save results

np.save('vh_'+str(yrmin)+'_'+str(yrmax) + '_' + str(lon1) + '_' + str(lon2) + '_' + str(lat2)+'.npy',vh)

#%% Plot regressions

# contour plot some regressions

# if output_plot:
    
#     # set colormaps
#     col_mapc = 'RdYlBu_r'
#     col_mapr = 'RdYlBu_r'
    
#     # set up lon
#     lonp = np.empty(len(lon)+1)
#     lonp[0:len(lon)] = lon
#     lonp[len(lon)]= lon[len(lon)-1]+lon[1]-lon[0]
    
#     # Loop through each PC
#     rmax=0
#     ip=0
        
#     for i in range(len(regm_list)):
#         regm = regm_list[i]
    
    
#         # find appropreate regression for this PC
#         regm1 = regm[ip,:,:]
        
#         # here I am finding the maximum value, then I will fix the contours and colorbar to be constant across months
#         if np.nanmax(np.abs(regm1)) >rmax:  # same contour interval for all plots
#             rmax = np.nanmax(np.abs(regm1))
    
    
#     nconts=60
#     contr = np.linspace(-rmax,rmax,nconts+1)
#     print(',  rmax ',rmax, ' - ',np.max(contr))
    
#     # Loop through each PC
#     for ip in range(0,pcmx):
        
#         for i in range(len(regm_list)):
            
#             regm = regm_list[i]
    
        
#             # find appropreate regression for this PC
#             regm1 = regm[ip,:,:]
        
#             # toggle True/False to plot maps of regressions
#             if True:
                
#                 # here we extend the array by one, so there is no gap in the contours
#                 prcpmp_sst = np.empty([latx,lonx+1])  
#                 prcpm_sst=regm1
#                 prcpmp_sst[:,0:len(lon)]= prcpm_sst
#                 prcpmp_sst[:,len(lon+1)]= prcpm_sst[:,0]
                
#                 # make figure
#                 #fig,ax = plt.subplots(figsize=(12, 4), dpi=100)
#                 fig = plt.figure(figsize=(8, 5),dpi=200)
                
#                 ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=cent_lon))
        
#                 # set up axes
#                # ax = plt.axes(projection=ccrs.Robinson(central_longitude=cent_lon))
        
#                 # Plot data
#                 ax.contourf(lonp, lat, prcpmp_sst, contr, cmap=col_mapr,
#                       transform=ccrs.PlateCarree())
        
#                 # add gridlines
#                 gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
#                           linewidth=1.5, color='gray', alpha=0.5, linestyle='-')
        
#                 # housekeeping
#                 ax.coastlines()
#                 #fig.colorbar(cax=ax)
#                 plt.title(names[i]+' Regression for EOF ' + str(ip+1) + ': ' + str(yrmin) + '-' + str(yrmax) + '  ' + str(lon1) + '-' + str(lon2) + '  ' + str(lat2) +'-' \
#                       + str(lat1))
                
#                 fig.savefig('../figures/EOF'+str(ip)+'_'+names[i]+'.png')
                

# #%5 Save results

np.save('SST_500hpa_vh_'+str(yrmin)+'_'+str(yrmax) + '_' + str(lon1) + '_' + str(lon2) + '_' + str(lat2)+'.npy',vh)
#np.save('SST_MSLP_vh_'+str(yrmin)+'_'+str(yrmax) + '_' + str(lon1) + '_' + str(lon2) + '_' + str(lat2)+'.npy',vh)
  
        
#%% Make Regression model

# Step 4: Define your response variable (time series you want to fit)
# Let's assume you have a pandas DataFrame 'response_df' containing the time series
rean = pd.read_csv('../data/era5_pwt.csv')
#fileter years
rean = rean[(rean['Year']>=yrmin) & (rean['Year']<=yrmax)]
# convert to numpy
rean_arr = rean['pwt_ann_500hpa'].to_numpy()
# remove trend
rean_vec = rean_arr - rean_arr.mean()
response_variable = rean_vec

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

PWT_model = []
for i in range(len(years)):

    dsum = coefficients[0]
    for j in range(num_PC):
        dsum+= coefficients[j+1] * predictors[i,j]

    PWT_model.append(dsum)



#%% Make plot of model results

fig,axs = plt.subplots(dpi=100)
axs.plot(rean_vec,PWT_model,'.')
axs.plot([-2, 2],[-2,2])
axs.set_xlabel('PWT Observed')
axs.set_ylabel('PWT Model')




