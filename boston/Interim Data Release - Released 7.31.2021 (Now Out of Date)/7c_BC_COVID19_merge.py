# -*- coding: utf-8 -*-
"""
Merge data from daily surveys, demographics, and on time assessments

BC COVID-19 data
Cunningham, T. J., Fields, E. C., & Kensinger, E. A. (2021). Boston College daily 
sleep and well-being survey data during early phase of the COVID-19 pandemic. 
Scientific Data, 8(110). https://doi.org/10.1038/s41597-021-00886-y

Author: Eric Fields
Version Date: 28 July 2021

Copyright (c) 2021, Eric Fields
This code is free and open source software made available under the 3-clause BSD license
https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import pandas as pd

from BC_COVID19_import_data import import_covid19_data
from BC_COVID19_demo_relabel import relabel_demo


def time_bin_data(data, time_bins, agg_fun=np.nanmean):
    """
    Average across given time bins for each subject from daily data
    """
    
    #Get time bin averaged data
    binned_data = pd.DataFrame()
    for time_bin in time_bins:
        
        #Get rows in date range
        start = pd.to_datetime(time_bins[time_bin][0], yearfirst=True)
        end = pd.to_datetime(time_bins[time_bin][1], yearfirst=True)
        date_idx = data['ref_date'].between(start, end)
        
        #Get rows for short and full survey
        short_idx = np.array(['S' in x for x in data.index.to_list()])
        full_idx = np.array(['L' in x for x in data.index.to_list()])
        assert not any(np.isnan(short_idx))
        assert not any(np.isnan(full_idx))
        assert sum(short_idx | full_idx) == len(data)
        assert not any(short_idx & full_idx)
        
        #Get the mean for this time_bin
        range_mean = data[date_idx].groupby('sub_id').aggregate(agg_fun)
        range_mean.insert(0, 'time_bin', time_bin)
        
        #Get the number of responses for short and full survey for each subject
        #in time bin
        range_mean.insert(1, 'full_N', np.nan)
        range_mean.insert(2, 'short_N', np.nan)
        short_counts = data.loc[date_idx & short_idx, 'sub_id'].value_counts()
        range_mean.loc[short_counts.index, 'short_N'] = short_counts.values
        full_counts = data.loc[date_idx & full_idx, 'sub_id'].value_counts()
        range_mean.loc[full_counts.index, 'full_N'] = full_counts.values
        range_mean['full_N'].replace({np.nan: 0}, inplace=True)
        range_mean['short_N'].replace({np.nan: 0}, inplace=True)
        assert range_mean[['short_N', 'full_N']].sum().sum() == sum(date_idx)
        
        #Add this time bin to the binned data
        binned_data = binned_data.append(range_mean)
    
    #Get rid of meaningless and empty columns
    binned_data.drop(['record_id', 'redcap_repeat_instance'], axis='columns', inplace=True)
    binned_data = binned_data.loc[:, binned_data.notna().any()]
    
    #Drop subjects with no data in the specified time bins
    binned_data = binned_data.loc[binned_data.notna().any(axis=1), :]
    
    return binned_data


def merge_covid_data(date_str, data_dir, output_file, daily_vars, demo_vars,
                     r1_vars, r2_vars, r3_vars, r4_vars, r5_vars, r6_vars,
                     time_bins=None, agg_fun=np.nanmean, demo_relabel=False):
    """
    Merge subset of variables from daily data, demographics, and one time assessments
    and output to csv file
    """
    
    #Make sure sub_id is included for all data sources
    if 'sub_id' not in daily_vars: daily_vars.insert(0, 'sub_id')
    if 'sub_id' not in demo_vars: demo_vars.insert(0, 'sub_id')
    if 'sub_id' not in r1_vars: r1_vars.insert(0, 'sub_id')
    if 'sub_id' not in r2_vars: r2_vars.insert(0, 'sub_id')
    if 'sub_id' not in r3_vars: r3_vars.insert(0, 'sub_id')
    if 'sub_id' not in r4_vars: r4_vars.insert(0, 'sub_id')
    if 'sub_id' not in r5_vars: r5_vars.insert(0, 'sub_id')
    if 'sub_id' not in r6_vars: r6_vars.insert(0, 'sub_id')
    
    #For daily data include either ref_dat or time_bin in merged data
    if len(daily_vars)==1 and daily_vars[0]=='sub_id':
        pass
    else:
        if time_bins:
            if 'time_bin' not in daily_vars: daily_vars.insert(1, 'time_bin')
            if 'full_N' not in daily_vars: daily_vars.insert(2, 'full_N')
            if 'short_N' not in daily_vars: daily_vars.insert(3, 'short_N')
        else:
            if 'ref_date' not in daily_vars: daily_vars.insert(1, 'ref_date')
    
    #Import data
    (data, demo, r1, r2, r3, r4, r5, r6) = import_covid19_data(data_dir, date_str)
    
    #Make labels in demographics more informative if requested
    if demo_relabel:
        demo = relabel_demo(demo)
    
    #Bin data by date if requested
    if time_bins:
        data = time_bin_data(data, time_bins, agg_fun=agg_fun).reset_index()
    elif len(daily_vars)==1 and daily_vars[0]=='sub_id':
        data = time_bin_data(data, 
                             {'all': (data['ref_date'].min(), data['ref_date'].max())},
                             agg_fun=agg_fun).reset_index()
    
    #Deal with repeated variable names
    all_vars = daily_vars + demo_vars + r1_vars + r2_vars + r3_vars + r4_vars + r5_vars + r6_vars
    demo_rep = dict([(var, 'demo_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in demo_vars])
    demo.rename(columns=demo_rep, inplace=True)
    demo_vars = list(demo_rep.values())
    r1_rep = dict([(var, 'r1_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r1_vars])
    r1.rename(columns=r1_rep, inplace=True)
    r1_vars = list(r1_rep.values())
    r2_rep = dict([(var, 'r2_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r2_vars])
    r2.rename(columns=r2_rep, inplace=True)
    r2_vars = list(r2_rep.values())
    r3_rep = dict([(var, 'r3_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r3_vars])
    r3.rename(columns=r3_rep, inplace=True)
    r3_vars = list(r3_rep.values())
    r4_rep = dict([(var, 'r4_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r4_vars])
    r4.rename(columns=r4_rep, inplace=True)
    r4_vars = list(r4_rep.values())
    r5_rep = dict([(var, 'r5_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r5_vars])
    r5.rename(columns=r5_rep, inplace=True)
    r5_vars = list(r5_rep.values())
    r6_rep = dict([(var, 'r6_'+var) if all_vars.count(var)>1 and var!='sub_id' else (var, var) for var in r6_vars])
    r6.rename(columns=r6_rep, inplace=True)
    r6_vars = list(r6_rep.values())
    
    #Merge all data sources
    merged_data = data[daily_vars].merge(demo[demo_vars], how='outer', on='sub_id').merge(r1[r1_vars], how='outer', on='sub_id').merge(r2[r2_vars], how='outer', on='sub_id').merge(r3[r3_vars], how='outer', on='sub_id').merge(r4[r4_vars], how='outer', on='sub_id').merge(r5[r5_vars], how='outer', on='sub_id').merge(r6[r6_vars], how='outer', on='sub_id')
    
    #Drop subjects with no data
    na_idx = merged_data.drop('sub_id', axis='columns').isna().all(axis=1)
    merged_data = merged_data[~na_idx]
    
    #Sort by sub_id
    merged_data.sort_values('sub_id', inplace=True)
    
    #Reset index
    merged_data.reset_index(inplace=True, drop=True)
    
    #Output to csv
    if output_file:
        merged_data.to_csv(output_file, index=False)
    
    return merged_data



#%%###########################################################################
################################  SET-UP  ####################################
##############################################################################

if __name__ == '__main__':

    #~~~~~ Change parameters below as necessary ~~~~~
    
    #Date string for cleaned data (found at the end of each csv file name)
    date_str = '2021-07-22_13_26'
    
    #Directory with COVID-19 csv data files
    data_dir = r'D:\COVID19\export'
    
    #Full path and filename for output csv file with merged data
    output_file = False #r'C:\Users\ecfne\Desktop\test.csv'
    
    #Time bins to average across in daily data; for un-binned data time bins = None
    #NOTE: Dates must be in year/month/day format
    time_bins = {'early': ('2020/3/21', '2020/4/30'), 
                  'late': ('2020/5/1', '2020/5/31')}
    
    #The function to use to aggregate across observations in each time bin
    #Usually the mean (nanmean ignores mising values), but any mathematical function can be used
    #This argument is ignored if there are no time bins (i.e., data is kept daily)
    agg_fun = np.nanmean
    
    #Variables to include from each source
    daily_vars = ['stress', 'PANAS_PA']
    demo_vars = ['age1', 'preferred_gender']
    r1_vars   = []
    r2_vars   = []
    r3_vars   = []
    r4_vars   = ['political']
    r5_vars   = []
    r6_vars   = []
    
    ##############################################################################
    ##############################################################################
    ##############################################################################

    merged_data = merge_covid_data(date_str, data_dir, output_file, daily_vars, 
                                   demo_vars, r1_vars, r2_vars, r3_vars, r4_vars, 
                                   r5_vars, r6_vars, time_bins, agg_fun)
