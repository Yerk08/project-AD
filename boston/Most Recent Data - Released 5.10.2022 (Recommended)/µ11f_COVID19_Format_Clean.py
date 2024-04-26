# -*- coding: utf-8 -*-
"""
Import, format, and check COVID-19 data

BC COVID-19 data
Cunningham, T. J., Fields, E. C., & Kensinger, E. A. (2021). Boston College daily 
sleep and well-being survey data during early phase of the COVID-19 pandemic. 
Scientific Data, 8(110). https://doi.org/10.1038/s41597-021-00886-y

Author: Eric Fields
Version Date: 6 February 2022

Copyright (c) 2021, Eric Fields
This code is free and open source software made available under the 3-clause BSD license
https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os
from os.path import join
import re

import numpy as np
import pandas as pd

main_dir = r'D:\COVID19'
os.chdir(main_dir)

sys.path.append(join(main_dir, 'code'))
from COVID19_QC import (daily_QC, demo_QC, R1_QC, R2_QC, R3_QC, R4_QC, R5_QC, 
                        R6_QC, R7_QC, R8_QC, R9_QC)


#Indicate whether to output files when the script is run
output = True


def timedelta2str(td):
    """
    Convert timedelta data being used to represent clock time to string in MM:SS format
    """
    if pd.isna(td):
        return ''
    else:
        assert td >= pd.Timedelta(0)
        assert td <= pd.Timedelta(24, unit='hours')
        return str(td.round('min')).split()[-1][0:5]


def df2csv(filename, df):
    """
    Write data frame to csv with dates and timedelta objects sensibly formatted
    """
    
    #Make copy before modifying
    out_data = df.copy()
    
    #Find Timedelta variables
    time_vars = out_data.dtypes[out_data.dtypes == np.dtype('<m8[ns]')].index.to_list()
    #Convert timedelta variables to string
    out_data[time_vars] = out_data[time_vars].applymap(timedelta2str)
    
    #Find dates without time
    date_vars = out_data.dtypes[out_data.dtypes == np.dtype('<M8[ns]')].index.to_list()
    #Convert to date only string
    for col in date_vars:
        if all(((out_data[col] - out_data[col].dt.normalize()) == pd.Timedelta(0)) | out_data[col].isna()):
            out_data[col] = out_data[col].apply(lambda x: '' if pd.isna(x) else str(x)[0:10])
    
    #output to csv
    out_data.to_csv(filename)


def sub_date_duplicates(data, keep=False):
    """
    Return boolean with rows of subject-date duplicates
    """
    sub_date = data[['sub_id', 'todays_date', 'redcap_timestamp']].copy()
    sub_date['todays_date'] = sub_date['todays_date'].dt.normalize()
    sub_date['redcap_timestamp'] = sub_date['redcap_timestamp'].dt.normalize()
    idx = sub_date.duplicated(['sub_id', 'todays_date'], keep=keep)
    #idx = idx | sub_date.duplicated(['sub_id', 'redcap_timestamp'], keep=keep)
    return idx


def calc_sleep_time(bed_time, awake_time, correct_12=False, min_12=0):
    """
    Calculate sleep time from bed_time and awake_time
    """
    
    #Confirm that times are in the right format and range
    if not all(bed_time.between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) | bed_time.isna()):
        raise ValueError('All bed times must be between 00:00 and 23:59')
    if not all(awake_time.between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) | awake_time.isna()):
        raise ValueError('All awake times must be between 00:00 and 23:59')
    
    #Get sleep time assuming 24 hour clock
    st = awake_time - bed_time
    #Convert to hours
    st = st.apply(lambda x: x.total_seconds()/3600)
    #Correct for circularity of time
    st[st<0] += 24
    
    #Correct for 12-hour clock usage
    if correct_12:
        idx = ((awake_time < bed_time)
               & (awake_time < pd.Timedelta(13, unit='hours'))
               & (bed_time < pd.Timedelta(13, unit='hours'))
               & (awake_time >= pd.Timedelta(1, unit='hours'))
               & (bed_time >= pd.Timedelta(1, unit='hours')))
        st[idx] += -12
        corr12 = pd.Series(index=st.index, dtype=int)
        corr12[idx] = 1
    
    #Check for impossible values        
    assert all(st.between(0, 24) | st.isna())
    
    #For people using 12 hour clock don't try to distinguish values close to 0 from values just over 12
    if min_12:
        idx = ((st < min_12)
               & (awake_time < pd.Timedelta(13, unit='hours'))
               & (bed_time < pd.Timedelta(13, unit='hours'))
               & (awake_time >= pd.Timedelta(1, unit='hours'))
               & (bed_time >= pd.Timedelta(1, unit='hours')))
        st[idx] = np.nan
    
    if correct_12:
        return(st, corr12)
    else:
        return st


def calc_sleep_midpoint(bed_time, awake_time, correct_12=False, min_12=0):
    """
    Calculate sleep midpoint from bed time and waking time
    """
    
    #Calculate sleep duration and find uses of 12 hour clock
    (sleep_duration, corr12) = calc_sleep_time(bed_time, awake_time, correct_12=correct_12, min_12=min_12)
    corr12 = corr12.apply(bool)
    
    #Calculate sleep midpoint
    sleep_midpoint = bed_time + sleep_duration.apply(pd.Timedelta, unit='hours')/2
    #Deal with circular nature of time
    idx = (sleep_midpoint >= pd.Timedelta(24, unit='hours')) & ~corr12
    sleep_midpoint[idx] += -pd.Timedelta(24, unit='hours')
    idx = (sleep_midpoint >= pd.Timedelta(13, unit='hours')) & corr12
    sleep_midpoint[idx] += -pd.Timedelta(12, unit='hours')
    
    #Round to nearest minute
    sleep_midpoint = sleep_midpoint.round('min')
    
    #Check that all sleep midpoints are valid clock times
    assert all(sleep_midpoint.between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) | sleep_midpoint.isna())
    
    return sleep_midpoint



#%% IMPORT AND MERGE

#Import raw data
df_short = pd.read_csv(join(main_dir, 'raw_data', 'COVIDLongitudinalDat_DATA_2022-01-06_1603.csv'))
df_long  = pd.read_csv(join(main_dir, 'raw_data', 'COVID19LongitudinalD_DATA_2022-01-06_1602.csv'))
demo     = pd.read_csv(join(main_dir, 'raw_data', 'COVID19-DEMOREPORT_DATA_2022-01-06_1602.csv'))
r1 = pd.read_csv(join(main_dir, 'raw_data', 'Round1', 'Round1COVIDAdditiona_DATA_2022-01-06_1605.csv'))
r2 = pd.read_csv(join(main_dir, 'raw_data', 'Round2', 'Round2COVIDAdditiona_DATA_2022-01-06_1605.csv'))
r3 = pd.read_csv(join(main_dir, 'raw_data', 'Round3', 'Round3COVIDAdditiona_DATA_2022-01-06_1606.csv'))
r4 = pd.read_csv(join(main_dir, 'raw_data', 'Round4', 'Round4COVIDAdditiona_DATA_2022-01-06_1606.csv'))
r5 = pd.read_csv(join(main_dir, 'raw_data', 'Round5', 'Round5COVIDAdditiona_DATA_2022-01-06_1606.csv'))
april18_data = pd.read_csv(join(main_dir, 'raw_data', 'Round6', 'April18_DATA_2022-01-06_1607.csv'))
r7_A = pd.read_csv(join(main_dir, 'raw_data', 'Round7', 'COVID_vaccine_longitudinal_retrospective_January 6, 2022_09.09.csv'))
r7_B = pd.read_csv(join(main_dir, 'raw_data', 'Round7', 'COVID_vaccine_longitudinal_retrospective_UNVAXX_January 6, 2022_09.12.csv'))
r8 = pd.read_csv(join(main_dir, 'raw_data', 'Round8', 'Round8COVIDAdditiona_DATA_2022-01-06_1612.csv'))
nov15_data = pd.read_csv(join(main_dir, 'raw_data', 'Round9', 'November15_DATA_2022-01-06_1614.csv'))


#Extract Round 6 data from the April 18 data
r6_vars = list(april18_data.loc[:, 'telephoneapril_fear': 'perished_2'].columns)
r6 = april18_data[['record_id', 'april_18_timestamp', 'subjid_2', 'todays_date', 
                   *r6_vars, 'april_18_complete']].copy()

#Get daily data from April 18 data
april18_daily = april18_data[[col for col in april18_data.columns if col not in r6_vars]].copy()

#Merge R7 data
r7 = r7_B.append(r7_A)
del r7_A, r7_B
r7.drop([0, 1], inplace=True)

#Extract Round 6 data from the April 18 data
r9_vars = list(nov15_data.loc[:, 'est_us':'vacc_plan'].columns)
r9 = nov15_data[['record_id', 'nov15_timestamp', 'subjid_2', 'todays_date', 
                 *r9_vars, 'nov15_complete']].copy()

#Get daily data from April 18 data
nov15_daily = nov15_data[[col for col in nov15_data.columns if col not in r9_vars]].copy()
                      
#Merge long and short surveys
data = df_long.merge(df_short, how='outer')
data = pd.concat((data, april18_daily, nov15_daily))

del df_short, df_long, april18_data, april18_daily, nov15_data, nov15_daily



#%% SUBJECT CORRECTIONS AND EXCLUSIONS

#Remove rows with missing sub ID and todays_date
print('Dropping %d rows with missing subjid_2 or todays_date'
      % sum(data['subjid_2'].isna() | data['todays_date'].isna()))
data.dropna(axis=0, subset=['subjid_2', 'todays_date'], inplace=True)
print('Dropping %d rows with missing subjid_1'
      % demo['subjid_1'].isna().sum())
demo.dropna(axis=0, subset=['subjid_1'], inplace=True)
print('Dropping %d rows with missing round 1 subject ID'
      % r1['subjid_rd1'].isna().sum())
r1.dropna(axis=0, subset=['subjid_rd1'], inplace=True)
print('Dropping %d rows with missing round 2 subject ID'
      % r2['subjid_rd2'].isna().sum())
r2.dropna(axis=0, subset=['subjid_rd2'], inplace=True)
print('Dropping %d rows with missing round 3 subject ID'
      % r3['subjid_rd3'].isna().sum())
r3.dropna(axis=0, subset=['subjid_rd3'], inplace=True)
print('Dropping %d rows with missing round 4 subject ID'
      % r4['subjid_rd4'].isna().sum())
r4.dropna(axis=0, subset=['subjid_rd4'], inplace=True)
print('Dropping %d rows with missing round 5 subject ID'
      % r5['subjid_rd1'].isna().sum())
r5.dropna(axis=0, subset=['subjid_rd1'], inplace=True)
print('Dropping %d rows with missing round 6 subject ID'
      % r6['subjid_2'].isna().sum())
r6.dropna(axis=0, subset=['subjid_2'], inplace=True)
print('Dropping %d rows with missing round 7 subject ID'
      % r7['SubjectID'].isna().sum())
r7.dropna(axis=0, subset=['SubjectID'], inplace=True)
print('Dropping %d rows with missing round 8 subject ID'
      % r8['subjid_rd1'].isna().sum())
r8.dropna(axis=0, subset=['subjid_rd1'], inplace=True)
print('Dropping %d rows with missing round 9 subject ID'
      % r9['subjid_2'].isna().sum())
r9.dropna(axis=0, subset=['subjid_2'], inplace=True)

#Remove whitespace from subject IDs and convert all to uppercase
data['subjid_2'] = data['subjid_2'].astype(str).str.strip().str.upper()
demo['subjid_1'] = demo['subjid_1'].astype(str).str.strip().str.upper()
r1['subjid_rd1'] = r1['subjid_rd1'].astype(str).str.strip().str.upper()
r2['subjid_rd2'] = r2['subjid_rd2'].astype(str).str.strip().str.upper()
r3['subjid_rd3'] = r3['subjid_rd3'].astype(str).str.strip().str.upper()
r4['subjid_rd4'] = r4['subjid_rd4'].astype(str).str.strip().str.upper()
r5['subjid_rd1'] = r5['subjid_rd1'].astype(str).str.strip().str.upper()
r6['subjid_2']   = r6['subjid_2'].astype(str).str.strip().str.upper()
r7['SubjectID']  = r7['SubjectID'].astype(str).str.strip().str.upper()
r8['subjid_rd1'] = r8['subjid_rd1'].astype(str).str.strip().str.upper()
r9['subjid_2']   = r9['subjid_2'].astype(str).str.strip().str.upper()

#Fix LSAS response for WSKA2
r6_LSAS_vars = [x for x in r6.columns if x.endswith('_fear') or x.endswith('_avoid')]
r6.loc[r6['subjid_2']=='WSKA2', r6_LSAS_vars] = r6.loc[r6['subjid_2']=='WSKA2_LSAS', r6_LSAS_vars]
r6 = r6[r6['subjid_2'] != 'WSKA2_LSAS']
data = data[data['subjid_2'] != 'WSKA2_LSAS']

#These subjects are under 18 and were included by mistake 
#or the sub_id was assigned twice
excl_subs = ['54DLL', 'A2YXX', '7QU6Y', 'PMMTT', 'QMP33']
data = data[~data['subjid_2'].isin(excl_subs)]
demo = demo[~demo['subjid_1'].isin(excl_subs)]
r1 = r1[~r1['subjid_rd1'].isin(excl_subs)]
r2 = r2[~r2['subjid_rd2'].isin(excl_subs)]
r3 = r3[~r3['subjid_rd3'].isin(excl_subs)]
r4 = r4[~r4['subjid_rd4'].isin(excl_subs)]
r5 = r5[~r5['subjid_rd1'].isin(excl_subs)]
r6 = r6[~r6['subjid_2'].isin(excl_subs)]
r7 = r7[~r7['SubjectID'].isin(excl_subs)]
r8 = r8[~r8['subjid_rd1'].isin(excl_subs)]
r9 = r9[~r9['subjid_2'].isin(excl_subs)]
assert demo['age1'].min() >= 18

#Remove non-printing space from sub IDs
data['subjid_2'] = data['subjid_2'].str.replace('\u200b', '')
demo['subjid_1'] = demo['subjid_1'].str.replace('\u200b', '')
r1['subjid_rd1'] = r1['subjid_rd1'].str.replace('\u200b', '')
r2['subjid_rd2'] = r2['subjid_rd2'].str.replace('\u200b', '')
r3['subjid_rd3'] = r3['subjid_rd3'].str.replace('\u200b', '')
r4['subjid_rd4'] = r4['subjid_rd4'].str.replace('\u200b', '')
r5['subjid_rd1'] = r5['subjid_rd1'].str.replace('\u200b', '')
r6['subjid_2']   = r6['subjid_2'].str.replace('\u200b', '')
r7['SubjectID']  = r7['SubjectID'].str.replace('\u200b', '')
r8['subjid_rd1'] = r8['subjid_rd1'].str.replace('\u200b', '')
r9['subjid_2']   = r9['subjid_2'].str.replace('\u200b', '')

#Get a lsit of valid subject IDs
valid_sub_ids = pd.read_csv(join(main_dir, 'raw_data', 'IDs_For_DEID_Data.csv'), 
                            names=['sub_id', 'sub_num'])
valid_sub_ids = valid_sub_ids['sub_id'].squeeze().astype(str).str.strip().str.upper().to_list()

#Get a dict of errors and corrections for sub IDs
sub_ids_corrections = pd.read_csv(join(main_dir, 'raw_data', 'SubjID_Replacements.csv'))
sub_ids_corrections['INCORRECT'] = sub_ids_corrections['INCORRECT'].astype(str).str.strip().str.upper()
sub_ids_corrections['CORRECT'] = sub_ids_corrections['CORRECT'].astype(str).str.strip().str.upper()
sub_ids_corrections = sub_ids_corrections.set_index('INCORRECT').squeeze().to_dict()

#Apply corrections
data['subjid_2'].replace(sub_ids_corrections, inplace=True)
demo['subjid_1'].replace(sub_ids_corrections, inplace=True)
r1['subjid_rd1'].replace(sub_ids_corrections, inplace=True)
r2['subjid_rd2'].replace(sub_ids_corrections, inplace=True)
r3['subjid_rd3'].replace(sub_ids_corrections, inplace=True)
r4['subjid_rd4'].replace(sub_ids_corrections, inplace=True)
r5['subjid_rd1'].replace(sub_ids_corrections, inplace=True)
r6['subjid_2'].replace(sub_ids_corrections, inplace=True)
r7['SubjectID'].replace(sub_ids_corrections, inplace=True)
r8['subjid_rd1'].replace(sub_ids_corrections, inplace=True)
r9['subjid_2'].replace(sub_ids_corrections, inplace=True)

#Output a list of suject IDs in the data that don't appear in valid list
if output:
    data.loc[~data['subjid_2'].isin(valid_sub_ids), 'subjid_2'].to_csv(join(main_dir, 'data_check', 'daily_problem_sub_ids.csv'), index=False)
    demo.loc[~demo['subjid_1'].isin(valid_sub_ids), 'subjid_1'].to_csv(join(main_dir, 'data_check', 'demo_problem_sub_ids.csv'), index=False)
    r1.loc[~r1['subjid_rd1'].isin(valid_sub_ids), 'subjid_rd1'].to_csv(join(main_dir, 'data_check', 'r1_problem_sub_ids.csv'), index=False)
    r2.loc[~r2['subjid_rd2'].isin(valid_sub_ids), 'subjid_rd2'].to_csv(join(main_dir, 'data_check', 'r2_problem_sub_ids.csv'), index=False)
    r3.loc[~r3['subjid_rd3'].isin(valid_sub_ids), 'subjid_rd3'].to_csv(join(main_dir, 'data_check', 'r3_problem_sub_ids.csv'), index=False)
    r4.loc[~r4['subjid_rd4'].isin(valid_sub_ids), 'subjid_rd4'].to_csv(join(main_dir, 'data_check', 'r4_problem_sub_ids.csv'), index=False)
    r5.loc[~r5['subjid_rd1'].isin(valid_sub_ids), 'subjid_rd1'].to_csv(join(main_dir, 'data_check', 'r5_problem_sub_ids.csv'), index=False)
    r6.loc[~r6['subjid_2'].isin(valid_sub_ids), 'subjid_2'].to_csv(join(main_dir, 'data_check', 'r6_problem_sub_ids.csv'), index=False)
    r7.loc[~r7['SubjectID'].isin(valid_sub_ids), 'SubjectID'].to_csv(join(main_dir, 'data_check', 'r7_problem_sub_ids.csv'), index=False)
    r8.loc[~r8['subjid_rd1'].isin(valid_sub_ids), 'subjid_rd1'].to_csv(join(main_dir, 'data_check', 'r8_problem_sub_ids.csv'), index=False)
    r9.loc[~r9['subjid_2'].isin(valid_sub_ids), 'subjid_2'].to_csv(join(main_dir, 'data_check', 'r9_problem_sub_ids.csv'), index=False)
    

#Drop subjects without a valid sub ID
print('Dropping %d subjects without a valid subject ID' % sum(~data['subjid_2'].isin(valid_sub_ids)))
data = data[data['subjid_2'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from demographics' 
      % sum(~demo['subjid_1'].isin(valid_sub_ids)))
demo = demo[demo['subjid_1'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 1'
      % sum(~r1['subjid_rd1'].isin(valid_sub_ids)))
r1 = r1[r1['subjid_rd1'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 2'
      % sum(~r2['subjid_rd2'].isin(valid_sub_ids)))
r2 = r2[r2['subjid_rd2'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 3'
      % sum(~r3['subjid_rd3'].isin(valid_sub_ids)))
r3 = r3[r3['subjid_rd3'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 4'
      % sum(~r4['subjid_rd4'].isin(valid_sub_ids)))
r4 = r4[r4['subjid_rd4'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 5'
      % sum(~r5['subjid_rd1'].isin(valid_sub_ids)))
r5 = r5[r5['subjid_rd1'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 6'
      % sum(~r6['subjid_2'].isin(valid_sub_ids)))
r6 = r6[r6['subjid_2'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 7'
      % sum(~r7['SubjectID'].isin(valid_sub_ids)))
r7 = r7[r7['SubjectID'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 8'
      % sum(~r8['subjid_rd1'].isin(valid_sub_ids)))
r8 = r8[r8['subjid_rd1'].isin(valid_sub_ids)]
print('Dropping %d subjects without a valid subject ID from round 9'
      % sum(~r9['subjid_2'].isin(valid_sub_ids)))
r9 = r9[r9['subjid_2'].isin(valid_sub_ids)]


#Replace subject strings with subject IDs
sub_num_key = pd.read_csv(join(main_dir, 'raw_data', 'IDs_For_DEID_Data.csv'), 
                          names=['sub_id', 'sub_num'])
sub_num_key['sub_id'] = sub_num_key['sub_id'].astype(str).str.strip().str.upper()
assert not sub_num_key.duplicated('sub_id', keep=False).any()
assert not sub_num_key.duplicated('sub_num', keep=False).any()
sub_num_key = sub_num_key.set_index('sub_id').squeeze().to_dict()
assert data['subjid_2'].isin(sub_num_key.keys()).all()
assert demo['subjid_1'].isin(sub_num_key.keys()).all()
assert r1['subjid_rd1'].isin(sub_num_key.keys()).all()
assert r2['subjid_rd2'].isin(sub_num_key.keys()).all()
assert r3['subjid_rd3'].isin(sub_num_key.keys()).all()
assert r4['subjid_rd4'].isin(sub_num_key.keys()).all()
assert r5['subjid_rd1'].isin(sub_num_key.keys()).all()
assert r6['subjid_2'].isin(sub_num_key.keys()).all()
assert r7['SubjectID'].isin(sub_num_key.keys()).all()
assert r8['subjid_rd1'].isin(sub_num_key.keys()).all()
assert r9['subjid_2'].isin(sub_num_key.keys()).all()
data['subjid_2'] = data['subjid_2'].replace(sub_num_key).astype(int)
data.rename({'subjid_2':'sub_id'}, axis='columns', inplace=True)
demo['subjid_1'] = demo['subjid_1'].replace(sub_num_key).astype(int)
demo.rename({'subjid_1':'sub_id'}, axis='columns', inplace=True)
r1['subjid_rd1'] = r1['subjid_rd1'].replace(sub_num_key).astype(int)
r1.rename({'subjid_rd1':'sub_id'}, axis='columns', inplace=True)
r2['subjid_rd2'] = r2['subjid_rd2'].replace(sub_num_key).astype(int)
r2.rename({'subjid_rd2':'sub_id'}, axis='columns', inplace=True)
r3['subjid_rd3'] = r3['subjid_rd3'].replace(sub_num_key).astype(int)
r3.rename({'subjid_rd3':'sub_id'}, axis='columns', inplace=True)
r4['subjid_rd4'] = r4['subjid_rd4'].replace(sub_num_key).astype(int)
r4.rename({'subjid_rd4':'sub_id'}, axis='columns', inplace=True)
r5['subjid_rd1'] = r5['subjid_rd1'].replace(sub_num_key).astype(int)
r5.rename({'subjid_rd1':'sub_id'}, axis='columns', inplace=True)
r6['subjid_2'] = r6['subjid_2'].replace(sub_num_key).astype(int)
r6.rename({'subjid_2':'sub_id'}, axis='columns', inplace=True)
r7['SubjectID'] = r7['SubjectID'].replace(sub_num_key).astype(int)
r7.rename({'SubjectID':'sub_id'}, axis='columns', inplace=True)
r8['subjid_rd1'] = r8['subjid_rd1'].replace(sub_num_key).astype(int)
r8.rename({'subjid_rd1':'sub_id'}, axis='columns', inplace=True)
r9['subjid_2'] = r9['subjid_2'].replace(sub_num_key).astype(int)
r9.rename({'subjid_2':'sub_id'}, axis='columns', inplace=True)



#%% INDEX & SORT

#Create unique ID from record_id column
assert data['redcap_repeat_instrument'].isin(['covid19', 'covid19_short_survey', np.nan]).all()
assert all(data['redcap_repeat_instrument'].isna() == (data['april_18_complete'].notna() | data['nov15_complete'].notna()))
data['unique_id'] = data['record_id']
data.loc[data['redcap_repeat_instrument']=='covid19', 'unique_id'] = data.loc[data['redcap_repeat_instrument']=='covid19', 'unique_id'].apply(lambda x: str(x)+'L')
data.loc[data['redcap_repeat_instrument']=='covid19_short_survey', 'unique_id'] = data.loc[data['redcap_repeat_instrument']=='covid19_short_survey', 'unique_id'].apply(lambda x: str(x)+'S')
data.loc[data['april_18_complete'].notna(), 'unique_id'] = data.loc[data['april_18_complete'].notna(), 'unique_id'].apply(lambda x: str(x)+'LA')
data.loc[data['nov15_complete'].notna(), 'unique_id'] = data.loc[data['nov15_complete'].notna(), 'unique_id'].apply(lambda x: str(x)+'LB')

#Index by record ID
data.set_index('unique_id', inplace=True)
assert not data.index.duplicated(keep=False).any()
demo.set_index('record_id', inplace=True)
assert not demo.index.duplicated(keep=False).any()
r1.set_index('record_id', inplace=True)
assert not r1.index.duplicated(keep=False).any()
r2.set_index('record_id', inplace=True)
assert not r2.index.duplicated(keep=False).any()
r3.set_index('record_id', inplace=True)
assert not r3.index.duplicated(keep=False).any()
r4.set_index('record_id', inplace=True)
assert not r4.index.duplicated(keep=False).any()
r5.set_index('record_id', inplace=True)
assert not r5.index.duplicated(keep=False).any()
r6.set_index('record_id', inplace=True)
assert not r6.index.duplicated(keep=False).any()
r7.set_index('ResponseId', inplace=True)
assert not r7.index.duplicated(keep=False).any()
r8.set_index('record_id', inplace=True)
assert not r8.index.duplicated(keep=False).any()
r9.set_index('record_id', inplace=True)
assert not r9.index.duplicated(keep=False).any()

#Sort
data.sort_values(['sub_id', 'todays_date'], inplace=True)
demo.sort_values(['sub_id', 'record_id'], inplace=True)
r1.sort_values(['sub_id', 'record_id'], inplace=True)
r2.sort_values(['sub_id', 'record_id'], inplace=True)
r3.sort_values(['sub_id', 'record_id'], inplace=True)
r4.sort_values(['sub_id', 'record_id'], inplace=True)
r5.sort_values(['sub_id', 'record_id'], inplace=True)
r6.sort_values(['sub_id', 'record_id'], inplace=True)
r7.sort_values(['sub_id', 'ResponseId'], inplace=True)
r8.sort_values(['sub_id', 'record_id'], inplace=True)
r9.sort_values(['sub_id', 'record_id'], inplace=True)



#%% PRE-RAW CORRECTIONS

#Consistent subject ID column naming
r5.rename({'subjid_rd1':'subjid_rd5'}, axis=1, inplace=True)
r6.rename({'subjid_2': 'subjid_rd6'}, axis=1, inplace=True)
r7.rename({'SubjectID': 'subjid_rd7'}, axis=1, inplace=True)
r8.rename({'subjid_rd1': 'subjid_rd8'}, axis=1, inplace=True)
r9.rename({'subjid_2': 'subjid_rd9'}, axis=1, inplace=True)

#Consistent date_time naming
r5.rename({'date_time_rd1':'date_time_rd5'}, axis=1, inplace=True)
r8.rename({'date_time_rd1': 'date_time_rd8'}, axis=1, inplace=True) 

#Remove identifying variables from R7
r7_remove_cols = ['IPAddress', 'Status', 'RecipientLastName', 'RecipientFirstName',
                  'RecipientEmail', 'ExternalReference', 'DistributionChannel',
                  'UserLanguage']
r7.drop(r7_remove_cols, axis='columns', inplace=True)

#Round latitude and longitude to make less identifiable
r7['LocationLatitude'] = pd.to_numeric(r7['LocationLatitude']).round(decimals=1)
r7['LocationLongitude'] = pd.to_numeric(r7['LocationLongitude']).round(decimals=1)

#Rename duplicate named columns in R7
vacc_recall_vars = [x for x in r7.columns if x.startswith('VaccinePhems_')]
repl_dict = {x: x.replace('.1', '_2ndDose').replace('.2', '_SingleDose')
             for x in r7[vacc_recall_vars]}
for key in repl_dict:
    if repl_dict[key].endswith('_2ndDose') or repl_dict[key].endswith('SingleDose'):
        pass
    else:
        repl_dict[key] += '_FirstDose'
r7.rename(repl_dict, axis='columns', inplace=True)
vacc_recall_vars = [x for x in r7.columns if x.endswith('.1')]
repl_dict = {x: x.replace('.1', '_Now') for x in r7[vacc_recall_vars].columns}
r7.rename(repl_dict, axis='columns', inplace=True)
r7.rename({x: x+'_During' for x in r7.loc[:, 'Anger':'Excitement'].columns},
          axis='columns', inplace=True)



#%% OUTPUT RAW DATA

if output:
    
    output_timestamp = str(pd.Timestamp('now').round('min')).replace(' ', '_').replace(':', '_')[:-3]

    daily_id_vars = ['sleepdiary_dreamcontent', 'sleepdiary_info', 'visit', 
                     'respiratory_describe', 'full_open', 'open_question']
    demo_id_vars = ['medical_description', 'institution_describe', 'additional_info', 
                    'school', 'occupation']
    r1_id_vars = ['psqi_5j2']
    r2_id_vars = ['challenging_free', 'positive_free', 'mundane_free', 'unusual_free']
    r3_id_vars = ['city', 'highrisk_othercheck', 'quar_free', 'positive_free_response',
                  'covid_impact_free', 'occupation_other', 'sleepchange_free',
                  'med_free', 'med_other', 'psych_free_1', 'psych_free_2',
                  'condition_free', 'mil_time_free', 'mistakes_free', 'open_anything',
                  'open_anything_2', 'covdream_free']
    r4_id_vars = ['fall_psqi_5j2', 'challenging_free_fut', 'positive_free_fut',
                  'mundane_free_fut', 'unusual_free_fut', 'ind_diff_rem_well', 
                  'ind_diff_forget']
    r5_id_vars = ['psqi_5j2', 'highrisk_othercheck', 'add_dets_cov', 
                  'positive_free_response', 'covid_impact_free', 'city']
    r6_id_vars = ['add_dets_cov']
    r7_id_vars = ['Race_7_TEXT', 'Gender_5_TEXT', 'RecruitmentSource_6_TEXT', 
                  'Q79', 'QID1', 'Q31', 'Q53', 'Q33', 'Q57', 'Q60', 'Q64', 
                  'LocationLatitude', 'LocationLongitude']
    r8_id_vars = ['psqi_5j2', 'city', 'highrisk_othercheck', 'add_dets_cov',
                  'positive_free_response', 'covid_impact_free']
    r9_id_vars = ['add_dets_cov']
    
    
    data.to_csv(join(main_dir, 'export', 'COVID19_combined_raw_%s.csv' % output_timestamp))
    data.drop(daily_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_combined_raw_deid_%s.csv' % output_timestamp))
    
    demo.to_csv(join(main_dir, 'export', 'COVID19_demographics_raw_%s.csv' % output_timestamp))
    demo.drop(demo_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_demographics_raw_deid_%s.csv' % output_timestamp))
    
    r1.to_csv(join(main_dir, 'export', 'COVID19_Round1_raw_%s.csv' % output_timestamp))
    r1.drop(r1_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round1_raw_deid_%s.csv' % output_timestamp))
    
    r2.to_csv(join(main_dir, 'export', 'COVID19_Round2_raw_%s.csv' % output_timestamp))
    r2.drop(r2_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round2_raw_deid_%s.csv' % output_timestamp))
    
    r3.to_csv(join(main_dir, 'export', 'COVID19_Round3_raw_%s.csv' % output_timestamp))
    r3.drop(r3_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round3_raw_deid_%s.csv' % output_timestamp))
    
    r4.to_csv(join(main_dir, 'export', 'COVID19_Round4_raw_%s.csv' % output_timestamp))
    r4.drop(r4_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round4_raw_deid_%s.csv' % output_timestamp))
    
    r5.to_csv(join(main_dir, 'export', 'COVID19_Round5_raw_%s.csv' % output_timestamp))
    r5.drop(r5_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round5_raw_deid_%s.csv' % output_timestamp))
    
    r6.to_csv(join(main_dir, 'export', 'COVID19_Round6_raw_%s.csv' % output_timestamp))
    r6.drop(r6_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round6_raw_deid_%s.csv' % output_timestamp))
    
    r7.to_csv(join(main_dir, 'export', 'COVID19_Round7_raw_%s.csv' % output_timestamp))
    r7.drop(r7_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round7_raw_deid_%s.csv' % output_timestamp))
    
    r8.to_csv(join(main_dir, 'export', 'COVID19_Round8_raw_%s.csv' % output_timestamp))
    r8.drop(r8_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round8_raw_deid_%s.csv' % output_timestamp))
    
    r9.to_csv(join(main_dir, 'export', 'COVID19_Round9_raw_%s.csv' % output_timestamp))
    r9.drop(r9_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round9_raw_deid_%s.csv' % output_timestamp))



#%% DROP DUPLICATES AND INCOMPLETE DATA

#Drop incomplete surveys
incomplete_idx = ((data['covid19_timestamp'] == '[not completed]') 
                  | (data['covid19_short_survey_timestamp'] == '[not completed]'))
data = data[~incomplete_idx]

#Check that there are no duplicates in demographic data
assert not demo['sub_id'].duplicated(keep=False).any()

#Drop Rounds 1 - 9 duplicates
r1.drop_duplicates('sub_id', keep='first', inplace=True)
r2.drop_duplicates('sub_id', keep='first', inplace=True)
r3.drop_duplicates('sub_id', keep='first', inplace=True)
r4.drop_duplicates('sub_id', keep='first', inplace=True)
r5.drop_duplicates('sub_id', keep='first', inplace=True)
r6.drop_duplicates('sub_id', keep='first', inplace=True)
r7.drop_duplicates('sub_id', keep='first', inplace=True)
r8.drop_duplicates('sub_id', keep='first', inplace=True)
r9.drop_duplicates('sub_id', keep='first', inplace=True)



#%% FORMATTING: DAILY SURVEYS

print('\n\n##### FORMATTING #####')

#Make survey type (short vs full) a categorical variable
data['redcap_repeat_instrument'] = data['redcap_repeat_instrument'].astype('category')

#Combine timestamp columns
data['redcap_timestamp'] = data['covid19_timestamp']
data.loc[data['redcap_timestamp'].isna(), 'redcap_timestamp'] = data.loc[data['redcap_timestamp'].isna(), 'covid19_short_survey_timestamp']
data.loc[data['april_18_complete'].notna(), 'redcap_timestamp'] = data.loc[data['april_18_complete'].notna(), 'april_18_timestamp']
data.loc[data['nov15_complete'].notna(), 'redcap_timestamp'] = data.loc[data['nov15_complete'].notna(), 'nov15_timestamp']
assert data['redcap_timestamp'].notna().all()

#Convert timestamps to datetime format
timestamp_vars = ['covid19_timestamp', 'covid19_short_survey_timestamp', 
                  'redcap_timestamp', 'todays_date']
for col in timestamp_vars:
    data[col] = data[col].replace({'[not completed]':pd.NaT})
    data[col] = pd.to_datetime(data[col])

#Convert sleepdiary times to timedelta
time_vars = ['sleepdiary_bedtime', 'sleepdiary_fallasleep', 
             'sleepdiary_waketime', 'sleepdiary_outofbed']
for col in time_vars:
    assert data.loc[data[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
    data[col] = pd.to_datetime(data[col]) - pd.Timestamp.today().normalize()

#Fix fever tempertures
data['fever_temp'] = data['fever_temp'].replace({'37,5': 37.5,
                                                 '38,8': 38.8,
                                                 '38,6': 38.6,
                                                 '38,5': 38.5,
                                                 '100.2 this morning': 100.2})
#Try to convert to numeric
for i in np.where(~data['fever_temp'].apply(np.isreal))[0]:
    try:
        data.iloc[i, data.columns.get_loc('fever_temp')] = float(data.iloc[i, data.columns.get_loc('fever_temp')])
    except ValueError:
        print('"%s" in fever_temp cannot be made numeric'
              % data.iloc[i, data.columns.get_loc('fever_temp')])
#Use nan for values that couldn't be converted to numeric
print('\nReplacing %d non_numeric fever_temp values with nan\n' 
      % sum(~data['fever_temp'].apply(np.isreal)))
data.loc[~data['fever_temp'].apply(np.isreal), 'fever_temp'] = np.nan
#Make the column numeric
data['fever_temp'] = data['fever_temp'].astype('float')

#Subtract 1 from depression columns so they start at 0
depression_vars = ['depression1', 'depression2', 'depression3', 'depression4',
                   'depression5', 'depression6', 'depression7', 'depression8']
data[depression_vars] += -1

#For questions where a numeric answer was conditional on a yes answer to another question
#insert 0 if the first question was no; for example, if the participants said they
#didn't nap, naptime becomes 0
data.loc[data['sleepdiary_wakes']==0, 'night_awakening_time'] = 0
data.loc[data['sleepdiary_nap']==0, 'sleepdiary_naptime'] = 0
data.loc[data['socialize']==0, 'socialize_min'] = 0

### Remove impossible values ###

data.loc[data['sleepdiary_sleeplatency'] > 24*60, 'sleepdiary_sleeplatency'] = np.nan

data.loc[data['sleepdiary_naptime'] > 24*60, 'sleepdiary_naptime'] = np.nan
data.loc[data['night_awakening_time'] > 24*60, 'night_awakening_time'] = np.nan
data.loc[data['socialize_min'] > 24*60, 'socialize_min'] = np.nan
data.loc[data['alcohol_bev']>48, 'alcohol_bev'] = np.nan
data.loc[(data['redcap_timestamp'] - pd.to_datetime('1/23/20')).dt.days < data['quarantine_days'], 'quarantine_days'] = np.nan



#%% REFERENCE DATE
#Find (best guess for) the date that answers are referring to in daily data

#Find rows where dates might be problematic because todays_date is duplicated
#or todays_date and redcap_timestamp don't match
idx = sub_date_duplicates(data, keep=False)
idx = idx | (np.abs(data['todays_date'] - data['redcap_timestamp']) > pd.Timedelta(1, unit='days'))
if output:
    data.loc[idx, ['sub_id', 'todays_date', 'redcap_timestamp', 'covid19_timestamp', 'covid19_short_survey_timestamp']].to_csv(join(main_dir, 'data_check', 'ref_date_problems.csv'))

#For rows without obvious date problems, assume response refers to the day before
data.loc[~idx, 'ref_date'] = data.loc[~idx, 'todays_date'] - pd.Timedelta(1, unit='days')

#Assume early morning times refer to two days before
idx = data.reset_index().set_index('todays_date').between_time('00:00', '04:00')['unique_id'].to_list()
data.loc[idx, 'ref_date'] += pd.Timedelta(-1, unit='days')

#Convert to date only
data['ref_date'] = data['ref_date'].dt.normalize()

#Days elapsed
data['days_elapsed'] = (data['ref_date'] - data['ref_date'].min()).dt.days



#%% FORMATTING: DEMOGRAPHIC DATA

#Convert timestamp to datetime
demo['date_time'] = pd.to_datetime(demo['date_time'])

#Make number of dependent children numeric
demo['dependent_children'] = demo['dependent_children'].replace({'None': 0,
                                                                 'No': 0,
                                                                 'no': 0,
                                                                 'none': 0,
                                                                 'Not applicable': 0})
for i in np.where(~demo['dependent_children'].apply(np.isreal))[0]:
    try:
        demo.iloc[i, demo.columns.get_loc('dependent_children')] = float(demo.iloc[i, demo.columns.get_loc('dependent_children')])
    except ValueError:
        print('%s in dependent_children could not be converted to numeric' %
              demo.iloc[i, demo.columns.get_loc('dependent_children')])
print('\nReplacing %d non_numeric dependent_children values with nan' 
      % sum(~demo['dependent_children'].apply(np.isreal)))
demo.loc[~demo['dependent_children'].apply(np.isreal), 'dependent_children'] = np.nan
demo['dependent_children'] = demo['dependent_children'].astype('float')

#Clean up country variable
country_corrections = {'USA': 'UNITED STATES',
                       'US': 'UNITED STATES',
                       'UNITED STATES OF AMERICA': 'UNITED STATES',
                       'U.S.': 'UNITED STATES',
                       'U.S': 'UNITED STATES',
                       'U.S.A.': 'UNITED STATES',
                       'AMERICAN': 'UNITED STATES',
                       'UNITED STATED OF AMERICA': 'UNITED STATES',
                       'UNITED STATES IF AMERICA': 'UNITED STATES',
                       'UNITED STATES OF AMERICAN': 'UNITED STATES',
                       'THE UNITED STATES': 'UNITED STATES',
                       'UNITED STTES': 'UNITED STATES',
                       'UNITED STATE OF AMERICA': 'UNITED STATES',
                       'UNITED STATE': 'UNITED STATES',
                       'AMERICA': 'UNITED STATES',
                       'EE.UU': 'UNITED STATES',
                       'ENGLAND': 'UNITED KINGDOM',
                       'SCOTLAND': 'UNITED KINGDOM',
                       'UK': 'UNITED KINGDOM',
                       'THE NETHERLANDS': 'NETHERLANDS',
                       'BRASIL': 'BRAZIL',
                       'MÉXICO': 'MEXICO',
                       'KOREA': 'SOUTH KOREA',
                       'KSA': 'SAUDI ARABIA',
                       'P. R. CHINA': 'CHINA'}
demo['country'] = demo['country'].str.strip().str.upper()
demo['country'] = demo['country'].replace(country_corrections)
demo.loc[1563, 'country'] = 'UNITED STATES'
demo.loc[1748, 'country'] = 'UNITED STATES'
demo.loc[1650, 'country'] = 'UNITED STATES'
with open(join(main_dir, 'code', 'countries.txt'), encoding='UTF_8') as f_in:
    countries = [x.strip().upper() for x in f_in.readlines()]
assert demo['country'].isin(countries).all()

#Clean up state variable
state_abbrev = pd.read_csv(join(main_dir, 'code', 'state_abbreviations.csv'))
demo['state'] = demo['state'].str.strip().str.upper().str.replace('.','')
demo['state'] = demo['state'].replace(state_abbrev.set_index('state')['short'].to_dict())
demo['state'] = demo['state'].replace(state_abbrev.set_index('long')['short'].to_dict())
state_corrections = {'WASHINGTON, DC': 'DC',
                     'BOSTON': 'MA',
                     'WASHINGTON STATE': 'WA',
                     'CINNECTICUT': 'CT',
                     'MANHATTAN, NY': 'NY',
                     'NEW YORK STATE': 'NY',
                     "HAWAI'I": 'HI',
                     'CHICAGO': 'IL',
                     'NEW YORK CITY': 'NY',
                     'NEW HAMPHIRE': 'NH',
                     'PROVIDENCE, RHODE ISLAND': 'RI',
                     'NYS': 'NY',
                     'CAMBRIDGE, BOSTON': 'MA',
                     'MASSACHUSSETS': 'MA',
                     'INDIANNA': 'IN',
                     'ONT': 'ON',
                     'TORONTO, ON': 'ON',
                     'PEI': 'PE',
                     'SOUTH CAROLINA/ GEORGIA':np.nan,
                     'NEW  YORK': 'NY',
                     '': np.nan,
                     'MAS': np.nan}
demo['state'] = demo['state'].replace(state_corrections)
assert demo.loc[demo['country'].isin(['UNITED STATES', 'CANADA']), 'state'].isin([*state_abbrev['short'], np.nan]).all()

#Standardize school response
school_replace = pd.read_csv(join(main_dir, 'raw_data', 'school_replacements_for_deid.csv')).set_index('ORIGINAL').squeeze().to_dict()
school_replace = {**school_replace,
                  '4 year college': '4 year college/university',
                  '4 year': '4 year college/university',
                  '4-year University': '4 year college/university',
                  '4 Year University': '4 year college/university',
                  'University (4 year)': '4 year college/university',
                  '4 Year': '4 year college/university',
                  '4 year': '4 year college/university',
                  '4 year University': '4 year college/university',
                  '4-year private university': '4 year college/university',
                  '2 year college': '2 year college/university',
                  '2-year College': '2 year college/university',
                  '2.5year college': '2 year college/university',
                  '2 Year College': '2 year college/university',
                  'community college 2 year': '2 year college/university',
                  '2 year': '2 year college/university',
                  '3 year college': '3 year college/university',
                  'med school': 'Medical School',
                  'Graduate school - medical school': 'Medical School',
                  'medical school': 'Medical School',
                  'graduate school, PhD': 'Graduate School (PhD)',
                  'grad school': 'Graduate School (unspecified)',
                  '4-year college/university': '4 year college/university',
                  "Integrated Master's (4 year university)": '4 year college/university',
                  '4 yr university': '4 year college/university',
                  'Business University': 'College/University (unspecified)',
                  'University of applied science': 'College/University (unspecified)',
                  'Mandeville high school': 'High School',
                  'Post grad, PhD': 'Graduate School (PhD)'}
demo['school'] = demo['school'].str.strip()
demo['school'] = demo['school'].replace(school_replace)
idx = ~demo['school'].isin([np.nan, *set(school_replace.values())])
print('Replacing the following school responses with nan:')
print(demo.loc[idx, 'school'])
demo.loc[idx, 'school'] = np.nan

#Standardize occupation response
occ_replace = pd.read_csv(join(main_dir, 'raw_data', 'occupation_replacements_for_deid.csv')).set_index('old').squeeze().to_dict()
demo['occupation'] = demo['occupation'].replace(occ_replace)

#Remove impossible values
demo.loc[demo['age1']>118, 'age1'] = np.nan
assert not (data['night_awakening_time'] > 24*60).any()
assert not (demo['dependents'] > 20).any()
assert not (demo['housing'] > 20).any()
assert not (demo['dependent_children'] > 20).any()



#%% FORMATTING: ROUND 1

#Timestamp variables
r1['round_1_timestamp'] = r1['round_1_timestamp'].replace({'[not completed]':pd.NaT})
r1['round_1_timestamp'] = pd.to_datetime(r1['round_1_timestamp'])
r1['date_time_rd1'] = pd.to_datetime(r1['date_time_rd1'])

#Time (clock) variables
r1_clock_vars = ['psqi_1', 'psqi_3', 'mtq_p3', 'mtq_p4', 'mtq_p5', 'mtq_p6', 
                 'mtq_3', 'mtq_p8', 'mtq_p9', 'mtq_p10']
for col in r1_clock_vars:
    assert r1.loc[r1[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
    r1[col] = pd.to_datetime(r1[col]) - pd.Timestamp.today().normalize()
    assert all(r1[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1[col].isna())

#Correct some naming errors
r1.rename({'psqi_5h_2': 'psqi_5i',
           'mtq_p8': 'mtq_4',
           'mtq_p9': 'mtq_5',
           'mtq_p10': 'mtq_6'},
          axis='columns',
          inplace=True)

#Re-scale PSQI response to start at 0
r1.loc[:, 'psqi_5a':'psqi_5j'] += -1
r1.loc[:, 'psqi_6':'psqi_9'] += -1

#Replace hours sleep greater than 24 with missing
r1.loc[r1['psqi_4'] > 24, 'psqi_4'] = np.nan

#Replace days/wk worked greater than 7 with missing
r1.loc[r1['mtq_p2']>7, 'mtq_p2'] = np.nan
r1.loc[r1['mtq_2']>7, 'mtq_2'] = np.nan



#%% FORMATTING: ROUND 2

#Timestamp variables
r2['round_2_timestamp'] = r2['round_2_timestamp'].replace({'[not completed]':pd.NaT})
r2['round_2_timestamp'] = pd.to_datetime(r2['round_2_timestamp'])
r2['date_time_rd2'] = pd.to_datetime(r2['date_time_rd2'])

#Date variables
r2_date_vars = ['stayhome_begin_us', 'stayhome_end_us', 'stayhome_begin', 
                'stayhome_end', 'normal_date', 'mask_date', 'meetings_date', 
                'bigevents_date', 'shaking_hands_date']
r2['mask_date'] = r2['mask_date'].replace({'9999-09-09':pd.NaT})
r2['bigevents_date'] = r2['bigevents_date'].replace({'0201-03-01':pd.NaT})
r2['shaking_hands_date'] = r2['shaking_hands_date'].replace({'0101-01-01':pd.NaT,
                                                             '3000-06-20':pd.NaT,
                                                             '5050-05-05':pd.NaT})
r2['stayhome_end'] = r2['stayhome_end'].replace({'0101-01-01':pd.NaT})
for col in r2_date_vars:
    r2[col] = pd.to_datetime(r2[col])



#%% FORMATTING: ROUND 3

#Timestamp variables
r3['round_3_timestamp'] = r3['round_3_timestamp'].replace({'[not completed]':pd.NaT})
r3['round_3_timestamp'] = pd.to_datetime(r3['round_3_timestamp'])
r3['date_time_rd3'] = pd.to_datetime(r3['date_time_rd3'])

#Age mistakes
r3.loc[r3['age']>118, 'age'] = np.nan

#Clean up countries
r3_country_corrections = {'USA': 'UNITED STATES',
                          'US': 'UNITED STATES',
                          'UNITED STATES OF AMERICA': 'UNITED STATES',
                          'U.S.': 'UNITED STATES',
                          'UNITED STAYED': 'UNITED STATES',
                          'AMERICA': 'UNITED STATES',
                          'THE US OF A': 'UNITED STATES',
                          'UNITES STATES': 'UNITED STATES',
                          'U.S.A.':'UNITED STATES',
                          'USAMA': 'UNITED STATES',
                          'ENGLAND': 'UNITED KINGDOM',
                          'UK': 'UNITED KINGDOM',
                          'THE NETHERLANDS (EUROPE)':'NETHERLANDS',
                          'THE NETHERLANDS':'NETHERLANDS',
                          'COMMONWEALTH OF THE NORTHERN MARIANA ISLANDS': 'NORTHERN MARIANA ISLANDS',
                          'MÉXICO': 'MEXICO',
                          'YES': np.nan,
                          '0': np.nan,
                          'NONE': np.nan,
                          'CANADA AND THE US EQUAL': np.nan}
r3['country_3mo'] = r3['country_3mo'].str.strip().str.upper()
r3['country_3mo'] = r3['country_3mo'].replace(r3_country_corrections)
assert all(r3['country_3mo'].isin(countries) | r3['country_3mo'].isna())

#Clean up states
r3['state_3mo'] = r3['state_3mo'].str.strip().str.upper().str.replace('.','')
r3['state_3mo'] = r3['state_3mo'].replace(state_abbrev.set_index('state')['short'].to_dict())
r3['state_3mo'] = r3['state_3mo'].replace(state_abbrev.set_index('long')['short'].to_dict())
r3.loc[~r3['country_3mo'].isin(['UNITED STATES', 'CANADA']), 'state_3mo'] = np.nan
r3_state_corrections = {'YES': np.nan,
                        'MASSACHUSSETTS': 'MA',
                        'MASDACHUSETTS': 'MA',
                        'AMHERST, MA': 'MA',
                        'WASHINGTON (STATE)': 'WA',
                        'MASSACHUSETT': 'MA',
                        'WASHINGTON STATE': 'WA',
                        'MASSACHUSETTES': 'MA',
                        'NY- LONG ISLAND': 'NY',
                        'WASHINGTON DC':'DC',
                        'BRITHISH COLUMBIA':'BC',
                        'NEW YORK/MASSACHUSETTS (EQUAL TIME)': np.nan}
r3['state_3mo'] = r3['state_3mo'].replace(r3_state_corrections)
assert r3.loc[r3['state_3mo'].notna(), 'state_3mo'].isin(state_abbrev['short']).all()

#TO DO: Figure out how to deal with date variables given inconsistent formatting
r3_date_vars = [x for x in r3.columns if x.endswith('_start') or x.endswith('_end')]



#%% FORMATTING: ROUND 4

#Timestamp variables
r4['round_4_timestamp'] = r4['round_4_timestamp'].replace({'[not completed]':pd.NaT})
r4['round_4_timestamp'] = pd.to_datetime(r4['round_4_timestamp'])
r4['date_time_rd4'] = pd.to_datetime(r4['date_time_rd4'])

#Fix incorrect date
r4.loc[r4['date_time_rd4'] < pd.Timestamp('9/27/20'), 'date_time_rd4'] = pd.NaT

#Date variables
r4_date_vars = ['stayhome_begin_us_fut', 'stayhome_end_us_fut', 
                'stayhome_begin_fut', 'stayhome_end_fut']
r4['stayhome_end_fut'] = r4['stayhome_end_fut'].replace({'3020-05-11':'2020-05-11'})
for col in r4_date_vars:
    r4[col] = pd.to_datetime(r4[col])

#Time (clock) variables
r4_clock_vars = ['fall_psqi_1', 'fall_psqi_3', 'fall_mtq_3', 'fall_mtq_4', 
                 'fall_mtq_5', 'fall_mtq_6']
for col in r4_clock_vars:
    assert r4.loc[r4[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
    r4[col] = pd.to_datetime(r4[col]) - pd.Timestamp.today().normalize()
    assert all(r4[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r4[col].isna())

#Re-scale fall_psqi response to start at 0
r4.loc[:, 'fall_psqi_5a':'fall_psqi_5j'] += -1
r4.loc[:, 'fall_psqi_6':'fall_psqi_9'] += -1

#Replace hours sleep greater than 24 with missing
r4.loc[r4['fall_psqi_4'] > 24, 'fall_psqi_4'] = np.nan

#Replace days/wk worked greater than 7 with missing
r4.loc[r4['fall_mtq_2']>7, 'fall_mtq_2'] = np.nan



#%% FORMATTING: ROUND 5

#Date variables
r5['round_5_timestamp'] = r5['round_5_timestamp'].replace({'[not completed]':pd.NaT})
r5['normal_date_feb'].replace({'3030-10-10': pd.NaT,
                               '2600-01-01': pd.NaT}, inplace=True)
r5['mask_date_feb'].replace({'2600-01-01': pd.NaT}, inplace=True)
r5['meetings_date_feb'].replace({'2600-01-01': pd.NaT}, inplace=True)
r5['bigevents_date_feb'].replace({'2600-01-01': pd.NaT}, inplace=True)
r5['shakinghands_date_feb'].replace({'2600-01-01': pd.NaT}, inplace=True)
r5_date_vars = (['round_5_timestamp', 'date_time_rd5', 'date_cov', 'vacc_date'] 
                + [col for col in r5.columns if '_date_' in col])
for col in r5_date_vars:
    r5[col] = pd.to_datetime(r5[col])

#Time (clock) variables
r5_clock_vars = ['psqi_1', 'psqi_3', 'mtq_3', 'mtq_p8', 'mtq_p9', 'mtq_p10']
for col in r5_clock_vars:
    assert r5.loc[r5[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
    r5[col] = pd.to_datetime(r5[col]) - pd.Timestamp.today().normalize()
    assert all(r5[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r5[col].isna())

#Clean up countries
r5_country_corrections = {'USA': 'UNITED STATES',
                          'US': 'UNITED STATES',
                          'UNITED STATES OF AMERICA': 'UNITED STATES',
                          'U.S.': 'UNITED STATES',
                          'THE UNITED STATES': 'UNITED STATES',
                          'THE UNITED STATES OF AMERICA': 'UNITED STATES',
                          'USTATESNITED': 'UNITED STATES',
                          'U.S.A.': 'UNITED STATES',
                          'US OF A': 'UNITED STATES',
                          'UNITED STATE': 'UNITED STATES',
                          'USAP': 'UNITED STATES',
                          'RUSSIAN FEDERATION': 'RUSSIA',
                          'ENGLAND': 'UNITED KINGDOM',
                          'UK': 'UNITED KINGDOM',
                          'MÉXICO': 'MEXICO',
                          'COMMONWEALTH OF THE NORTHERN MARIANA ISLANDS': 'NORTHERN MARIANA ISLANDS',
                          '1': np.nan}
r5['country_3mo'] = r5['country_3mo'].str.strip().str.upper()
r5['country_3mo'] = r5['country_3mo'].replace(r5_country_corrections)
assert all(r5['country_3mo'].isin(countries) | r5['country_3mo'].isna())

#Clean up states
r5['state_3mo'] = r5['state_3mo'].str.strip().str.upper().str.replace('.','')
r5['state_3mo'] = r5['state_3mo'].replace(state_abbrev.set_index('state')['short'].to_dict())
r5['state_3mo'] = r5['state_3mo'].replace(state_abbrev.set_index('long')['short'].to_dict())
r5.loc[~r5['country_3mo'].isin(['UNITED STATES', 'CANADA']), 'state_3mo'] = np.nan
r5_state_corrections = {'MASSACHUSETTES': 'MA',
                        'WASHINGTON DC': 'DC',
                        'EST': np.nan,
                        'MASSACHUSETS': 'MA',
                        'FLORIDA AND MASSACHUSETTS': np.nan,
                        'ILLNOIS': 'IL',
                        'MASSACHUSSETTS': 'MA',
                        'MASSACHUSETTS AND RHODE ISLAND': np.nan,
                        'STAYING IN': np.nan,
                        '50% OHIO & 50% CALIFORNIA': np.nan,
                        'WORCESTER': 'MA',
                        'MASSACHUCETTES': 'MA',
                        'WASHINGTON STATE': 'WA'}
r5['state_3mo'] = r5['state_3mo'].replace(r5_state_corrections)
assert r5.loc[r5['state_3mo'].notna(), 'state_3mo'].isin(state_abbrev['short']).all()

#Clean up vaccine type
r5['vacc_type'] = r5['vacc_type'].str.strip().str.upper()
r5.replace({'PHIZER': 'PFIZER',
            'OXFORD': 'ASTRAZENECA',
            'MODERN': 'MODERNA',
            'PFOZER': 'PFIZER',
            '1': np.nan}, inplace=True)
assert r5['vacc_type'].isin(['PFIZER', 'MODERNA', 'ASTRAZENECA', 'SINOVAC', 'JJ', np.nan]).all()

#Re-scale PSQI response to start at 0
r5.loc[:, 'psqi_5a':'psqi_5j'] += -1
r5.loc[:, 'psqi_6':'psqi_9'] += -1

#Rescale to days per week
r5['bs_fall'] = r5['bs_fall']/2
r5['bs_sp2021'] = r5['bs_sp2021']/2

#Remove impossible values
r5.loc[r5['psqi_4']>24, 'psqi_4'] = np.nan
r5.loc[r5['mtq_2']>7, 'mtq_2'] = np.nan



#%% FORMATTING: ROUND 6

#Format dates
r6['april_18_timestamp'].replace({'[not completed]':pd.NaT}, inplace=True)
r6_date_vars = ['april_18_timestamp', 'todays_date', 'vacc_date', 'date_cov']
for col in r6_date_vars:
    r6[col] = pd.to_datetime(r6[col])
    
#Clean up vaccine type
r6['vacc_type'] = r6['vacc_type'].str.strip().str.upper()
r6.replace({'JOHNSON & JOHNSON': 'JJ',
            'J&J': 'JJ',
            'JOHNSON&JOHNSON': 'JJ',
            'PHIZER': 'PFIZER',
            'PFISZER': 'PFIZER',
            'PFIXER': 'PFIZER',
            'OXFORD': 'ASTRAZENECA',
            'ASTRA ZENECA': 'ASTRAZENECA',
            'CORONAVAC': 'SINOVAC'}, 
           inplace=True)
assert r6['vacc_type'].isin(['PFIZER', 'MODERNA', 'ASTRAZENECA', 'SINOVAC', 'JJ', np.nan]).all()



#%% FORMATTING: ROUND 7

r7['Finished'].replace({'True':True, 'False':False}, inplace=True)

#Date variables
r7_date_vars = ['StartDate', 'RecordedDate']
for col in r7_date_vars:
    r7[col] = pd.to_datetime(r7[col])
    
#Numbers stored as text
for col in r7.columns:
    try:
        if all(r7[col].isna() | r7[col].str.isnumeric()):
            r7[col] = pd.to_numeric(r7[col])
    except AttributeError:
        pass
    
#Likert scales stored as text
likert_dict = {'Strongly disagree':1, 'Disagree':2, 'Neither disagree nor agree':3, 
               'Agree':4, 'Strongly agree':5}
likert2num_vars = [col for col in r7.columns if col.startswith('SilverLinings_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(1, 5) | r7[col].isna())
likert_dict = {'Not Likely': 1, 'Somewhat Likely': 2, 'Moderately Likely': 3, 
               'Very Likely': 4}
likert2num_vars = [col for col in r7.columns if col.startswith('Q89_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(1, 4) | r7[col].isna())
likert_dict = {'Would NEVER doze': 0, 'SLIGHT chance of dozing': 1, 
               'MODERATE chance of dozing': 2, 'HIGH chance of dozing': 3}
likert2num_vars = [col for col in r7.columns if col.startswith('Q90_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(0, 3) | r7[col].isna())
likert_dict = {'Not true at all': 0, 'Rarely true': 1, 'Sometimes true': 2, 
              'Often true': 3, 'True nearly all the time': 4}
likert2num_vars = [col for col in r7.columns if col.startswith('Q91_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(0, 4) | r7[col].isna())
likert_dict = {'Not at all': 1, 'Slightly': 2, 'Moderately': 3, 'A lot': 4, 
               'Extremely': 5}
likert2num_vars = [col for col in r7.columns if col.startswith('Q92_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(1, 5) | r7[col].isna())
likert_dict = {'All of the time': 0, 'Often': 1, 'Sometimes': 2, 'Rarely': 3, 'Never': 4}
likert2num_vars = [col for col in r7.columns if col.startswith('Q93_')]
for col in likert2num_vars:
    assert r7[col].isin([*likert_dict.keys(), np.nan]).all()
    r7[col].replace(likert_dict, inplace=True)
    assert all(r7[col].between(0, 4) | r7[col].isna())



#%% FORMATTING: ROUND 8
    
#Date variables
r8['round_8_timestamp'] = r8['round_8_timestamp'].replace({'[not completed]':pd.NaT})
r8_date_vars = ['round_8_timestamp', 'date_time_rd8', 'date_cov', 'vacc_date']
for col in r8_date_vars:
    r8[col] = pd.to_datetime(r8[col])
    
#Time (clock) variables
r8_clock_vars = ['psqi_1', 'psqi_3', 'mtq_3', 'mtq_p8', 'mtq_p9', 'mtq_p10']
for col in r8_clock_vars:
    assert r8.loc[r8[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
    r8[col] = pd.to_datetime(r8[col]) - pd.Timestamp.today().normalize()
    assert all(r8[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r8[col].isna())
    
#Replace impossible values
r8.loc[r8['psqi_4'] > 24, 'psqi_4'] = np.nan
r8.loc[r8['mtq_2'] > 7, 'mtq_2'] = np.nan
r8.loc[r8['vacc_date'] < pd.to_datetime('2020/01/01'), 'vacc_date'] = pd.NaT

#Re-scale PSQI response to start at 0
r8.loc[:, 'psqi_5a':'psqi_5j'] += -1
r8.loc[:, 'psqi_6':'psqi_9'] += -1

#Clean up countries
r8['country_3mo'] = r8['country_3mo'].str.strip().str.upper()
r8_country_corrections = {'USA': 'UNITED STATES',
                          'US': 'UNITED STATES',
                          'UNITED STATES OF AMERICA': 'UNITED STATES',
                          'U.S.': 'UNITED STATES',
                          'UNITED STATE': 'UNITED STATES',
                          'U.S.A.': 'UNITED STATES',
                          'THE UNITED STATES': 'UNITED STATES',
                          'UNITE STATES OF AMERICA': 'UNITED STATES',
                          'AMERICA': 'UNITED STATES',
                          'SCOTLAND': 'UNITED KINGDOM',
                          '1': np.nan,
                          '0': np.nan}
r8['country_3mo'] = r8['country_3mo'].replace(r8_country_corrections)
assert all(r8['country_3mo'].isin(countries) | r8['country_3mo'].isna())

#Clean up states
r8['state_3mo'] = r8['state_3mo'].str.strip().str.upper().str.replace('.','')
r8['state_3mo'] = r8['state_3mo'].replace(state_abbrev.set_index('state')['short'].to_dict())
r8['state_3mo'] = r8['state_3mo'].replace(state_abbrev.set_index('long')['short'].to_dict())
r8.loc[~r8['country_3mo'].isin(['UNITED STATES', 'CANADA']), 'state_3mo'] = np.nan
r8_state_corrections = {'NEW HAMSPHIRE': 'NH',
                        'MASSACHUSSETTS': 'MA',
                        'MASSACHUSETTTS': 'MA',
                        'PENNISYLVANIA': 'PA',
                        'WASHINGTON STATE': 'WA',
                        'WORCESTER': 'MA',
                        'DOUGLAS': np.nan,
                        '50% OHIO & 50% CALIFORNIA': np.nan,
                        'FLORIDA AND MASSACHUSETTS': np.nan}
r8['state_3mo'] = r8['state_3mo'].replace(r8_state_corrections)
assert r8['state_3mo'].isin([*state_abbrev['short'], np.nan]).all()

#Clean up vaccine type
r8['vacc_type'] = r8['vacc_type'].str.strip().str.upper()
r8.replace({'JOHNSON & JOHNSON': 'JJ',
            'JOHNSON AND JOHNSON': 'JJ',
            'PRIZER': 'PFIZER',
            'PFIZER3': 'PFIZER',
            'BIONTECH': 'PFIZER',
            'MODERNA 2 SHOTS AND PFIZER ONE SHOT': 'MIXED',
            'J&J, PFIZER': 'MIXED',
            'BIONTECH PFIZER': 'PFIZER',
            'PHIZER': 'PFIZER',
            'JENSEN': 'JJ',
            'SINOVAC AND PFIZER ON THE BOOSTER': 'MIXED',
            'CORONAVAC': 'SINOVAC',
            'COVISHIELD': 'ASTRAZENECA',
            'ASTRAZENECA + PFIZER': 'MIXED',
            'PFIZER/BIONTECH': 'PFIZER',
            'ASTRAZENICA': 'ASTRAZENECA',
            'MORDERNA': 'MODERNA',
            'PFLIZER': 'PFIZER',
            'PFIZER (2) AND MODERNA (BOOSTER)': 'MIXED',
            'ASTRAZENENCA (COVISHIELD IN INDIA)': 'ASTRAZENECA',
            'JOHNSON&JOHNSON': 'JJ',
            'JOHNSON & JOHONSON': 'JJ',
            "PFIZER - I RECEIVED 3 DOSES, BUT THE NEXT QUESTION DOESN'T PROVIDE THAT OPTION.": 'PFIZER',
            '4/8/21': np.nan}, inplace=True)
assert r8['vacc_type'].isin(['PFIZER', 'MODERNA', 'ASTRAZENECA', 'SINOVAC', 'JJ', 
                             'MIXED', np.nan]).all()

#TO DO: WHY DOES THIS NEED TO BE DONE AGAIN?
#Date variables
r8['round_8_timestamp'] = r8['round_8_timestamp'].replace({'[not completed]': pd.NaT})
r8_date_vars = ['round_8_timestamp', 'date_time_rd8', 'date_cov', 'vacc_date']
for col in r8_date_vars:
    r8[col] = pd.to_datetime(r8[col])



#%% FORMATTING: ROUND 9

#Date variables
r9['nov15_timestamp'] = r9['nov15_timestamp'].replace({'[not completed]':pd.NaT})
r9['vacc_date_boost'].replace({'0421-01-01': pd.NaT}, inplace=True)
r9_date_vars = ['nov15_timestamp', 'todays_date', 'date_cov', 'vacc_date', 'vacc_date_boost']
for col in r9_date_vars:
    r9[col] = pd.to_datetime(r9[col])

#Clean up states
r9['est_state'] = r9['est_state'].str.strip().str.upper().str.replace('.','')
r9['est_state'] = r9['est_state'].replace(state_abbrev.set_index('state')['short'].to_dict())
r9['est_state'] = r9['est_state'].replace(state_abbrev.set_index('long')['short'].to_dict())
r9_state_corrections = {'MASSACHUSSETTS': 'MA',
                        'ARKANSAA': 'AR',
                        'WASHINGTON STATE': 'WA',
                        '0': np.nan}
r9['est_state'] = r9['est_state'].replace(r9_state_corrections)
assert r9['est_state'].isin([*state_abbrev['short'], np.nan]).all()

#Clean up vaccine type
r9['vacc_type'] = r9['vacc_type'].str.strip().str.upper()
r9['vacc_type_boost'] = r9['vacc_type_boost'].str.strip().str.upper()
vacc_corrections = {'JOHNSON & JOHNSON': 'JJ',
                    'JOHNSON AND JOHNSON': 'JJ',
                    'BIONTECH': 'PFIZER',
                    'MODERNA 2 SHOTS AND PFIZER ONE SHOT': 'MIXED',
                    'CORONAVAC': 'SINOVAC',
                    'COVISHIELD': 'ASTRAZENECA',
                    'PFIZER/BIONTECH': 'PFIZER',
                    'JOHNSON&JOHNSON': 'JJ',
                    'JOHNSON & JOHONSON': 'JJ',
                    'PHIZER': 'PFIZER',
                    'SPUTNIK V': 'SPUTNIK',
                    'JENSEN': 'JJ',
                    'ASTRAZENICA': 'ASTRAZENECA',
                    'PFRIZER': 'PFIZER',
                    'J AND J': 'JJ'}
r9['vacc_type'].replace(vacc_corrections, inplace=True)
r9['vacc_type_boost'].replace(vacc_corrections, inplace=True)
assert r9['vacc_type'].isin(['PFIZER', 'MODERNA', 'ASTRAZENECA', 'SINOVAC', 'JJ', 
                             'SPUTNIK', 'MIXED', np.nan]).all()
assert r9['vacc_type_boost'].isin(['PFIZER', 'MODERNA', 'ASTRAZENECA', 'SINOVAC', 'JJ', 
                                   'SPUTNIK', 'MIXED', np.nan]).all()

#Replace impossible values
r9.loc[r9['vacc_date'] > pd.to_datetime('2022/3/31'), 'vacc_date'] = pd.NaT
r9.loc[r9['vacc_date_boost'] < pd.to_datetime('2021/01/01'), 'vacc_date_boost'] = pd.NaT
r9.loc[r9['vacc_date_boost'] > pd.to_datetime('2022/3/31'), 'vacc_date_boost'] = pd.NaT



#%% CALCULATED VARIABLES: DAILY SURVEYS & DEMOGRAPHICS

#Calculate time in bed variable
(data['TIB'], data['TIB_12']) = calc_sleep_time(data['sleepdiary_bedtime'], 
                                                data['sleepdiary_outofbed'], 
                                                correct_12=True, min_12=3)

#Calculate sleepattempt
(data['sleepattempt'], data['TST_12']) = calc_sleep_time(data['sleepdiary_fallasleep'], 
                                                         data['sleepdiary_waketime'], 
                                                         correct_12=True, min_12=2)

#Calculating TST
data['TST'] = data['sleepattempt'] - (data['night_awakening_time'] + data['sleepdiary_sleeplatency'])/60
#Replace negative TST with missing values
data.loc[data['TST'] < 0, 'TST'] = np.nan

#Sleep efficiency
data['SE'] = data['TST'] / data['TIB']

#Replace apparent mistakes with missing values
data.loc[data['SE'] > 1, ['SE', 'TST', 'TIB', 'sleepattempt']] = np.nan
data.loc[data['TIB'] == 0, ['SE', 'TST', 'TIB', 'sleepattempt']] = np.nan

#Let's check our work!
if output:
    df2csv(join(main_dir, 'data_check', 'sleepvariables.csv'),
           data[['sub_id', 'todays_date', 'redcap_timestamp', *time_vars, 
                 'sleepdiary_sleeplatency', 'night_awakening_time', 'TST', 'TIB', 
                 'SE', 'sleepattempt']])

#PANAS positive scale
panas_pa_vars = ['panas_interested3', 'panas_excited3', 'panas_strong3', 
                 'panas_enthusiastic3', 'panas_proud3', 'panas_alert3', 
                 'panas_inspired3', 'panas_determined3', 'panas_attentive3', 
                 'panas_active3']
data['PANAS_PA'] = data[panas_pa_vars].sum(axis=1, skipna=False)

#PANAS negative scale
panas_na_vars = ['panas_distressed3', 'panas_upset3', 'panas_guilty3', 
                 'panas_scared3', 'panas_hostile3', 'panas_irritable3', 
                 'panas_ashamed3', 'panas_nervous3', 'panas_jittery3', 
                 'panas_afraid3']
data['PANAS_NA'] = data[panas_na_vars].sum(axis=1, skipna=False)

#Worry scale
worry_vars = ['worry_health', 'family_health', 'community_1health',
              'national_health', 'worry_finances']
data['worry_scale'] = data[worry_vars].sum(axis=1, skipna=False)

#PHQ9
data['PHQ9'] = data[depression_vars].sum(axis=1, skipna=False)

#Exercise
data['exercise'] = ((data['sleepdiary_exercise']>0) | 
                    (data[['sleepdiary_exercise___1', 
                           'sleepdiary_exercise___2', 
                           'sleepdiary_exercise___3']]>0).any(axis=1)).astype(int)

#Get fever temperatures in celsius only
data.loc[data['temp_measure']==1, 'fever_temp_C'] = data.loc[data['temp_measure']==1, 'fever_temp']
data.loc[data['temp_measure']==2, 'fever_temp_C'] = (data.loc[data['temp_measure']==2, 'fever_temp'] - 32) * (5/9)
idx = data['fever_temp_C'].notna() & ~data['fever_temp_C'].between(24, 44)
data.loc[idx, ['fever_temp', 'fever_temp_C']] = np.nan

#Normal days: convert various units into days
demo.loc[demo['normal_units']==1, 'normal_days'] = demo.loc[demo['normal_units']==1, 'normal']
demo.loc[demo['normal_units']==2, 'normal_days'] = demo.loc[demo['normal_units']==2, 'normal'] * 7
demo.loc[demo['normal_units']==3, 'normal_days'] = demo.loc[demo['normal_units']==3, 'normal'] * 30.5



#%% CALCULATED VARIABLES: ROUND 1

##### PSQI CALCULATIONS #####
#Duration of sleep
r1['PSQIDURAT'] = np.nan
r1.loc[r1['psqi_4']>=7, 'PSQIDURAT'] = 0
r1.loc[r1['psqi_4']<7, 'PSQIDURAT'] = 1
r1.loc[r1['psqi_4']<6, 'PSQIDURAT'] = 2
r1.loc[r1['psqi_4']<5, 'PSQIDURAT'] = 3
assert r1['PSQIDURAT'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep disturbance
r1.loc[r1['psqi_5j2'].isna() | r1['psqi_5j'].isna(), 'psqi_5j'] = 0
r1['PSQIDISTB'] = r1.loc[:, 'psqi_5b':'psqi_5j'].sum(axis=1, skipna=False)
assert all(r1.loc[r1['PSQIDISTB'].notna(), 'PSQIDISTB'].round() == r1.loc[r1['PSQIDISTB'].notna(), 'PSQIDISTB'])
r1.loc[r1['PSQIDISTB'].between(1, 9), 'PSQIDISTB'] = 1
r1.loc[r1['PSQIDISTB'].between(10, 18), 'PSQIDISTB'] = 2
r1.loc[r1['PSQIDISTB']>18, 'PSQIDISTB'] = 3
assert r1['PSQIDISTB'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep latency
r1['psqi_2NEW'] = np.nan
r1.loc[r1['psqi_2'].between(0, 15), 'psqi_2NEW'] = 0
r1.loc[r1['psqi_2']>15, 'psqi_2NEW'] = 1
r1.loc[r1['psqi_2']>30, 'psqi_2NEW'] = 2
r1.loc[r1['psqi_2']>60, 'psqi_2NEW'] = 3
assert r1['psqi_2NEW'].isin([np.nan, 0, 1, 2, 3]).all()
r1['PSQILATEN'] = r1[['psqi_2NEW', 'psqi_5a']].sum(axis=1, skipna=False)
r1.loc[r1['PSQILATEN'].isin([1, 2]), 'PSQILATEN'] = 1
r1.loc[r1['PSQILATEN'].isin([3, 4]), 'PSQILATEN'] = 2
r1.loc[r1['PSQILATEN'].isin([5, 6]), 'PSQILATEN'] = 3
assert r1['PSQILATEN'].isin([np.nan, 0, 1, 2, 3]).all()
#Days dysfunction due to sleepiness
r1['PSQIDAYDYS'] = r1[['psqi_8', 'psqi_9']].sum(axis=1, skipna=False)
r1.loc[r1['PSQIDAYDYS'].isin([1, 2]), 'PSQIDAYDYS'] = 1
r1.loc[r1['PSQIDAYDYS'].isin([3, 4]), 'PSQIDAYDYS'] = 2
r1.loc[r1['PSQIDAYDYS'].isin([5, 6]), 'PSQIDAYDYS'] = 3
#Sleep efficiency
(r1['PSQI_TIB'], r1['PSQI_TIB_12']) = calc_sleep_time(r1['psqi_1'], r1['psqi_3'],
                                                      correct_12=True, min_12=3)
r1['PSQI_sleep_eff'] = r1['psqi_4'] / r1['PSQI_TIB']
r1.loc[r1['PSQI_sleep_eff']>1, 'PSQI_sleep_eff'] = np.nan
r1.loc[r1['PSQI_sleep_eff']>1, 'PSQI_TIB'] = np.nan
#Sleep effiency category
r1['PSQIHSE'] = np.nan
r1.loc[r1['PSQI_sleep_eff'] >= 0.85, 'PSQIHSE'] = 0
r1.loc[(r1['PSQI_sleep_eff'] >= 0.75) & (r1['PSQI_sleep_eff']<0.85), 'PSQIHSE'] = 1
r1.loc[(r1['PSQI_sleep_eff'] >= 0.65) & (r1['PSQI_sleep_eff']<0.75), 'PSQIHSE'] = 2
r1.loc[(r1['PSQI_sleep_eff']<0.65), 'PSQIHSE'] = 3
#Overall sleep quality
r1['PSQISLPQUAL'] = r1['psqi_6']
#Need meds to sleep
r1['PSQIMEDS'] = r1['psqi_7']
#PSQI total score
r1['PSQI_TOTAL'] = r1[['PSQIDURAT', 'PSQIDISTB', 'PSQILATEN', 'PSQIDAYDYS', 'PSQIHSE', 
                 'PSQISLPQUAL', 'PSQIMEDS']].sum(axis=1, skipna=False)
assert all(r1['PSQI_TOTAL'].between(0, 21) | r1['PSQI_TOTAL'].isna())

#Munich ChronoType Questionnaire calculations
r1['mtq_precovid_freedays']  = 7 - r1['mtq_p2']
r1['mtq_postcovid_freedays'] = 7 - r1['mtq_2']
mtq_renames = {'mtq_p2': 'mtq_precovid_workdays',
               'mtq_p3': 'mtq_precovid_workday_sleeponset',
               'mtq_p4': 'mtq_precovid_workday_sleepend',
               'mtq_p5': 'mtq_precovid_freeday_sleeponset',
               'mtq_p6': 'mtq_precovid_freeday_sleepend',
               'mtq_2':  'mtq_postcovid_workdays',
               'mtq_3':  'mtq_postcovid_workday_sleeponset',
               'mtq_4':  'mtq_postcovid_workday_sleepend',
               'mtq_5':  'mtq_postcovid_freeday_sleeponset',
               'mtq_6':  'mtq_postcovid_freeday_sleepend'}
for col in mtq_renames:
    r1[mtq_renames[col]] = r1[col] 
#Pre-COVID Sleep duration
(r1['mtq_precovid_workday_sleepduration'], r1['mtq_precovid_workday_sleepduration_12']) = calc_sleep_time(r1['mtq_precovid_workday_sleeponset'], 
                                                                                                          r1['mtq_precovid_workday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
(r1['mtq_precovid_freeday_sleepduration'], r1['mtq_precovid_freeday_sleepduration_12']) = calc_sleep_time(r1['mtq_precovid_freeday_sleeponset'], 
                                                                                                          r1['mtq_precovid_freeday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
#Pre-COVID Sleep midpoint
r1['mtq_precovid_workday_sleepmidpoint'] = calc_sleep_midpoint(r1['mtq_precovid_workday_sleeponset'], 
                                                               r1['mtq_precovid_workday_sleepend'], 
                                                               correct_12=True, min_12=2)
r1['mtq_precovid_freeday_sleepmidpoint'] = calc_sleep_midpoint(r1['mtq_precovid_freeday_sleeponset'], 
                                                               r1['mtq_precovid_freeday_sleepend'], 
                                                               correct_12=True, min_12=2)
#Pre-COVID Average sleep duration
r1['mtq_precovid_avg_wk_sleepduration'] = (r1['mtq_precovid_workday_sleepduration']*r1['mtq_precovid_workdays']
                                           + r1['mtq_precovid_freeday_sleepduration']*r1['mtq_precovid_freedays'])/7
#Pre-COVID Chronotype
r1['mtq_precovid_chronotype'] = r1['mtq_precovid_freeday_sleepmidpoint']
idx = r1['mtq_precovid_freeday_sleepduration'] > r1['mtq_precovid_workday_sleepduration']
r1.loc[idx, 'mtq_precovid_chronotype'] -= ((r1.loc[idx, 'mtq_precovid_freeday_sleepduration'] 
                                           - r1.loc[idx, 'mtq_precovid_workday_sleepduration'])/2).apply(pd.Timedelta, unit='hours')
r1['mtq_precovid_chronotype'] = r1['mtq_precovid_chronotype'].round('min')
#Post-COVID sleep duration
(r1['mtq_postcovid_workday_sleepduration'], r1['mtq_postcovid_workday_sleepduration_12']) = calc_sleep_time(r1['mtq_postcovid_workday_sleeponset'], 
                                                                                                            r1['mtq_postcovid_workday_sleepend'],
                                                                                                            correct_12=True, min_12=2)
(r1['mtq_postcovid_freeday_sleepduration'], r1['mtq_postcovid_freeday_sleepduration_12']) = calc_sleep_time(r1['mtq_postcovid_freeday_sleeponset'], 
                                                                                                            r1['mtq_postcovid_freeday_sleepend'],
                                                                                                            correct_12=True, min_12=2)
#Post-COVID Sleep midpoint
r1['mtq_postcovid_workday_sleepmidpoint'] = calc_sleep_midpoint(r1['mtq_postcovid_workday_sleeponset'], 
                                                                r1['mtq_postcovid_workday_sleepend'], 
                                                                correct_12=True, min_12=2)
r1['mtq_postcovid_freeday_sleepmidpoint'] = calc_sleep_midpoint(r1['mtq_postcovid_freeday_sleeponset'], 
                                                                r1['mtq_postcovid_freeday_sleepend'], 
                                                                correct_12=True, min_12=2)
#Post-COVID Average sleep duration
r1['mtq_postcovid_avg_wk_sleepduration'] = (r1['mtq_postcovid_workday_sleepduration']*r1['mtq_postcovid_workdays']
                                            + r1['mtq_postcovid_freeday_sleepduration']*r1['mtq_postcovid_freedays'])/7
#Post-COVID Chronotype
r1['mtq_postcovid_chronotype'] = r1['mtq_postcovid_freeday_sleepmidpoint']
idx = r1['mtq_postcovid_freeday_sleepduration'] > r1['mtq_postcovid_workday_sleepduration']
r1.loc[idx, 'mtq_postcovid_chronotype'] -= ((r1.loc[idx, 'mtq_postcovid_freeday_sleepduration'] 
                                           - r1.loc[idx, 'mtq_postcovid_workday_sleepduration'])/2).apply(pd.Timedelta, unit='hours')
r1['mtq_postcovid_chronotype'] = r1['mtq_postcovid_chronotype'].round('min')


#GAD-7 total
gad_vars = [x for x in r1.columns if 'gad' in x]
r1['gad_7_total'] = r1[gad_vars].sum(axis=1, skipna=False)

#Cognitive Emotion Regulation Questionnaire
r1['CERQ_Self_Blame'] = r1['cerq_14'] + r1['cerq_4']
r1['CERQ_Acceptance'] = r1['cerq_1'] + r1['cerq_5']
r1['CERQ_Catastrophizing'] = r1['cerq_9'] + r1['cerq_17']
r1['CERQ_Other_blame'] = r1['cerq_18'] + r1['cerq_10']
r1['CERQ_Rumination'] = r1['cerq_2'] + r1['cerq_6']
r1['CERQ_Positive_Refocusing'] = r1['cerq_11'] + r1['cerq_7']
r1['CERQ_Refocus_on_Planning'] = r1['cerq_15'] + r1['cerq_12']
r1['CERQ_Positive_Reappraisal'] = r1['cerq_8'] + r1['cerq_3']
r1['CERQ_Putting_into_Perspective'] = r1['cerq_13'] + r1['cerq_16']

#LSAS
LSAS_cols = r1.loc[:, 'telephone_fear':'salesperson2_avoid'].columns
r1['LSAS_Fear_PreCovid'] = r1[[x for x in LSAS_cols 
                                if 'fear' in x and '2' not in x]].sum(axis=1, skipna=False)
r1['LSAS_Anxiety_PreCovid'] = r1[[x for x in LSAS_cols 
                                   if 'avoid' in x and '2' not in x]].sum(axis=1, skipna=False)
r1['LSAS_TOTAL_PreCovid'] = r1['LSAS_Fear_PreCovid'] + r1['LSAS_Anxiety_PreCovid']
r1['LSAS_Fear_PostCovid'] = r1[[x for x in LSAS_cols 
                                 if 'fear' in x and '2' in x]].sum(axis=1, skipna=False)
r1['LSAS_Anxiety_PostCovid'] = r1[[x for x in LSAS_cols 
                                    if 'avoid' in x and '2' in x]].sum(axis=1, skipna=False)
r1['LSAS_TOTAL_PostCovid'] = r1['LSAS_Fear_PostCovid'] + r1['LSAS_Anxiety_PostCovid']

#Big 5 calculations
r1['Big_5_Extraversion'] = + (6 - r1['big5_1'])+ r1['big5_6']+ r1['big5_11']+ r1['big5_16']+ (6 - r1['big5_21'])+ (6 - r1['big5_26'])
r1['Big_5_Agreeableness'] = + r1['big5_2']+ (6 - r1['big5_7'])+ r1['big5_12']+ (6 - r1['big5_17'])+ r1['big5_22']+ (6 - r1['big5_27'])
r1['Big_5_Conscientiousness'] = + (6 - r1['big5_3'])+ (6 - r1['big5_8'])+ r1['big5_13']+ r1['big5_18']+ r1['big5_23']+ (6 - r1['big5_28'])
r1['Big_5_Negative_Emotionality'] = + r1['big5_4']+ r1['big5_9']+ (6 - r1['big5_14'])+ (6 - r1['big5_19'])+ (6 - r1['big5_24'])+ r1['big5_29']
r1['Big_5_Open_Mindedness'] = + r1['big5_5']+ (6 - r1['big5_10'])+ r1['big5_15']+ (6 - r1['big5_20'])+ r1['big5_25']+ (6 - r1['big5_30'])
r1['Big_5_Sociability'] = + (6 - r1['big5_1'])+ r1['big5_16']
r1['Big_5_Assertiveness'] = + r1['big5_6']+ (6 - r1['big5_21'])
r1['Big_5_Energy_Level'] = + r1['big5_11']+ (6 - r1['big5_26'])
r1['Big_5_Compassion'] = + r1['big5_2']+ (6 - r1['big5_17'])
r1['Big_5_Respectfulness'] = + (6 - r1['big5_7'])+ r1['big5_22']
r1['Big_5_Trust'] = + r1['big5_12']+ (6 - r1['big5_27'])
r1['Big_5_Organization'] = + (6 - r1['big5_3'])+ r1['big5_18']
r1['Big_5_Productiveness'] = + (6 - r1['big5_8'])+ r1['big5_23']
r1['Big_5_Responsibility'] = + r1['big5_13']+ (6 - r1['big5_28'])
r1['Big_5_Anxiety'] = + r1['big5_4']+ (6 - r1['big5_19'])
r1['Big_5_Depression'] = + r1['big5_9']+ (6 - r1['big5_24'])
r1['Big_5_Emotional_Volatility'] = + (6 - r1['big5_14'])+ r1['big5_29']
r1['Big_5_Aesthetic_Sensitivity'] = + r1['big5_5']+ (6 - r1['big5_20'])
r1['Big_5_Intellectual_Curiosity'] = + (6 - r1['big5_10'])+ r1['big5_25']
r1['Big_5_Creative_Imagination'] = + r1['big5_15']+ (6 - r1['big5_30'])



#%% CALCULATED VARIABLES: ROUND 2

isi_vars = [x for x in r2.columns if re.fullmatch('isi_\d', x)]
r2['ISI_Total'] = r2[isi_vars].sum(axis=1, skipna=False)

meq_vars = [x for x in r2.columns if re.fullmatch('meq_\d', x)]
r2['MEQ_Total'] = r2[meq_vars].sum(axis=1, skipna=False)

r2['TEQ_TOTAL'] =  (r2[['teq_1', 'teq_3', 'teq_5', 'teq_6', 'teq_8', 'teq_9', 'teq_13', 'teq_16']].sum(axis=1, skipna=False)
                    + (4 - r2[['teq_2', 'teq_4', 'teq_7', 'teq_10', 'teq_11', 'teq_12', 'teq_14', 'teq_15']]).sum(axis=1, skipna=False))

r2['PSS_TOTAL'] = (r2[['pss_1', 'pss_2', 'pss_3', 'pss_6', 'pss_9', 'pss_10']].sum(axis=1, skipna=False)
                   + (4 - r2[['pss_4', 'pss_5', 'pss_7', 'pss_8']]).sum(axis=1, skipna=False))



#%% CALCULATED VARIABLES: ROUND 3

#Brief self-control scale
r3['BSCS_Total'] = (r3[['bscs_1', 'bscs_6', 'bscs_8', 'bscs_11']].sum(axis=1, skipna=False) 
                    + (6 - r3[['bscs_2', 'bscs_3', 'bscs_4', 'bscs_5', 'bscs_7', 'bscs_9', 'bscs_10', 'bscs_12', 'bscs_13']]).sum(axis=1, skipna=False))

#Short impulsive behavior scale
r3['SUPPS_Neg_Urg'] = r3[['sibs_4', 'sibs_7', 'sibs_12', 'sibs_17']].sum(axis=1, skipna=False) 
r3['SUPPS_Lack_Pers'] = r3[['sibs_5', 'sibs_8', 'sibs_11', 'sibs_16']].sum(axis=1, skipna=False)
r3['SUPPS_Lack_Premed'] = r3[['sibs_1', 'sibs_6', 'sibs_13', 'sibs_19']].sum(axis=1, skipna=False) 
r3['SUPPS_Sen_Seek'] = r3[['sibs_3', 'sibs_9', 'sibs_14', 'sibs_18']].sum(axis=1, skipna=False) 
r3['SUPPS_Pos_Urg'] = r3[['sibs_2', 'sibs_10', 'sibs_15', 'sibs_20']].sum(axis=1, skipna=False) 

#Intolerance of uncertainty
r3['IU_PA'] = r3.loc[:, 'iu_1':'iu_7'].sum(axis=1, skipna=False)
r3['IU_IA'] = r3.loc[:, 'iu_8':'iu_12'].sum(axis=1, skipna=False)
r3['IU_Total'] = r3.loc[:, 'iu_1':'iu_12'].sum(axis=1, skipna=False)

#Emotion regulation questionnaire
r3['ERQ_Cog_Reapp'] = r3[['erq_1', 'erq_3', 'erq_5', 'erq_7', 'erq_8', 'erq_10']].sum(axis=1, skipna=False)
r3['ERQ_Exp_Supp'] = r3[['erq_2', 'erq_4', 'erq_6', 'erq_9']].sum(axis=1, skipna=False)

#COVID positive
r3['COVID_Pos_Total'] = r3[[x for x in r3.columns if re.fullmatch('covpos_\d', x)]].sum(axis=1, skipna=False)

#Positive social behavior
r3['Pos_Social_Behavior_Total'] = r3[[x for x in r3.columns if re.fullmatch('sd_\d*', x)]].sum(axis=1, skipna=False)

#Dream lucidity scale
r3['Lucidity_Insight'] = r3[['luc_1', 'luc_3', 'luc_8', 'luc_9', 'luc_16', 'luc_19']].sum(axis=1, skipna=False)
r3['Lucidity_Control'] = r3[['luc_4', 'luc_6', 'luc_10', 'luc_14', 'luc_23']].sum(axis=1, skipna=False)
r3['Lucidity_Thought'] = r3[['luc_5', 'luc_12', 'luc_22']].sum(axis=1, skipna=False)	
r3['Lucidity_realism'] = r3[['luc_7', 'luc_17', 'luc_20']].sum(axis=1, skipna=False)
r3['Lucidity_Memory'] = r3[['luc_2', 'luc_13', 'luc_18', 'luc_24']].sum(axis=1, skipna=False)
r3['Lucidity_Dissociation'] = r3[['luc_11', 'luc_15', 'luc_21']].sum(axis=1, skipna=False)
r3['Lucidity_Neg_emotion'] = r3[['luc_26', 'luc_28']].sum(axis=1, skipna=False)	
r3['Lucidity_Pos_emotion'] = r3[['luc_25', 'luc_27']].sum(axis=1, skipna=False)

#Dream PANAS scale
r3['Dream_PANAS_PA'] = r3.loc[:, 'pandr_1':'pandr_9'].sum(axis=1, skipna=False)
r3['Dream_PANAS_NA'] = r3.loc[:, 'pandr_10':'pandr_18'].sum(axis=1, skipna=False)

#Mindwandering scale
r3['MW_Deliberate'] = r3[['mw_1', 'mw_2', 'mw_3', 'mw_7']].sum(axis=1, skipna=False)
r3['MW_Spontaneous'] = r3[['mw_4', 'mw_5', 'mw_6', 'mw_8']].sum(axis=1, skipna=False)



#%% CALCULATED VARIABLES: ROUND 4

##### Fall_PSQI CALCULATIONS #####
#Duration of sleep
r4['fall_PSQIDURAT'] = np.nan
r4.loc[r4['fall_psqi_4']>=7, 'fall_PSQIDURAT'] = 0
r4.loc[r4['fall_psqi_4']<7, 'fall_PSQIDURAT'] = 1
r4.loc[r4['fall_psqi_4']<6, 'fall_PSQIDURAT'] = 2
r4.loc[r4['fall_psqi_4']<5, 'fall_PSQIDURAT'] = 3
assert r4['fall_PSQIDURAT'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep disturbance
r4.loc[r4['fall_psqi_5j2'].isna() | r4['fall_psqi_5j'].isna(), 'fall_psqi_5j'] = 0
r4['fall_PSQIDISTB'] = r4.loc[:, 'fall_psqi_5b':'fall_psqi_5j'].sum(axis=1, skipna=False)
r4.loc[r4['fall_PSQIDISTB'].between(1, 9), 'fall_PSQIDISTB'] = 1
r4.loc[r4['fall_PSQIDISTB'].between(10, 18), 'fall_PSQIDISTB'] = 2
r4.loc[r4['fall_PSQIDISTB']>18, 'fall_PSQIDISTB'] = 3
assert r4['fall_PSQIDISTB'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep latency
r4['fall_psqi_2NEW'] = np.nan
r4.loc[r4['fall_psqi_2'].between(0, 15), 'fall_psqi_2NEW'] = 0
r4.loc[r4['fall_psqi_2']>15, 'fall_psqi_2NEW'] = 1
r4.loc[r4['fall_psqi_2']>30, 'fall_psqi_2NEW'] = 2
r4.loc[r4['fall_psqi_2']>60, 'fall_psqi_2NEW'] = 3
assert r4['fall_psqi_2NEW'].isin([np.nan, 0, 1, 2, 3]).all()
r4['fall_PSQILATEN'] = r4[['fall_psqi_2NEW', 'fall_psqi_5a']].sum(axis=1, skipna=False)
r4.loc[r4['fall_PSQILATEN'].isin([1, 2]), 'fall_PSQILATEN'] = 1
r4.loc[r4['fall_PSQILATEN'].isin([3, 4]), 'fall_PSQILATEN'] = 2
r4.loc[r4['fall_PSQILATEN'].isin([5, 6]), 'fall_PSQILATEN'] = 3
assert r4['fall_PSQILATEN'].isin([np.nan, 0, 1, 2, 3]).all()
#Days dysfunction due to sleepiness
r4['fall_PSQIDAYDYS'] = r4[['fall_psqi_8', 'fall_psqi_9']].sum(axis=1, skipna=False)
r4.loc[r4['fall_PSQIDAYDYS'].isin([1, 2]), 'fall_PSQIDAYDYS'] = 1
r4.loc[r4['fall_PSQIDAYDYS'].isin([3, 4]), 'fall_PSQIDAYDYS'] = 2
r4.loc[r4['fall_PSQIDAYDYS'].isin([5, 6]), 'fall_PSQIDAYDYS'] = 3
#Sleep efficiency
(r4['fall_PSQI_TIB'], r4['fall_PSQI_TIB_12']) = calc_sleep_time(r4['fall_psqi_1'], r4['fall_psqi_3'],
                                                                correct_12=True, min_12=3)
r4['fall_PSQI_sleep_eff'] = r4['fall_psqi_4'] / r4['fall_PSQI_TIB']
r4.loc[r4['fall_PSQI_sleep_eff']>1, 'fall_PSQI_sleep_eff'] = np.nan
r4.loc[r4['fall_PSQI_sleep_eff']>1, 'fall_PSQI_TIB'] = np.nan
#Sleep effiency category
r4['fall_PSQIHSE'] = np.nan
r4.loc[r4['fall_PSQI_sleep_eff'] >= 0.85, 'fall_PSQIHSE'] = 0
r4.loc[(r4['fall_PSQI_sleep_eff'] >= 0.75) & (r4['fall_PSQI_sleep_eff']<0.85), 'fall_PSQIHSE'] = 1
r4.loc[(r4['fall_PSQI_sleep_eff'] >= 0.65) & (r4['fall_PSQI_sleep_eff']<0.75), 'fall_PSQIHSE'] = 2
r4.loc[(r4['fall_PSQI_sleep_eff']<0.65), 'fall_PSQIHSE'] = 3
#Overall sleep quality
r4['fall_PSQISLPQUAL'] = r4['fall_psqi_6']
#Need meds to sleep
r4['fall_PSQIMEDS'] = r4['fall_psqi_7']
#fall_PSQI total score
r4['fall_PSQI_TOTAL'] = r4[['fall_PSQIDURAT', 'fall_PSQIDISTB', 'fall_PSQILATEN', 
                            'fall_PSQIDAYDYS', 'fall_PSQIHSE', 'fall_PSQISLPQUAL', 
                            'fall_PSQIMEDS']].sum(axis=1, skipna=False)
assert all(r4['fall_PSQI_TOTAL'].between(0, 21) | r4['fall_PSQI_TOTAL'].isna())

#ISI
r4['fall_ISI_Total'] = r4[[col for col in r4.columns if 'fall_isi' in col]].sum(axis=1, skipna=False)

##### Munich ChronoType Questionnaire calculations #####
r4['fall_mtq_freedays'] = 7 - r4['fall_mtq_2']
fall_mtq_renames = {'fall_mtq_2': 'fall_mtq_workdays',
                    'fall_mtq_3': 'fall_mtq_workday_sleeponset',
                    'fall_mtq_4': 'fall_mtq_workday_sleepend',
                    'fall_mtq_5': 'fall_mtq_freeday_sleeponset',
                    'fall_mtq_6': 'fall_mtq_freeday_sleepend'}
for col in fall_mtq_renames:
    r4[fall_mtq_renames[col]] = r4[col]
#Sleep duration
(r4['fall_mtq_workday_sleepduration'], r4['fall_mtq_workday_sleepduration_12']) = calc_sleep_time(r4['fall_mtq_workday_sleeponset'], 
                                                                                                  r4['fall_mtq_workday_sleepend'],
                                                                                                  correct_12=True, min_12=2)
(r4['fall_mtq_freeday_sleepduration'], r4['fall_mtq_freeday_sleepduration_12']) = calc_sleep_time(r4['fall_mtq_freeday_sleeponset'], 
                                                                                                  r4['fall_mtq_freeday_sleepend'],
                                                                                                  correct_12=True, min_12=2)
#Sleep midpoint
r4['fall_mtq_workday_sleepmidpoint'] = calc_sleep_midpoint(r4['fall_mtq_workday_sleeponset'],
                                                           r4['fall_mtq_workday_sleepend'], 
                                                           correct_12=True, min_12=2)
r4['fall_mtq_freeday_sleepmidpoint'] = calc_sleep_midpoint(r4['fall_mtq_freeday_sleeponset'], 
                                                           r4['fall_mtq_freeday_sleepend'], 
                                                           correct_12=True, min_12=2)
#Average sleep duration
r4['fall_mtq_avg_wk_sleepduration'] = (r4['fall_mtq_workday_sleepduration']*r4['fall_mtq_workdays']
                                       + r4['fall_mtq_freeday_sleepduration']*r4['fall_mtq_freedays'])/7
#Chronotype
r4['fall_mtq_chronotype'] = r4['fall_mtq_freeday_sleepmidpoint']
idx = r4['fall_mtq_freeday_sleepduration'] > r4['fall_mtq_workday_sleepduration']
r4.loc[idx, 'fall_mtq_chronotype'] -= ((r4.loc[idx, 'fall_mtq_freeday_sleepduration']
                                        - r4.loc[idx, 'fall_mtq_workday_sleepduration'])/2).apply(pd.Timedelta, unit='hours')
r4['fall_mtq_chronotype'] = r4['fall_mtq_chronotype'].round('min')
r4.loc[r4['fall_mtq_chronotype']<pd.Timedelta(0), 'fall_mtq_chronotype'] = pd.NaT

#GAD
r4['fall_gad_7_total'] = r4[[col for col in r4.columns if 'fall_gad' in col]].sum(axis=1, skipna=False)



#%% CALCULATED VARIABLES: ROUND 5

##### PSQI CALCULATIONS #####
#Duration of sleep
r5['Feb21_PSQIDURAT'] = np.nan
r5.loc[r5['psqi_4']>=7, 'Feb21_PSQIDURAT'] = 0
r5.loc[r5['psqi_4']<7, 'Feb21_PSQIDURAT'] = 1
r5.loc[r5['psqi_4']<6, 'Feb21_PSQIDURAT'] = 2
r5.loc[r5['psqi_4']<5, 'Feb21_PSQIDURAT'] = 3
assert r5['Feb21_PSQIDURAT'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep disturbance
r5.loc[r5['psqi_5j2'].isna() | r5['psqi_5j'].isna(), 'psqi_5j'] = 0
r5['Feb21_PSQIDISTB'] = r5.loc[:, 'psqi_5b':'psqi_5j'].sum(axis=1, skipna=False)
assert all(r5.loc[r5['Feb21_PSQIDISTB'].notna(), 'Feb21_PSQIDISTB'].round() == r5.loc[r5['Feb21_PSQIDISTB'].notna(), 'Feb21_PSQIDISTB'])
r5.loc[r5['Feb21_PSQIDISTB'].between(1, 9), 'Feb21_PSQIDISTB'] = 1
r5.loc[r5['Feb21_PSQIDISTB'].between(10, 18), 'Feb21_PSQIDISTB'] = 2
r5.loc[r5['Feb21_PSQIDISTB']>18, 'Feb21_PSQIDISTB'] = 3
assert r5['Feb21_PSQIDISTB'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep latency
r5['Feb21_psqi_2NEW'] = np.nan
r5.loc[r5['psqi_2'].between(0, 15), 'Feb21_psqi_2NEW'] = 0
r5.loc[r5['psqi_2']>15, 'Feb21_psqi_2NEW'] = 1
r5.loc[r5['psqi_2']>30, 'Feb21_psqi_2NEW'] = 2
r5.loc[r5['psqi_2']>60, 'Feb21_psqi_2NEW'] = 3
assert r5['Feb21_psqi_2NEW'].isin([np.nan, 0, 1, 2, 3]).all()
r5['Feb21_PSQILATEN'] = r5[['Feb21_psqi_2NEW', 'psqi_5a']].sum(axis=1, skipna=False)
r5.loc[r5['Feb21_PSQILATEN'].isin([1, 2]), 'Feb21_PSQILATEN'] = 1
r5.loc[r5['Feb21_PSQILATEN'].isin([3, 4]), 'Feb21_PSQILATEN'] = 2
r5.loc[r5['Feb21_PSQILATEN'].isin([5, 6]), 'Feb21_PSQILATEN'] = 3
assert r5['Feb21_PSQILATEN'].isin([np.nan, 0, 1, 2, 3]).all()
#Days dysfunction due to sleepiness
r5['Feb21_PSQIDAYDYS'] = r5[['psqi_8', 'psqi_9']].sum(axis=1, skipna=False)
r5.loc[r5['Feb21_PSQIDAYDYS'].isin([1, 2]), 'Feb21_PSQIDAYDYS'] = 1
r5.loc[r5['Feb21_PSQIDAYDYS'].isin([3, 4]), 'Feb21_PSQIDAYDYS'] = 2
r5.loc[r5['Feb21_PSQIDAYDYS'].isin([5, 6]), 'Feb21_PSQIDAYDYS'] = 3
#Sleep efficiency
(r5['PSQI_TIB'], r5['PSQI_TIB_12']) = calc_sleep_time(r5['psqi_1'], r5['psqi_3'],
                                                      correct_12=True, min_12=3)
r5['Feb21_PSQI_sleep_eff'] = r5['psqi_4'] / r5['PSQI_TIB']
r5.loc[r5['Feb21_PSQI_sleep_eff']>1, 'Feb21_PSQI_sleep_eff'] = np.nan
r5.loc[r5['Feb21_PSQI_sleep_eff']>1, 'PSQI_TIB'] = np.nan
#Sleep effiency category
r5['Feb21_PSQIHSE'] = np.nan
r5.loc[r5['Feb21_PSQI_sleep_eff'] >= 0.85, 'Feb21_PSQIHSE'] = 0
r5.loc[(r5['Feb21_PSQI_sleep_eff'] >= 0.75) & (r5['Feb21_PSQI_sleep_eff']<0.85), 'Feb21_PSQIHSE'] = 1
r5.loc[(r5['Feb21_PSQI_sleep_eff'] >= 0.65) & (r5['Feb21_PSQI_sleep_eff']<0.75), 'Feb21_PSQIHSE'] = 2
r5.loc[(r5['Feb21_PSQI_sleep_eff']<0.65), 'Feb21_PSQIHSE'] = 3
#Overall sleep quality
r5['Feb21_PSQISLPQUAL'] = r5['psqi_6']
#Need meds to sleep
r5['Feb21_PSQIMEDS'] = r5['psqi_7']
#PSQI total score
r5['Feb21_PSQI_TOTAL'] = r5[['Feb21_PSQIDURAT', 'Feb21_PSQIDISTB', 'Feb21_PSQILATEN', 
                             'Feb21_PSQIDAYDYS', 'Feb21_PSQIHSE', 'Feb21_PSQISLPQUAL', 'Feb21_PSQIMEDS']].sum(axis=1, skipna=False)
assert all(r5['Feb21_PSQI_TOTAL'].between(0, 21) | r5['Feb21_PSQI_TOTAL'].isna())

#Munich ChronoType Questionnaire calculations
r5['Feb21_mtq_freedays']  = 7 - r5['mtq_2']
mtq_renames = {'mtq_2': 'Feb21_mtq_workdays',
               'mtq_3': 'Feb21_mtq_workday_sleeponset',
               'mtq_p8': 'Feb21_mtq_workday_sleepend',
               'mtq_p9': 'Feb21_mtq_freeday_sleeponset',
               'mtq_p10': 'Feb21_mtq_freeday_sleepend'}
for col in mtq_renames:
    r5[mtq_renames[col]] = r5[col] 
#Sleep duration
(r5['Feb21_mtq_workday_sleepduration'], r5['Feb21_mtq_workday_sleepduration_12']) = calc_sleep_time(r5['Feb21_mtq_workday_sleeponset'], 
                                                                                                          r5['Feb21_mtq_workday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
(r5['Feb21_mtq_freeday_sleepduration'], r5['Feb21_mtq_freeday_sleepduration_12']) = calc_sleep_time(r5['Feb21_mtq_freeday_sleeponset'], 
                                                                                                          r5['Feb21_mtq_freeday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
#Sleep midpoint
r5['Feb21_mtq_workday_sleepmidpoint'] = calc_sleep_midpoint(r5['Feb21_mtq_workday_sleeponset'], 
                                                               r5['Feb21_mtq_workday_sleepend'], 
                                                               correct_12=True, min_12=2)
r5['Feb21_mtq_freeday_sleepmidpoint'] = calc_sleep_midpoint(r5['Feb21_mtq_freeday_sleeponset'], 
                                                               r5['Feb21_mtq_freeday_sleepend'], 
                                                               correct_12=True, min_12=2)
#Average sleep duration
r5['Feb21_mtq_avg_wk_sleepduration'] = (r5['Feb21_mtq_workday_sleepduration']*r5['Feb21_mtq_workdays']
                                           + r5['Feb21_mtq_freeday_sleepduration']*r5['Feb21_mtq_freedays'])/7
#Chronotype
r5['Feb21_mtq_chronotype'] = r5['Feb21_mtq_freeday_sleepmidpoint']
idx = r5['Feb21_mtq_freeday_sleepduration'] > r5['Feb21_mtq_workday_sleepduration']
r5.loc[idx, 'Feb21_mtq_chronotype'] -= ((r5.loc[idx, 'Feb21_mtq_freeday_sleepduration'] 
                                           - r5.loc[idx, 'Feb21_mtq_workday_sleepduration'])/2).apply(pd.Timedelta, unit='hours')
r5['Feb21_mtq_chronotype'] = r5['Feb21_mtq_chronotype'].round('min')
print('R5: dropping %d negative MTQ chronotypes' % sum(r5['Feb21_mtq_chronotype'] < pd.Timedelta(0)))
r5.loc[r5['Feb21_mtq_chronotype'] < pd.Timedelta(0), 'Feb21_mtq_chronotype'] = pd.NaT

#ISI
isi_vars = [x for x in r5.columns if re.fullmatch('isi_\d', x)]
r5['Feb21_ISI_Total'] = r5[isi_vars].sum(axis=1, skipna=False)

#PROMIS
r5['Feb21_PROMIS_Total'] = r5[['promis_1', 'promis_2', 'promis_3', 'promis_4', 
                               'promis_5', 'promis_6']].sum(axis=1, skipna=False) + (6 - r5['promis_7'])

#PSS
r5['Feb21_PSS_TOTAL'] = (r5[['pss_1', 'pss_2', 'pss_3', 'pss_6', 'pss_9', 'pss_10']].sum(axis=1, skipna=False)
                         + (4 - r5[['pss_4', 'pss_5', 'pss_7', 'pss_8']]).sum(axis=1, skipna=False))

#GAD-7				   
r5['Feb21_gad_7_total'] = r5[[col for col in r5.columns if col.startswith('gad_')]].sum(axis=1, skipna=False)

r5['MMQ_Satisfaction_Feb21'] = (r5[['mmq_2', 'mmq_4', 'mmq_5', 'mmq_7', 'mmq_8', 
                                    'mmq_10', 'mmq_11', 'mmq_14', 'mmq_15', 
                                    'mmq_16', 'mmq_18']].sum(axis=1, skipna=False) +
                                (4 - r5[['mmq_1', 'mmq_3', 'mmq_6', 'mmq_9', 
                                         'mmq_12', 'mmq_13', 'mmq_17']]).sum(axis=1, skipna=False))

r5['Feb21_COVID_Pos_Total'] = r5[[col for col in r5 if 'covpos_' in col]].sum(axis=1, skipna=False)

#Interpersonal reactivity index
r5['IRI_Perspective_Taking'] = 	(r5[['iri_8', 'iri_11', 'iri_21', 'iri_25', 'iri_28']].sum(axis=1, skipna=False) 
                                 + (4 - r5[['iri_3', 'iri_15']]).sum(axis=1, skipna=False))
r5['IRI_Fantasy'] = (r5[['iri_1', 'iri_5', 'iri_16', 'iri_23', 'iri_26']].sum(axis=1, skipna=False) 
                     + (4 - r5[['iri_7', 'iri_12']]).sum(axis=1, skipna=False))
r5['IRI_Empathic_Concern'] = (r5[['iri_2', 'iri_9', 'iri_20', 'iri_22']].sum(axis=1, skipna=False) 
                              + (4 - r5[['iri_4', 'iri_14', 'iri_18']]).sum(axis=1, skipna=False))
r5['IRI_Personal_Distress'] = (r5[['iri_6', 'iri_10', 'iri_17', 'iri_24', 'iri_27']].sum(axis=1, skipna=False) 
                               + (4 - r5[['iri_13', 'iri_19']]).sum(axis=1, skipna=False))

#Personality inventory for DSM
r5['PID_Total_Raw_Score'] = r5[[col for col in r5.columns if 'pid_' in col]].sum(axis=1, skipna=False)
r5['PID_Total_Negative_Affect'] = r5[['pid_8', 'pid_9', 'pid_10', 'pid_11', 'pid_15']].sum(axis=1, skipna=False)
r5['PID_Total_Detachment'] = r5[['pid_4', 'pid_13', 'pid_14', 'pid_16', 'pid_18']].sum(axis=1, skipna=False)
r5['PID_Total_Antagonism'] = r5[['pid_17', 'pid_19', 'pid_20', 'pid_22', 'pid_25']].sum(axis=1, skipna=False)
r5['PID_Total_Disinhibition'] = r5[['pid_1', 'pid_2', 'pid_3', 'pid_5', 'pid_6']].sum(axis=1, skipna=False)
r5['PID_Total_Psychoticisim'] = r5[['pid_7', 'pid_12', 'pid_21', 'pid_23', 'pid_24']].sum(axis=1, skipna=False)

#John Henryism Active Coping Scale
r5['JHACS_TOTAL'] = r5[[col for col in r5.columns if 'jhacs_' in col]].sum(axis=1, skipna=False)

#Adverse Childhood Events
r5['ACE_Original_10'] =  r5[['ace_%d' % i for i in range(1, 11)]].sum(axis=1, skipna=False)
r5['ACE_Added_8'] = r5[['ace_%d' % i for i in range(11, 19)]].sum(axis=1, skipna=False)
r5['ACE_TOTAL'] =  r5[[col for col in r5.columns if 'ace_' in col]].sum(axis=1, skipna=False)



#%% CALCULATED VARIABLES: ROUND 6

#LSAS
r6['LSAS_Fear'] = r6[[col for col in r6.columns if '_fear' in col]].sum(axis=1, skipna=False)
r6['LSAS_Anxiety'] = r6[[col for col in r6.columns if '_avoid' in col]].sum(axis=1, skipna=False)
r6['LSAS_TOTAL'] = r6['LSAS_Fear'] + r6['LSAS_Anxiety']



#%% CALCULATED VARIABLES: ROUND 7

r7['FIRST_Total'] = r7[[x for x in r7.columns if x.startswith('Q89')]].sum(axis=1, skipna=False)
r7['Epworth_Total'] = r7[[x for x in r7.columns if x.startswith('Q90')]].sum(axis=1, skipna=False)
r7['CDRISC_10_Total'] = r7[[x for x in r7.columns if x.startswith('Q91')]].sum(axis=1, skipna=False)
r7['CDRISC_flexibility'] = 	r7[['Q91_1', 'Q91_5']].sum(axis=1, skipna=False)
r7['CDRISC_self_efficacy']	= r7[['Q91_2', 'Q91_4', 'Q91_9']].sum(axis=1, skipna=False)
r7['CDRISC_regulate_emotions'] = r7['Q91_10']
r7['CDRISC_optimism'] = r7[['Q91_3', 'Q91_6', 'Q91_8']].sum(axis=1, skipna=False)
r7['CDRISC_cognitive_focus'] = r7['Q91_7']
r7['PSAS_Total'] = r7[[x for x in r7.columns if x.startswith('Q92')]].sum(axis=1, skipna=False)
r7['PSAS_Somatic'] = r7.loc[:, 'Q92_1':'Q92_8'].sum(axis=1, skipna=False)
r7['PSAS_Cognitive'] = r7.loc[:, 'Q92_9':'Q92_16'].sum(axis=1, skipna=False)
r7['MMQ_Ability_total'] = r7[[x for x in r7.columns if x.startswith('Q93')]].sum(axis=1, skipna=False)



#%% CALCULATED VARIABLES: ROUND 8

#Negative memory
r8['NEG_MEMORY_Total'] = (r8[['mem_oct1', 'mem_oct4', 'mem_oct5']].sum(axis=1, skipna=False) 
                          + (6 - r8[['mem_oct2', 'mem_oct3', 'mem_oct6']]).sum(axis=1, skipna=False))

#Nostalgia questions
r8['Nostalgia_Total'] = r8[[x for x in r8.columns 
                            if x.startswith('pine')]].sum(axis=1, skipna=False)

##### PSQI CALCULATIONS #####
#Duration of sleep
r8['Oct21_PSQIDURAT'] = np.nan
r8.loc[r8['psqi_4']>=7, 'Oct21_PSQIDURAT'] = 0
r8.loc[r8['psqi_4']<7, 'Oct21_PSQIDURAT'] = 1
r8.loc[r8['psqi_4']<6, 'Oct21_PSQIDURAT'] = 2
r8.loc[r8['psqi_4']<5, 'Oct21_PSQIDURAT'] = 3
assert r8['Oct21_PSQIDURAT'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep disturbance
r8.loc[r8['psqi_5j2'].isna() | r8['psqi_5j'].isna(), 'psqi_5j'] = 0
r8['Oct21_PSQIDISTB'] = r8.loc[:, 'psqi_5b':'psqi_5j'].sum(axis=1, skipna=False)
assert all(r8.loc[r8['Oct21_PSQIDISTB'].notna(), 'Oct21_PSQIDISTB'].round() == r8.loc[r8['Oct21_PSQIDISTB'].notna(), 'Oct21_PSQIDISTB'])
r8.loc[r8['Oct21_PSQIDISTB'].between(1, 9), 'Oct21_PSQIDISTB'] = 1
r8.loc[r8['Oct21_PSQIDISTB'].between(10, 18), 'Oct21_PSQIDISTB'] = 2
r8.loc[r8['Oct21_PSQIDISTB']>18, 'Oct21_PSQIDISTB'] = 3
assert r8['Oct21_PSQIDISTB'].isin([np.nan, 0, 1, 2, 3]).all()
#Sleep latency
r8['Oct21_psqi_2NEW'] = np.nan
r8.loc[r8['psqi_2'].between(0, 15), 'Oct21_psqi_2NEW'] = 0
r8.loc[r8['psqi_2']>15, 'Oct21_psqi_2NEW'] = 1
r8.loc[r8['psqi_2']>30, 'Oct21_psqi_2NEW'] = 2
r8.loc[r8['psqi_2']>60, 'Oct21_psqi_2NEW'] = 3
assert r8['Oct21_psqi_2NEW'].isin([np.nan, 0, 1, 2, 3]).all()
r8['Oct21_PSQILATEN'] = r8[['Oct21_psqi_2NEW', 'psqi_5a']].sum(axis=1, skipna=False)
r8.loc[r8['Oct21_PSQILATEN'].isin([1, 2]), 'Oct21_PSQILATEN'] = 1
r8.loc[r8['Oct21_PSQILATEN'].isin([3, 4]), 'Oct21_PSQILATEN'] = 2
r8.loc[r8['Oct21_PSQILATEN'].isin([5, 6]), 'Oct21_PSQILATEN'] = 3
assert r8['Oct21_PSQILATEN'].isin([np.nan, 0, 1, 2, 3]).all()
#Days dysfunction due to sleepiness
r8['Oct21_PSQIDAYDYS'] = r8[['psqi_8', 'psqi_9']].sum(axis=1, skipna=False)
r8.loc[r8['Oct21_PSQIDAYDYS'].isin([1, 2]), 'Oct21_PSQIDAYDYS'] = 1
r8.loc[r8['Oct21_PSQIDAYDYS'].isin([3, 4]), 'Oct21_PSQIDAYDYS'] = 2
r8.loc[r8['Oct21_PSQIDAYDYS'].isin([5, 6]), 'Oct21_PSQIDAYDYS'] = 3
#Sleep efficiency
(r8['Oct21_PSQI_TIB'], r8['Oct21_PSQI_TIB_12']) = calc_sleep_time(r8['psqi_1'], r8['psqi_3'],
                                                      correct_12=True, min_12=3)
r8['Oct21_PSQI_sleep_eff'] = r8['psqi_4'] / r8['Oct21_PSQI_TIB']
r8.loc[r8['Oct21_PSQI_sleep_eff']>1, 'Oct21_PSQI_sleep_eff'] = np.nan
r8.loc[r8['Oct21_PSQI_sleep_eff']>1, 'Oct21_PSQI_TIB'] = np.nan
#Sleep effiency category
r8['Oct21_PSQIHSE'] = np.nan
r8.loc[r8['Oct21_PSQI_sleep_eff'] >= 0.85, 'Oct21_PSQIHSE'] = 0
r8.loc[(r8['Oct21_PSQI_sleep_eff'] >= 0.75) & (r8['Oct21_PSQI_sleep_eff']<0.85), 'Oct21_PSQIHSE'] = 1
r8.loc[(r8['Oct21_PSQI_sleep_eff'] >= 0.65) & (r8['Oct21_PSQI_sleep_eff']<0.75), 'Oct21_PSQIHSE'] = 2
r8.loc[(r8['Oct21_PSQI_sleep_eff']<0.65), 'Oct21_PSQIHSE'] = 3
#Overall sleep quality
r8['Oct21_PSQISLPQUAL'] = r8['psqi_6']
#Need meds to sleep
r8['Oct21_PSQIMEDS'] = r8['psqi_7']
#PSQI total score
r8['Oct21_PSQI_TOTAL'] = r8[['Oct21_PSQIDURAT', 'Oct21_PSQIDISTB', 'Oct21_PSQILATEN', 
                             'Oct21_PSQIDAYDYS', 'Oct21_PSQIHSE', 'Oct21_PSQISLPQUAL', 
                             'Oct21_PSQIMEDS']].sum(axis=1, skipna=False)
assert all(r8['Oct21_PSQI_TOTAL'].between(0, 21) | r8['Oct21_PSQI_TOTAL'].isna())

#Insomnia Severity Index
r8['Oct21_ISI_Total'] = r8[[x for x in r8.columns if x.startswith('isi_')]].sum(axis=1, skipna=False)

#Munich ChronoType Questionnaire calculations
r8['Oct21_mtq_freedays']  = 7 - r8['mtq_2']
mtq_renames = {'mtq_2': 'Oct21_mtq_workdays',
               'mtq_3': 'Oct21_mtq_workday_sleeponset',
               'mtq_p8': 'Oct21_mtq_workday_sleepend',
               'mtq_p9': 'Oct21_mtq_freeday_sleeponset',
               'mtq_p10': 'Oct21_mtq_freeday_sleepend'}
for col in mtq_renames:
    r8[mtq_renames[col]] = r8[col]
#Sleep duration
(r8['Oct21_mtq_workday_sleepduration'], r8['Oct21_mtq_workday_sleepduration_12']) = calc_sleep_time(r8['Oct21_mtq_workday_sleeponset'], 
                                                                                                          r8['Oct21_mtq_workday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
(r8['Oct21_mtq_freeday_sleepduration'], r8['Oct21_mtq_freeday_sleepduration_12']) = calc_sleep_time(r8['Oct21_mtq_freeday_sleeponset'], 
                                                                                                          r8['Oct21_mtq_freeday_sleepend'],
                                                                                                          correct_12=True, min_12=2)
#Sleep midpoint
r8['Oct21_mtq_workday_sleepmidpoint'] = calc_sleep_midpoint(r8['Oct21_mtq_workday_sleeponset'], 
                                                               r8['Oct21_mtq_workday_sleepend'], 
                                                               correct_12=True, min_12=2)
r8['Oct21_mtq_freeday_sleepmidpoint'] = calc_sleep_midpoint(r8['Oct21_mtq_freeday_sleeponset'], 
                                                               r8['Oct21_mtq_freeday_sleepend'], 
                                                               correct_12=True, min_12=2)
#Average sleep duration
r8['Oct21_mtq_avg_wk_sleepduration'] = (r8['Oct21_mtq_workday_sleepduration']*r8['Oct21_mtq_workdays']
                                           + r8['Oct21_mtq_freeday_sleepduration']*r8['Oct21_mtq_freedays'])/7
#Chronotype
r8['Oct21_mtq_chronotype'] = r8['Oct21_mtq_freeday_sleepmidpoint']
idx = r8['Oct21_mtq_freeday_sleepduration'] > r8['Oct21_mtq_workday_sleepduration']
r8.loc[idx, 'Oct21_mtq_chronotype'] -= ((r8.loc[idx, 'Oct21_mtq_freeday_sleepduration'] 
                                           - r8.loc[idx, 'Oct21_mtq_workday_sleepduration'])/2).apply(pd.Timedelta, unit='hours')
r8['Oct21_mtq_chronotype'] = r8['Oct21_mtq_chronotype'].round('min')
print('r8: dropping %d negative MTQ chronotypes' % sum(r8['Oct21_mtq_chronotype'] < pd.Timedelta(0)))
r8.loc[r8['Oct21_mtq_chronotype'] < pd.Timedelta(0), 'Oct21_mtq_chronotype'] = pd.NaT

#PROMIS
r8['Oct21_PROMIS_Total'] = (r8.loc[:, 'promis_1':'promis_6'].sum(axis=1, skipna=False)
                            + (6 - r8['promis_7']))

#PROMIS sleep disturbance
r8['PROMIS_Sleep_Disturbance_Total'] = (r8[['promis_sd_1', 'promis_sd_4', 'promis_sd_5', 
                                            'promis_sd_6']].sum(axis=1, skipna=False)
                                        + (6 - r8[['promis_sd_2', 'promis_sd_3', 
                                              'promis_sd_7', 'promis_sd_8']]).sum(axis=1, skipna=False))

#PROMIS sleep imapairment
r8['PROMISE_SRI_Total'] =( r8[['promis_sri_1', 'promis_sri_3', 'promis_sri_4', 
                               'promis_sri_5', 'promis_sri_6', 'promis_sri_7', 
                               'promis_sri_8']].sum(axis=1, skipna=False) 
                          + (6 -  r8['promis_sri_2']))

#Generalized anxiety
r8['Oct21_gad_7_total'] = r8[[x for x in r8.columns if x.startswith('gad_')]].sum(axis=1, skipna=False)

#PSS
r8['Oct21_PSS_TOTAL'] = (r8[['pss_1', 'pss_2', 'pss_3', 'pss_6', 'pss_9', 'pss_10']].sum(axis=1, skipna=False) 
                         + (4 - r8[['pss_4', 'pss_5', 'pss_7', 'pss_8']]).sum(axis=1, skipna=False))

#LSAS
r8['LSAS_Fear'] = r8[[col for col in r8.columns if '_fear' in col]].sum(axis=1, skipna=False)
r8['LSAS_Anxiety'] = r8[[col for col in r8.columns if '_avoid' in col]].sum(axis=1, skipna=False)
r8['LSAS_TOTAL'] = r8['LSAS_Fear'] + r8['LSAS_Anxiety']

#Personality inventory for DSM
r8['PID_Oct21_Total_Raw_Score'] = r8[[col for col in r8.columns if 'pid_' in col]].sum(axis=1, skipna=False)
r8['PID_Oct21_Total_Negative_Affect'] = r8[['pid_8', 'pid_9', 'pid_10', 'pid_11', 'pid_15']].sum(axis=1, skipna=False)
r8['PID_Oct21_Total_Detachment'] = r8[['pid_4', 'pid_13', 'pid_14', 'pid_16', 'pid_18']].sum(axis=1, skipna=False)
r8['PID_Oct21_Total_Antagonism'] = r8[['pid_17', 'pid_19', 'pid_20', 'pid_22', 'pid_25']].sum(axis=1, skipna=False)
r8['PID_Oct21_Total_Disinhibition'] = r8[['pid_1', 'pid_2', 'pid_3', 'pid_5', 'pid_6']].sum(axis=1, skipna=False)
r8['PID_Oct21_Total_Psychoticisim'] = r8[['pid_7', 'pid_12', 'pid_21', 'pid_23', 'pid_24']].sum(axis=1, skipna=False)

#Iowa Sleep Disturbance Inventory
r8['ISDI_TOTAL'] = (r8[['isdi_fallasleep', 'isdi_nightmares', 'isdi_wakefallasleep', 
                        'isdi_lightsleep', 'isdi_legs', 'isdi_movesleep', 'isdi_tiredday', 
                        'isdi_hardwakeweek', 'isdi_wakeearly', 'isdi_longnaps', 
                        'isdi_irregularbedtime', 'isdi_legpaincramps', 'isdi_awakeworrying', 
                        'isdi_troublefallasleep', 'isdi_sitdrowsy', 'isdi_recurringbaddreams', 
                        'isdi_wakefrequently', 'isdi_napanywhere', 'isdi_awakenoises', 
                        'isdi_legsensations', 'isdi_nervousness', 'isdi_kicklegs', 
                        'isdi_lessenergy', 'isdi_dreamsdisturb', 'isdi_feelworsemorning', 
                        'isdi_timebacktosleep', 'isdi_dozetv', 'isdi_sleeproutine', 
                        'isdi_legsstill', 'isdi_anxietyasleep', 'isdi_legsjerk', 
                        'isdi_layawake', 'isdi_enoughenergy', 'isdi_nightmareswake', 
                        'isdi_tiredmorning', 'isdi_troublestayasleep', 'isdi_sleepday', 
                        'isdi_wakeirregular', 'isdi_movelegsuncomfortable', 'isdi_mindraces', 
                        'isdi_frighteningdreams', 'isdi_movearound', 'isdi_troublewaking', 
                        'isdi_wakenoreason', 'isdi_dozeoffrelax', 'isdi_sleepdisturbed', 
                        'isdi_wokenlegs', 'isdi_thinkingevents', 'isdi_kickpunch', 
                        'isdi_hardrelaxbedtime', 'isdi_focustired', 'isdi_dreamsvividfeel', 
                        'isdi_attentiontired', 'isdi_dreamsunpleasant', 'isdi_awakethinking', 
                        'isdi_tiredwakeup', 'isdi_sleeppoorly', 'isdi_trytoohard', 
                        'isdi_strugglealert', 'isdi_baddreams', 'isdi_upearlier', 
                        'isdi_upearlierthanplanned', 'isdi_baddreamhappened', 
                        'isdi_wakebeforeneed', 'isdi_nightmareshard', 'isdi_hardcomfortable', 
                        'isdi_sleepyday', 'isdi_nightmaresphysical', 'isdi_daytimesleepy', 
                        'isdi_cantmovewakeup', 'isdi_intenseimages', 'isdi_musclesfrozen', 
                        'isdi_lyingpresence', 'isdi_unablemove', 'isdi_seehearnotreal', 
                        'isdi_dreamlikemorning']].sum(axis=1, skipna=False) 
                    + (1 - r8[['isdi_wideawake', 'isdi_rested', 'isdi_naps', 
                               'isdi_waketime', 'isdi_worries', 'isdi_sleepquickly', 
                               'isdi_nonightmares', 'isdi_deepsleeper', 'isdi_dontmove', 
                               'isdi_energized', 'isdi_eveningsleeptime', 'isdi_fallasleepminutes', 
                               'isdi_loudnoises', 'isdi_sleepthroughanything', 
                               'isdi_raretroubleasleep', 'isdi_drifteasily', 
                               'isdi_sleepybed']]).sum(axis=1, skipna=False))
r8['ISDI_Nightmares'] = (r8[['isdi_nightmares', 'isdi_recurringbaddreams',
                             'isdi_dreamsdisturb', 'isdi_nightmareswake', 
                             'isdi_frighteningdreams', 'isdi_dreamsvividfeel', 
                             'isdi_dreamsunpleasant', 'isdi_baddreams', 
                             'isdi_baddreamhappened', 'isdi_nightmareshard', 
                             'isdi_nightmaresphysical']].sum(axis=1, skipna=False) 
                         + (1 - r8['isdi_nonightmares']))
r8['ISDI_Initial_Insomnia'] = (r8[['isdi_fallasleep', 'isdi_troublefallasleep', 
                                   'isdi_layawake', 'isdi_hardrelaxbedtime', 
                                   'isdi_trytoohard', 
                                   'isdi_hardcomfortable']].sum(axis=1, skipna=False) 
                               + (1 - r8[['isdi_sleepquickly', 'isdi_fallasleepminutes', 
                                          'isdi_raretroubleasleep', 'isdi_drifteasily', 
                                          'isdi_sleepybed']]).sum(axis=1, skipna=False))
r8['ISDI_Fatigue'] = (r8[['isdi_tiredday', 'isdi_sitdrowsy', 'isdi_lessenergy', 
                          'isdi_enoughenergy', 'isdi_focustired', 'isdi_attentiontired', 
                          'isdi_strugglealert', 'isdi_sleepyday', 
                          'isdi_daytimesleepy']].sum(axis=1, skipna=False) 
                      + (1 - r8['isdi_wideawake']))
r8['ISDI_Nonrestorative_Sleep'] = (r8[['isdi_hardwakeweek', 'isdi_feelworsemorning', 
                                       'isdi_tiredmorning', 'isdi_troublewaking', 
                                       'isdi_tiredwakeup', 
                                       'isdi_upearlier']].sum(axis=1, skipna=False) 
                                   + (1 - r8[['isdi_rested', 
                                              'isdi_energized']]).sum(axis=1, skipna=False))
r8['ISDI_Daytime_Disturbances'] = r8[['ISDI_Fatigue', 'ISDI_Nonrestorative_Sleep']].sum(axis=1, skipna=False)
r8['ISDI_Fragmented_sleep'] = r8[['isdi_wakefallasleep', 'isdi_wakeearly', 
                                  'isdi_wakefrequently', 'isdi_timebacktosleep', 
                                  'isdi_troublestayasleep', 'isdi_wakenoreason', 
                                  'isdi_sleeppoorly', 'isdi_upearlierthanplanned', 
                                  'isdi_wakebeforeneed']].sum(axis=1, skipna=False)
r8['ISDI_Anxiety_Night'] = (r8[['isdi_awakeworrying', 'isdi_nervousness', 
                                'isdi_anxietyasleep', 'isdi_mindraces', 
                                'isdi_thinkingevents', 'isdi_awakethinking']].sum(axis=1, skipna=False) 
                            + (1 - r8['isdi_worries']))
r8['ISDI_Light_Sleep'] = (r8[['isdi_lightsleep', 'isdi_awakenoises', 
                              'isdi_sleepdisturbed']].sum(axis=1, skipna=False) 
                          + (1 - r8[['isdi_deepsleeper', 'isdi_loudnoises', 
                                     'isdi_sleepthroughanything']]).sum(axis=1, skipna=False))
r8['ISDI_Movement_Night'] = (r8[['isdi_movesleep', 'isdi_kicklegs', 'isdi_legsjerk', 
                                 'isdi_movearound', 'isdi_kickpunch']].sum(axis=1, skipna=False) 
                             + (1 - r8['isdi_dontmove']))
r8['ISDI_Sensations_Night'] = r8[['isdi_legs', 'isdi_legpaincramps', 'isdi_legsensations', 
                                  'isdi_legsstill', 'isdi_movelegsuncomfortable', 
                                  'isdi_wokenlegs']].sum(axis=1, skipna=False)
r8['ISDI_Excessive_Sleep'] = (r8[['isdi_longnaps', 'isdi_napanywhere', 'isdi_dozetv', 
                                  'isdi_sleepday', 'isdi_dozeoffrelax']].sum(axis=1, skipna=False) 
                              + (1 - r8['isdi_naps']))
r8['ISDI_Irregular_Schedule'] = (r8[['isdi_irregularbedtime', 'isdi_sleeproutine', 
                                     'isdi_wakeirregular']].sum(axis=1, skipna=False) 
                                 + (1 - r8[['isdi_waketime', 'isdi_eveningsleeptime']]).sum(axis=1, skipna=False))
r8['ISDI_Sleep_Paralysis'] = r8[['isdi_cantmovewakeup', 'isdi_musclesfrozen', 
                                 'isdi_unablemove']].sum(axis=1, skipna=False)
r8['ISDI_Sleep_Hallucinations'] = r8[['isdi_intenseimages', 'isdi_lyingpresence', 
                                      'isdi_seehearnotreal', 
                                      'isdi_dreamlikemorning']].sum(axis=1, skipna=False)



#%% QUALITY CONTROL

report_missing = False

daily_QC(data, report_missing=report_missing)
demo_QC(demo, report_missing=report_missing)
R1_QC(r1, report_missing=report_missing)
R2_QC(r2, report_missing=report_missing)
R3_QC(r3, report_missing=report_missing)
R4_QC(r4, report_missing=report_missing)
R5_QC(r5, report_missing=report_missing)
R6_QC(r6, report_missing=report_missing)
R7_QC(r7, report_missing=report_missing)
R8_QC(r8, report_missing=report_missing)
R9_QC(r9, report_missing=report_missing)



#%% CLEANED OUTPUT

if output:
    
    #Daily
    df2csv(join(main_dir, 'export', 'COVID19_combined_cleaned_%s.csv' % output_timestamp), data)
    df2csv(join(main_dir, 'export', 'COVID19_combined_cleaned_deid_%s.csv' % output_timestamp), data.drop(daily_id_vars, axis='columns'))
    
    #Demographics
    demo.to_csv(join(main_dir, 'export', 'COVID19_demographics_cleaned_%s.csv' % output_timestamp))
    demo.drop(demo_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_demographics_cleaned_deid_%s.csv' % output_timestamp))
    
    #Round 1
    df2csv(join(main_dir, 'export', 'COVID19_Round1_cleaned_%s.csv' % output_timestamp), r1)
    df2csv(join(main_dir, 'export', 'COVID19_Round1_cleaned_deid_%s.csv' % output_timestamp), r1.drop(r1_id_vars, axis='columns'))
    
    #Round 2
    df2csv(join(main_dir, 'export', 'COVID19_Round2_cleaned_%s.csv' % output_timestamp), r2)
    df2csv(join(main_dir, 'export', 'COVID19_Round2_cleaned_deid_%s.csv' % output_timestamp), r2.drop(r2_id_vars, axis='columns'))
    
    #Round 3
    df2csv(join(main_dir, 'export', 'COVID19_Round3_cleaned_%s.csv' % output_timestamp), r3)
    df2csv(join(main_dir, 'export', 'COVID19_Round3_cleaned_deid_%s.csv' % output_timestamp), r3.drop(r3_id_vars, axis='columns'))
    
    #Round 4
    df2csv(join(main_dir, 'export', 'COVID19_Round4_cleaned_%s.csv' % output_timestamp), r4)
    df2csv(join(main_dir, 'export', 'COVID19_Round4_cleaned_deid_%s.csv' % output_timestamp), r4.drop(r4_id_vars, axis='columns'))
    
    #Round 5
    df2csv(join(main_dir, 'export', 'COVID19_Round5_cleaned_%s.csv' % output_timestamp), r5)
    df2csv(join(main_dir, 'export', 'COVID19_Round5_cleaned_deid_%s.csv' % output_timestamp), r5.drop(r5_id_vars, axis='columns'))
    
    #Round 6
    df2csv(join(main_dir, 'export', 'COVID19_Round6_cleaned_%s.csv' % output_timestamp), r6)
    df2csv(join(main_dir, 'export', 'COVID19_Round6_cleaned_deid_%s.csv' % output_timestamp), r6.drop(r6_id_vars, axis='columns'))
    
    #Round 7
    df2csv(join(main_dir, 'export', 'COVID19_Round7_cleaned_%s.csv' % output_timestamp), r7)
    df2csv(join(main_dir, 'export', 'COVID19_Round7_cleaned_deid_%s.csv' % output_timestamp), r7.drop(r7_id_vars, axis='columns'))
    
    #Round 8
    df2csv(join(main_dir, 'export', 'COVID19_Round8_cleaned_%s.csv' % output_timestamp), r8)
    df2csv(join(main_dir, 'export', 'COVID19_Round8_cleaned_deid_%s.csv' % output_timestamp), r8.drop(r8_id_vars, axis='columns'))
    
    #Round 9
    df2csv(join(main_dir, 'export', 'COVID19_Round9_cleaned_%s.csv' % output_timestamp), r9)
    df2csv(join(main_dir, 'export', 'COVID19_Round9_cleaned_deid_%s.csv' % output_timestamp), r9.drop(r9_id_vars, axis='columns'))
