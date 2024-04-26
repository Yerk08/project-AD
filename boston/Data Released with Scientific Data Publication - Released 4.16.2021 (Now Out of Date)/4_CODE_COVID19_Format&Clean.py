# -*- coding: utf-8 -*-
"""
Import, format, and check COVID-19 data

Author: Eric Fields
Version Date: 28 September 2020
"""

import sys
import os
from os.path import join
import re
#from dateutil.parser import ParserError

import numpy as np
import pandas as pd

main_dir = r'D:\COVID19'
os.chdir(main_dir)

sys.path.append(join(main_dir, 'code'))
from COVID19_QC import daily_QC, demo_QC, R1_QC, R2_QC, R3_QC


#Indicate whether to output files when the script is run
output = False


def timedelta2str(td):
    """
    Convert timedelta data being used to represent clock time to MM:SS format string
    """
    if pd.isna(td):
        return ''
    else:
        assert td >= pd.Timedelta(0)
        assert td <= pd.Timedelta(24, unit='hours')
        return str(td.round('min')).split()[-1][0:5]


def df2csv(filename, df):
    """
    Write datafram to csv with dates and timedelta objects sensibly formatted
    """
    
    #Make copy before modifying
    out_data = df.copy()
    
    #Find time delta variables
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
    
    #Get sleep time assuming 24 hour clock
    st = awake_time - bed_time
    st = st.apply(lambda x: x.total_seconds()/3600)
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
    
    #Confirm that times are in the right format and range
    if not all(bed_time.between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) | bed_time.isna()):
        raise ValueError('All bed times must be between 00:00 and 23:59')
    if not all(awake_time.between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) | awake_time.isna()):
        raise ValueError('All awake times must be between 00:00 and 23:59')
    
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
df_short = pd.read_csv(join(main_dir, 'raw_data', 'COVIDLongitudinalDat_DATA_2020-09-02_1716.csv'))
df_long  = pd.read_csv(join(main_dir, 'raw_data', 'COVID19LongitudinalD_DATA_2020-09-02_1715.csv'))
demo     = pd.read_csv(join(main_dir, 'raw_data', 'COVID19-DEMOREPORT_DATA_2020-09-02_1715.csv'))
r1 = pd.read_csv(join(main_dir, 'raw_data', 'Round1', 'Round1COVIDAdditiona_DATA_2020-09-02_1721.csv'))
r2 = pd.read_csv(join(main_dir, 'raw_data', 'Round2', 'Round2COVIDAdditiona_DATA_2020-09-02_1725.csv'))
r3 = pd.read_csv(join(main_dir, 'raw_data', 'Round3', 'Round3COVIDAdditiona_DATA_2020-09-02_1725.csv'))

#Merge long and short surveys
data = df_long.merge(df_short, how='outer')
del df_short, df_long



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

#Remove whitespace from subject IDs and convert all to uppercase
data['subjid_2'] = data['subjid_2'].astype(str).str.strip().str.upper()
demo['subjid_1'] = demo['subjid_1'].astype(str).str.strip().str.upper()
r1['subjid_rd1'] = r1['subjid_rd1'].astype(str).str.strip().str.upper()
r2['subjid_rd2'] = r2['subjid_rd2'].astype(str).str.strip().str.upper()
r3['subjid_rd3'] = r3['subjid_rd3'].astype(str).str.strip().str.upper()

#These subjects are under 18 and were included by mistake or the sub_id was
#assigned twice
excl_subs = ['54DLL', 'A2YXX', '7QU6Y']
data = data[~data['subjid_2'].isin(excl_subs)]
demo = demo[~demo['subjid_1'].isin(excl_subs)]
r1 = r1[~r1['subjid_rd1'].isin(excl_subs)]
r2 = r2[~r2['subjid_rd2'].isin(excl_subs)]
r3 = r3[~r3['subjid_rd3'].isin(excl_subs)]

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

#Output a list of suject IDs in the data that don't appear in valid list
if output:
    data.loc[~data['subjid_2'].isin(valid_sub_ids), 'subjid_2'].to_csv(join(main_dir, 'data_check', 'daily_problem_sub_ids.csv'), index=False)
    demo.loc[~demo['subjid_1'].isin(valid_sub_ids), 'subjid_1'].to_csv(join(main_dir, 'data_check', 'demo_problem_sub_ids.csv'), index=False)
    r1.loc[~r1['subjid_rd1'].isin(valid_sub_ids), 'subjid_rd1'].to_csv(join(main_dir, 'data_check', 'r1_problem_sub_ids.csv'), index=False)
    r2.loc[~r2['subjid_rd2'].isin(valid_sub_ids), 'subjid_rd2'].to_csv(join(main_dir, 'data_check', 'r2_problem_sub_ids.csv'), index=False)
    r3.loc[~r3['subjid_rd3'].isin(valid_sub_ids), 'subjid_rd3'].to_csv(join(main_dir, 'data_check', 'r3_problem_sub_ids.csv'), index=False)

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



#%% INDEX & SORT

#Create unique ID from record_id column
assert data['redcap_repeat_instrument'].isin(['covid19', 'covid19_short_survey']).all()
data['unique_id'] = data['record_id']
data.loc[data['redcap_repeat_instrument']=='covid19', 'unique_id'] = data.loc[data['redcap_repeat_instrument']=='covid19', 'unique_id'].apply(lambda x: str(x)+'L')
data.loc[data['redcap_repeat_instrument']=='covid19_short_survey', 'unique_id'] = data.loc[data['redcap_repeat_instrument']=='covid19_short_survey', 'unique_id'].apply(lambda x: str(x)+'S')

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

#Sort
data.sort_values(['sub_id', 'todays_date'], inplace=True)
demo.sort_values('sub_id', inplace=True)
r1.sort_values('sub_id', inplace=True)
r2.sort_values('sub_id', inplace=True)
r3.sort_values('sub_id', inplace=True)



#%% OUTPUT RAW DATA

if output:

    daily_id_vars = ['sleepdiary_dreamcontent', 'sleepdiary_info', 'visit', 'respiratory_describe', 'full_open', 'open_question']
    demo_id_vars = ['medical_description', 'institution_describe', 'additional_info', 'school', 'occupation']
    r1_id_vars = ['psqi_5j2']
    r2_id_vars = ['challenging_free', 'positive_free', 'mundane_free', 'unusual_free']
    r3_id_vars = ['city', 'highrisk_othercheck', 'quar_free', 'positive_free_response',
                  'covid_impact_free', 'occupation_other', 'sleepchange_free',
                  'med_free', 'med_other', 'psych_free_1', 'psych_free_2',
                  'condition_free', 'mil_time_free', 'mistakes_free', 'open_anything',
                  'open_anything_2', 'covdream_free']
    
    data.to_csv(join(main_dir, 'export', 'COVID19_combined_raw.csv'))
    data.drop(daily_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_combined_raw_deid.csv'))
    
    demo.to_csv(join(main_dir, 'export', 'COVID19_demographics_raw.csv'))
    demo.drop(demo_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_demographics_raw_deid.csv'))
    
    r1.to_csv(join(main_dir, 'export', 'COVID19_Round1_raw.csv'))
    r1.drop(r1_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round1_raw_deid.csv'))
    
    r2.to_csv(join(main_dir, 'export', 'COVID19_Round2_raw.csv'))
    r2.drop(r2_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round2_raw_deid.csv'))
    
    r3.to_csv(join(main_dir, 'export', 'COVID19_Round3_raw.csv'))
    r3.drop(r3_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_Round3_raw_deid.csv'))



#%% DROP DUPLICATES AND INCOMPLETE DATA

#Drop incomplete survyes that are date-subject duplicates
incomplete_idx = ((data['covid19_timestamp'] == '[not completed]') 
                  | (data['covid19_short_survey_timestamp'] == '[not completed]'))
data = data[~incomplete_idx]

#Check that there are no duplicates in demographic data
assert not demo['sub_id'].duplicated(keep=False).any()

#Drop Round 1 & 2 duplicates
r1.drop_duplicates('sub_id', keep='first', inplace=True)
r2.drop_duplicates('sub_id', keep='first', inplace=True)
r3.drop_duplicates('sub_id', keep='first', inplace=True)



#%% FORMATTING: DAILY SURVEYS

print('\n\n##### FORMATTING #####')

#Make survey type (short vs full) a categorical variable
data['redcap_repeat_instrument'] = data['redcap_repeat_instrument'].astype('category')

#Combine timestamp columns
data['redcap_timestamp'] = data['covid19_timestamp']
data.loc[data['redcap_timestamp'].isna(), 'redcap_timestamp'] = data.loc[data['redcap_timestamp'].isna(), 'covid19_short_survey_timestamp']
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
                                                 '38,6': 38.6})
#Try to convert to numeric
for i in np.where(~data['fever_temp'].apply(np.isreal))[0]:
    try:
        data.iloc[i, data.columns.get_loc('fever_temp')] = np.float64(data.iloc[i, data.columns.get_loc('fever_temp')])
    except ValueError:
        print('"%s" in fever_temp cannot be made numeric'
              % data.iloc[i, data.columns.get_loc('fever_temp')])
#Use nan for values that couldn't be converted to numeric
print('\nReplacing %d non_numeric fever_temp values with nan\n' 
      % sum(~data['fever_temp'].apply(np.isreal)))
data.loc[~data['fever_temp'].apply(np.isreal), 'fever_temp'] = np.nan
#Make the column numeric
data['fever_temp'] = data['fever_temp'].astype(np.float64)

#Subtract 1 from depression columns so they start at 0
depression_vars = ['depression1', 'depression2', 'depression3', 'depression4',
                   'depression5', 'depression6', 'depression7', 'depression8']
data[depression_vars] += -1

#For questions where a numeric answer was conditional on a yes answer to another question
#insert 0 if the first question was no; for example, if the participants said they
#didn't name, naptime becomes 0
data.loc[data['sleepdiary_wakes']==0, 'night_awakening_time'] = 0
data.loc[data['sleepdiary_nap']==0, 'sleepdiary_naptime'] = 0
data.loc[data['socialize']==0, 'socialize_min'] = 0



#%% REFERENCE DATE
#Find (best guess for) the date that answers are referring to in daily data

#Find rows where dates might be problematic because todays_date is duplicated
#or todays_date and redcap_timestamp don't match
idx = sub_date_duplicates(data, keep=False)
idx = idx | (np.abs(data['todays_date'] - data['redcap_timestamp']) > pd.Timedelta(1, unit='days'))
if output:
    data[idx].to_csv(join(main_dir, 'data_check', 'ref_date_problems.csv'))

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
        demo.iloc[i, demo.columns.get_loc('dependent_children')] = np.float(demo.iloc[i, demo.columns.get_loc('dependent_children')])
    except ValueError:
        print('%s in dependent_children could not be converted to numeric' %
              demo.iloc[i, demo.columns.get_loc('dependent_children')])
print('\nReplacing %d non_numeric dependent_children values with nan' 
      % sum(~demo['dependent_children'].apply(np.isreal)))
demo.loc[~demo['dependent_children'].apply(np.isreal), 'dependent_children'] = np.nan
demo['dependent_children'] = demo['dependent_children'].astype(np.float64)

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
                       'AMERICA': 'UNITED STATES',
                       'ENGLAND': 'UNITED KINGDOM',
                       'SCOTLAND': 'UNITED KINGDOM',
                       'UK': 'UNITED KINGDOM',
                       'THE NETHERLANDS': 'NETHERLANDS',
                       'BRASIL': 'BRAZIL',
                       'MÉXICO': 'MEXICO',
                       'KOREA': 'SOUTH KOREA',
                       'KSA': 'SAUDI ARABIA'}
demo['country'] = demo['country'].str.strip().str.upper()
demo['country'] = demo['country'].replace(country_corrections)
demo.loc[1563, 'country'] = 'UNITED STATES'
demo.loc[1748, 'country'] = 'UNITED STATES'
demo.loc[1650, 'country'] = 'UNITED STATES'
with (open(join(main_dir, 'code', 'countries.txt'))) as f_in:
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
                     'SOUTH CAROLINA/ GEORGIA':np.nan}
demo['state'] = demo['state'].replace(state_corrections)
assert demo.loc[(demo.loc[demo['country'].isin(['UNITED STATES', 'CANADA']), 'state'].isin(state_abbrev['short']) 
                & demo['state'].notna()), 'state'].isin(state_abbrev['short']).all()

#Standardize school response
school_replace = pd.read_csv(join(main_dir, 'raw_data', 'school_replacements_for_deid.csv')).set_index('ORIGINAL').squeeze().to_dict()
school_replace = {**school_replace,
                  '4 year college': '4 year college/university',
                  '4 year': '4 year college/university',
                  '4-year University': '4 year college/university',
                  '4 Year University': '4 year college/university',
                  '4 Year': '4 year college/university',
                  '4 year': '4 year college/university',
                  '2 year college': '2 year college/university',
                  '2-year College': '2 year college/university',
                  '3 year college': '3 year college/university',
                  'med school': 'Medical School',
                  'graduate school, PhD': 'Graduate School (PhD)',
                  '4-year college/university': '4 year college/university',
                  "Integrated Master's (4 year university)": '4 year college/university',
                  '4 yr university': '4 year college/university'}
demo['school'] = demo['school'].str.strip()
demo['school'] = demo['school'].replace(school_replace)
idx = ~demo['school'].isin([np.nan, *set(school_replace.values())])
print('Replacing the following school responses with nan:')
print(demo.loc[idx, 'school'])
demo.loc[idx, 'school'] = np.nan

#Standardize occupation response
occ_replace = pd.read_csv(join(main_dir, 'raw_data', 'occupation_replacements_for_deid.csv')).set_index('old').squeeze().to_dict()
demo['occupation'] = demo['occupation'].replace(occ_replace)


### Remove impossible values ###

data.loc[data['sleepdiary_sleeplatency'] > 24*60, 'sleepdiary_sleeplatency'] = np.nan

data.loc[data['sleepdiary_naptime'] > 24*60, 'sleepdiary_naptime'] = np.nan
data.loc[data['socialize_min'] > 24*60, 'socialize_min'] = np.nan
data.loc[data['alcohol_bev']>48, 'alcohol_bev'] = np.nan
data.loc[(data['redcap_timestamp'] - pd.to_datetime('1/23/20')).dt.days < data['quarantine_days'], 'quarantine_days'] = np.nan

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
r3.loc[r3['age']>120, 'age'] = np.nan

#Clean up countries
r3_country_corrections = {'USA': 'UNITED STATES',
                          'US': 'UNITED STATES',
                          'UNITED STATES OF AMERICA': 'UNITED STATES',
                          'U.S.': 'UNITED STATES',
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
                        'AMHERST, MA': 'MA',
                        'WASHINGTON (STATE)': 'WA',
                        'MASSACHUSETT': 'MA',
                        'WASHINGTON STATE': 'WA',
                        'MASSACHUSETTES': 'MA',
                        'NY- LONG ISLAND': 'NY',
                        'WASHINGTON DC':'DC',
                        'BRITHISH COLUMBIA':'BC'}
r3['state_3mo'] = r3['state_3mo'].replace(r3_state_corrections)
r3.loc[~r3['state_3mo'].isin(state_abbrev['short']) & r3['state_3mo'].notna(), 'state_3mo'] = np.nan

#TO DO: Date variables
r3_date_vars = [x for x in r3.columns if x.endswith('_start') or x.endswith('_end')]



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
data['exercise'] = ((data['sleepdiary_exercise']>0) 
                    | (data[['sleepdiary_exercise___1', 
                             'sleepdiary_exercise___2', 
                             'sleepdiary_exercise___3']]>0).any(axis=1)).astype(int)

#Get fever temperatures in celsius only
data.loc[data['temp_measure']==1, 'fever_temp_C'] = data.loc[data['temp_measure']==1, 'fever_temp']
data.loc[data['temp_measure']==2, 'fever_temp_C'] = (data.loc[data['temp_measure']==2, 'fever_temp'] - 32) * (5/9)
assert data.loc[data['fever_temp_C'].notna(), 'fever_temp_C'].between(24, 44).all()

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
r1.loc[r1['PSQIDISTB'].between(1, 9), 'PSQIDISTB'] = 1
r1.loc[r1['PSQIDISTB'].between(9, 18), 'PSQIDISTB'] = 2
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
               'mtq_2': 'mtq_postcovid_workdays',
               'mtq_3': 'mtq_postcovid_workday_sleeponset',
               'mtq_4': 'mtq_postcovid_workday_sleepend',
               'mtq_5': 'mtq_postcovid_freeday_sleeponset',
               'mtq_6': 'mtq_postcovid_freeday_sleepend'}
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
r1['LSAS_Fear_ PreCovid'] = r1[[x for x in LSAS_cols 
                                if 'fear' in x and '2' not in x]].sum(axis=1, skipna=False)
r1['LSAS_Anxiety_ PreCovid'] = r1[[x for x in LSAS_cols 
                                   if 'avoid' in x and '2' not in x]].sum(axis=1, skipna=False)
r1['LSAS_TOTAL_ PreCovid'] = r1['LSAS_Fear_ PreCovid'] + r1['LSAS_Anxiety_ PreCovid']
r1['LSAS_Fear_ PostCovid'] = r1[[x for x in LSAS_cols 
                                 if 'fear' in x and '2' in x]].sum(axis=1, skipna=False)
r1['LSAS_Anxiety_ PostCovid'] = r1[[x for x in LSAS_cols 
                                    if 'avoid' in x and '2' in x]].sum(axis=1, skipna=False)
r1['LSAS_TOTAL_ PostCovid'] = r1['LSAS_Fear_ PostCovid'] + r1['LSAS_Anxiety_ PostCovid']

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



#%% QUALITY CONTROL

daily_QC(data)
demo_QC(demo)
R1_QC(r1)
R2_QC(r2)
R3_QC(r3)



#%% CLEANED OUTPUT

if output:
    
    #Daily
    df2csv(join(main_dir, 'export', 'COVID19_combined_cleaned.csv'), data)
    df2csv(join(main_dir, 'export', 'COVID19_combined_cleaned_deid.csv'), data.drop(daily_id_vars, axis='columns'))
    
    #Demographics
    demo.to_csv(join(main_dir, 'export', 'COVID19_demographics_cleaned.csv'))
    demo.drop(demo_id_vars, axis='columns').to_csv(join(main_dir, 'export', 'COVID19_demographics_cleaned_deid.csv'))
    
    #Round 1
    df2csv(join(main_dir, 'export', 'COVID19_Round1_cleaned.csv'), r1)
    df2csv(join(main_dir, 'export', 'COVID19_Round1_cleaned_deid.csv'), r1.drop(r1_id_vars, axis='columns'))
    
    #Round 2
    df2csv(join(main_dir, 'export', 'COVID19_Round2_cleaned.csv'), r2)
    df2csv(join(main_dir, 'export', 'COVID19_Round2_cleaned_deid.csv'), r2.drop(r2_id_vars, axis='columns'))
    
    #Round 3
    df2csv(join(main_dir, 'export', 'COVID19_Round3_cleaned.csv'), r3)
    df2csv(join(main_dir, 'export', 'COVID19_Round3_cleaned_deid.csv'), r3.drop(r3_id_vars, axis='columns'))
