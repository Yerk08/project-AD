# -*- coding: utf-8 -*-
"""
Quality control functions for COVID-19 data

BC COVID-19 data
Cunningham, T. J., Fields, E. C., & Kensinger, E. A. (2021). Boston College daily 
sleep and well-being survey data during early phase of the COVID-19 pandemic. 
Scientific Data, 8(110). https://doi.org/10.1038/s41597-021-00886-y

Author: Eric Fields
Version Date: 22 July 2021

Copyright (c) 2021, Eric Fields
This code is free and open source software made available under the 3-clause BSD license
https://opensource.org/licenses/BSD-3-Clause
"""

import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def daily_QC(data, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    #Make sure any changes in this function don't change the data outside this function
    data = data.copy()
    
    print('\n\n##### FULL & SHORT SURVEY QC #####\n')
    
    print('\nQUALITY CONTROL REPORT\n')
    
    #Record ID
    assert is_numeric_dtype(data['record_id'])
    assert data['record_id'].isna().sum() == 0
    assert data.loc[~data['covid19_complete'].isna(), 'record_id'].duplicated().sum() == 0
    assert data.loc[~data['covid19_short_survey_complete'].isna(), 'record_id'].duplicated().sum() == 0
    
    #Which version was this
    assert data['redcap_repeat_instrument'].isin(['covid19', 'covid19_short_survey', np.nan]).all()
    
    #Check that survey version and completion codes match
    assert (data.loc[(data['covid19_complete'].isna() & data['april_18_complete'].isna()), 'redcap_repeat_instrument'] == 'covid19_short_survey').all()
    assert (data.loc[(data['covid19_short_survey_complete'].isna() & data['april_18_complete'].isna()), 'redcap_repeat_instrument'] == 'covid19').all()
    assert data.loc[(data['covid19_short_survey_complete'].isna() & data['covid19_complete'].isna()), 'redcap_repeat_instrument'].isna().all()
    assert data.loc[(data['covid19_short_survey_complete'].isna() & data['covid19_complete'].isna()), 'april_18_complete'].notna().all()
    
    #Todays date as input by subject
    today = pd.to_datetime('today')
    print('%d todays_date after today' % sum(data['todays_date'] > today))
    print('%d todays date before 3/21' % sum(data['todays_date'] < pd.to_datetime('3/21/20')))
    print('%d todays_date greater than 24 hours different than redcap timestamp'
          % sum(np.abs(data['todays_date'] - data['redcap_timestamp']) > pd.Timedelta(24, unit='hours')))
    print('todays_date range: %s to %s'
          % (data['todays_date'].min(), data['todays_date'].max()))
    
    #Redcap timestamps
    assert sum(data['covid19_timestamp'] > pd.to_datetime('today')) == 0
    print('covid19_timestamp range: %s to %s'
          % (data['covid19_timestamp'].min(), data['covid19_timestamp'].max()))
    
    #Mismtach in timestamps
    print('%d covid19_timestamp and todays_date differ by more than one day'
          % sum(np.abs(data['todays_date'] - data['covid19_timestamp']) > pd.Timedelta(1, unit='days')))
    print('%d covid19_short_survey_timestamp and todays_date differ by more than one day'
          % sum(np.abs(data['todays_date'] - data['covid19_short_survey_timestamp']) > pd.Timedelta(1, unit='days')))
        
    #Bed time
    assert (data.loc[data['sleepdiary_bedtime'].notna(), 'sleepdiary_bedtime'] >= pd.Timedelta(0)).all()
    assert (data.loc[data['sleepdiary_bedtime'].notna(), 'sleepdiary_bedtime'] < pd.Timedelta(1, unit='days')).all()
    print('%d bed times between 7am and 7pm'
          % sum((data['sleepdiary_bedtime'] < pd.Timedelta(19, unit='hours')) 
                & (data['sleepdiary_bedtime'] > pd.Timedelta(7, unit='hours'))))
    
    #Attempt to fall asleep time
    assert (data.loc[data['sleepdiary_fallasleep'].notna(), 'sleepdiary_fallasleep'] >= pd.Timedelta(0)).all()
    assert (data.loc[data['sleepdiary_fallasleep'].notna(), 'sleepdiary_fallasleep'] < pd.Timedelta(1, unit='days')).all()
    print('%d fallasleep times between 7am and 7pm'
          % sum((data['sleepdiary_fallasleep'] < pd.Timedelta(19, unit='hours')) 
                & (data['sleepdiary_fallasleep'] > pd.Timedelta(7, unit='hours'))))
    
    #Wake time
    assert (data.loc[data['sleepdiary_waketime'].notna(), 'sleepdiary_waketime'] >= pd.Timedelta(0)).all()
    assert (data.loc[data['sleepdiary_waketime'].notna(), 'sleepdiary_waketime'] < pd.Timedelta(1, unit='days')).all()
    print('%d wake times between 2pm and 2am'
          % sum((data['sleepdiary_waketime'] < pd.Timedelta(2, unit='hours')) 
                | (data['sleepdiary_waketime'] > pd.Timedelta(14, unit='hours'))))
    
    #Out of bed
    assert (data.loc[data['sleepdiary_outofbed'].notna(), 'sleepdiary_outofbed'] >= pd.Timedelta(0)).all()
    assert (data.loc[data['sleepdiary_outofbed'].notna(), 'sleepdiary_outofbed'] < pd.Timedelta(1, unit='days')).all()
    print('%d Out of bed times between 2pm and 2am'
          % sum((data['sleepdiary_outofbed'] < pd.Timedelta(2, unit='hours')) 
                | (data['sleepdiary_outofbed'] > pd.Timedelta(14, unit='hours'))))
    
    #Time taken to fall asleep
    assert is_numeric_dtype(data['sleepdiary_sleeplatency'])
    print('%d sleepdiary_sleeplatency greater than 8 hours, max = %.1f minutes' 
          % (sum(data['sleepdiary_sleeplatency']>8*60), data['sleepdiary_sleeplatency'].max()))
    
    #Number of times waking each night
    assert is_numeric_dtype(data['sleepdiary_wakes'])
    assert data['sleepdiary_wakes'].isin([0,1,2,3,4,5, np.nan]).all()
    
    #Minutes spent awake throughout the night
    assert is_numeric_dtype(data['night_awakening_time'])
    print('%d night_awakening_time greater than 8 hours, max = %.1f minutes' 
          % (sum(data['night_awakening_time']>8*60), data['night_awakening_time'].max()))
    
    #Remember dreaming
    assert is_numeric_dtype(data['sleepdiary_dreams'])
    assert data['sleepdiary_dreams'].isin([1,2,3, np.nan]).all()
    
    #Napped previous day
    assert is_numeric_dtype(data['sleepdiary_nap'])
    assert data['sleepdiary_nap'].isin([0,1, np.nan]).all()
    
    #Minutes napped
    assert is_numeric_dtype(data['sleepdiary_naptime'])
    print('%d sleepdiary_naptime greater than 8 hours, max = %.1f minutes' 
          % (sum(data['sleepdiary_naptime']>8*60), data['sleepdiary_naptime'].max()))
    
    #How hard was it to fall asleep
    assert is_numeric_dtype(data['sleepdiary_fellasleep'])
    assert data['sleepdiary_fellasleep'].isin([1,2,3, np.nan]).all()
    
    #Used sleep tracker
    assert is_numeric_dtype(data['cst'])
    assert data['cst'].isin([0,1, np.nan]).all()
    
    #Used step counter
    assert is_numeric_dtype(data['step_counter'])
    assert data['step_counter'].isin([0,1, np.nan]).all()
    
    #Number of steps
    assert is_numeric_dtype(data['steps'])
    print('%d steps greater than 20,000, max = %.0f' 
          % (sum(data['steps']>20000), data['steps'].max()))\
    
    #Did you leave the house
    assert is_numeric_dtype(data['leave_house'])
    assert data['leave_house'].isin([0,1,np.nan]).all()
    
    #How many people did you come in contact with
    assert is_numeric_dtype(data['people_contact'])
    print('%d people_contact greater than 1000, max = %d' 
          % (sum(data['people_contact']>1000), data['people_contact'].max()))
    
    #Did you socialize virtually
    assert is_numeric_dtype(data['socialize'])
    assert data['socialize'].isin([0,1,np.nan]).all()
    
    #Minutes socializing
    assert is_numeric_dtype(data['socialize_min'])
    print('%d socialize_min greater than 16 hours, max = %d minutes'
          % (sum(data['socialize_min']>16*60), data['socialize_min'].max()))
    
    #Excercise questions
    exercise_vars = ['sleepdiary_exercise',
                       'sleepdiary_exercise___0', 'sleepdiary_exercise___1',
                       'sleepdiary_exercise___2', 'sleepdiary_exercise___3']
    for col in exercise_vars:
        assert is_numeric_dtype(data[col])
    assert data['sleepdiary_exercise'].isin([0,1,2,3,np.nan]).all()
    assert data[exercise_vars[1:]].isin([0,1,np.nan]).all().all()
    print('%d inconsistent exercise variables'
          % sum((data['sleepdiary_exercise___0']==1)
                & data[exercise_vars[2:]].any(axis=1)))
    
    #Number of alcoholic beverages
    assert is_numeric_dtype(data['alcohol_bev'])
    print('%d alcohol_bev greater than 24, max = %d'
          % (sum(data['alcohol_bev']>24), data['alcohol_bev'].max()))
    
    #Are you in quarantine
    assert is_numeric_dtype(data['quarantine'])
    assert data['quarantine'].isin([0,1,np.nan]).all()
    
    #How many days have you been in quarantine
    assert is_numeric_dtype(data['quarantine_days'])
    print('%d impossible quarantine_days, max = %d'
          % (sum((data['redcap_timestamp'] - pd.to_datetime('1/23/20')).dt.days < data['quarantine_days']), 
             data['quarantine_days'].max()))
    
    #Do you have a fever
    assert is_numeric_dtype(data['fever'])
    assert data['fever'].isin([0,1,np.nan]).all()
    
    #Fever severity
    assert is_numeric_dtype(data['feverseverity'])
    assert data['feverseverity'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #Temperature scale
    assert is_numeric_dtype(data['temp_measure'])
    assert data['temp_measure'].isin([1,2,np.nan]).all()
    
    #Fever temp
    assert is_numeric_dtype(data['fever_temp'])
    print('%d Celsius fever_temp values outside of 35 to 43'
          % sum((data.loc[data['temp_measure']==1, 'fever_temp'] < 35) | (data.loc[data['temp_measure']==1, 'fever_temp'] > 43)))
    print('%d Fahrenheit fever_temp values outside of 95 to 109'
          % sum((data.loc[data['temp_measure']==2, 'fever_temp'] < 95) | (data.loc[data['temp_measure']==2, 'fever_temp'] > 109)))
    print('Celsius temps: min = %.1f, max = %.1f'
          % (data.loc[data['temp_measure']==1, 'fever_temp'].min(), data.loc[data['temp_measure']==1, 'fever_temp'].max()))
    print('Fahrenheit temps: min = %.1f, max = %.1f'
          % (data.loc[data['temp_measure']==2, 'fever_temp'].min(), data.loc[data['temp_measure']==2, 'fever_temp'].max()))
    
    #Experiencing respiratory symptoms
    assert is_numeric_dtype(data['respiratory'])
    assert data['respiratory'].isin([0,1,np.nan]).all()
    
    #Respiratory severity
    assert is_numeric_dtype(data['respiratory_severity'])
    assert data['respiratory_severity'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #Tested for COVID
    assert is_numeric_dtype(data['tested'])
    assert data['tested'].isin([0,1,np.nan]).all()
    
    #Diagnosed with COVID
    assert is_numeric_dtype(data['covid_status'])
    assert data['covid_status'].isin([0,1,np.nan]).all()
    
    #How stressed are you
    assert is_numeric_dtype(data['stress'])
    assert data['stress'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How socially isolated do you feel
    assert is_numeric_dtype(data['isolation'])
    assert data['isolation'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How worried are you about your health
    assert is_numeric_dtype(data['worry_health'])
    assert data['worry_health'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How worried are you about your family health
    assert is_numeric_dtype(data['family_health'])
    assert data['family_health'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How worried are you about your community's health
    assert is_numeric_dtype(data['community_1health'])
    assert data['community_1health'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How worried are you about the national/global health crisis
    assert is_numeric_dtype(data['national_health'])
    assert data['national_health'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #How worried are you about financial impact
    assert is_numeric_dtype(data['worry_finances'])
    assert data['worry_finances'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #Worry scale
    assert is_numeric_dtype(data['worry_scale'])
    assert data.loc[data['worry_scale'].notna(), 'worry_scale'].between(5, 35).all()
    assert data.loc[data['covid19_complete']==2, 'worry_scale'].notna().all()
    
    #PANAS scale
    panas_cols = [col for col in data.columns if 'panas' in col]
    for col in panas_cols:
        assert is_numeric_dtype(data[col])
        assert data[col].isin([1,2,3,4,5,np.nan]).all()
    assert is_numeric_dtype(data['PANAS_PA'])
    assert data.loc[data['covid19_complete']==2, 'PANAS_PA'].notna().all()
    assert data.loc[data['PANAS_PA'].notna(), 'PANAS_PA'].between(10, 50).all()
    assert is_numeric_dtype(data['PANAS_NA'])
    assert data.loc[data['covid19_complete']==2, 'PANAS_NA'].notna().all()
    assert data.loc[data['PANAS_NA'].notna(), 'PANAS_NA'].between(10, 50).all()
    
    #Depression scale (PHQ9)
    depression_cols = [col for col in data.columns if 'depression' in col]
    for col in depression_cols:
        assert is_numeric_dtype(data[col])
        assert data[col].isin([0,1,2,3,np.nan]).all()
    assert is_numeric_dtype(data['PHQ9'])
    assert data.loc[data['covid19_complete']==2, 'PHQ9'].notna().all()
    assert data.loc[data['PHQ9'].notna(), 'PHQ9'].between(0, 24).all()
        
    #Survey completion codes
    assert is_numeric_dtype(data['covid19_complete'])
    assert is_numeric_dtype(data['covid19_short_survey_complete'])
    assert data['covid19_complete'].isin([0, 2, np.nan]).all()
    assert data['covid19_short_survey_complete'].isin([0, 2, np.nan]).all()
    assert not data[['covid19_short_survey_complete', 'covid19_complete', 'april_18_complete']].isna().all(axis=1).any()
    assert all(data[['covid19_short_survey_complete', 'covid19_complete', 'april_18_complete']].notna().sum(axis=1) == 1)
    
    #Time in bed
    assert data['TIB_12'].isin([0,1]).all()
    assert is_numeric_dtype(data['TIB'])
    assert (data.loc[data['TIB'].notna(), 'TIB'] <= 24).all()
    assert (data.loc[data['TIB'].notna(), 'TIB'] >= 0).all()
    print('%d Time in bed (TIB) less than 4 or greatert than 12'
           % sum((data['TIB'] < 4) | (data['TIB'] > 12)))
    print('Time in bed (TIB): min = %.1f, max = %.1f'
          % (data['TIB'].min(), data['TIB'].max()))
    
    #Total sleep time
    assert is_numeric_dtype(data['TST'])
    # assert (data.loc[data['TST'].notna(), 'TST'] <= 24).all()
    # assert (data.loc[data['TST'].notna(), 'TST'] >= 0).all()
    print('%d Total sleep time (TST) less than 4 or greatert than 10.5'
           % sum((data['TST'] < 4) | (data['TST'] > 10.5)))
    print('Total sleep time (TST): min = %.1f, max = %.1f'
          % (data['TST'].min(), data['TST'].max()))
    
    #Sleep efficiency
    assert is_numeric_dtype(data['SE'])
    # assert (data.loc[data['SE'].notna(), 'SE'] <= 1).all()
    # assert (data.loc[data['SE'].notna(), 'SE'] >= 0).all()
    print('%d sleep efficiency (SE) less than 0.5' % sum(data['SE'] < 0.5))
    print('Sleep efficiency: min = %.2f, max = %.2f'
          % (data['SE'].min(), data['SE'].max()))
    
    #Time asleep should be less than time in bed
    print('%d time in bed less than total sleep time' % sum(data['TIB'] < data['TST']))
    
    #Binary "did you exercise" variable
    assert data['exercise'].isin([0,1]).all()
    
    
    if report_missing:
        
        print('\nMISSING DATA REPORT\n')
        
        #Only calculate missing from completed surveys
        data.dropna(subset=['redcap_timestamp'], inplace=True)
        
        print('%d missing sub_id' % data['sub_id'].isna().sum())
        print('%d missing todays_date' % data['todays_date'].isna().sum())
        print('%d missing sleepdiary_bedtime' % data['sleepdiary_bedtime'].isna().sum())
        print('%d missing sleepdiary_fallasleep' % data['sleepdiary_fallasleep'].isna().sum())
        print('%d missing sleepdiary_sleeplatency' % data['sleepdiary_sleeplatency'].isna().sum())
        print('%d missing sleepdiary_wakes' % data['sleepdiary_wakes'].isna().sum())
        print('%d missing night_awakening_time' % 
              sum(data['night_awakening_time'].isna() & data['sleepdiary_wakes']!=0))
        print('%d missing sleepdiary_waketime' % data['sleepdiary_waketime'].isna().sum())
        print('%d missing sleepdiary_outofbed' % data['sleepdiary_outofbed'].isna().sum())
        print('%d missing sleepdiary_dreams' % data['sleepdiary_dreams'].isna().sum())
        print('%d missing sleepdiary_nap' % data['sleepdiary_nap'].isna().sum())
        print('%d sleepdiary_naptime missing' 
              % sum(data['sleepdiary_naptime'].isna() & (data['sleepdiary_nap']==1)))
        print('%d missing sleepdiary_fellasleep'
              % sum(~data['covid19_timestamp'].isna() & data['sleepdiary_fellasleep'].isna()))
        print('%d missing cst' % data['cst'].isna().sum())
        print('%d missing step_counter' % data['step_counter'].isna().sum())
        print('%d missing steps' % sum(data['steps'].isna() & data['step_counter']==1))
        print('%d missing leave_house' % data['leave_house'].isna().sum())
        print('%d missing people_contact' % data['people_contact'].isna().sum())
        print('%d missing socialize' % data['socialize'].isna().sum())
        print('%d missing socialize_min'
              % sum(data['socialize_min'].isna() & data['socialize']==1))
        print('%d missing across exercise variables' % data[exercise_vars].isna().all(axis=1).sum())
        print('%d missing alcohol_bev' % data['alcohol_bev'].isna().sum())
        print('%d missing quarantine' % data['quarantine'].isna().sum())
        print('%d missing quarantine_days'
              % sum(data['quarantine_days'].isna() & (data['quarantine']==1)))
        print('%d missing fever' % data['fever'].isna().sum())
        print('%d missing feverseverity'
              % sum(data['feverseverity'].isna() & (data['fever']==1)))
        print('%d missing temp_measure'
              % sum(data['temp_measure'].isna() & (data['fever']==1)))
        print('%d missing fever_temp'
              % sum(data['fever_temp'].isna() & (data['fever_temp']==1)))
        print('%d missing respiratory' % data['respiratory'].isna().sum())
        print('%d missing respiratory_severity' % 
              sum(data['respiratory_severity'].isna() & (data['respiratory']==1)))
        print('%d missing tested' % data['tested'].isna().sum())
        print('%d missing covid_status' % data['covid_status'].isna().sum())
        print('%d missing stress' % data['stress'].isna().sum())
        print('%d missing isolation'
              % sum(data['isolation'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d missing worry_health'
              % sum(data['worry_health'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d missing family_health'
              % sum(data['family_health'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d missing community_1health'
              % sum(data['community_1health'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d missing national_health'
              % sum(data['national_health'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d missing worry_finances'
              % sum(data['worry_finances'].isna() & (~data['covid19_timestamp'].isna())))
        print('%d incomplete panas'
              % sum(data[panas_cols].isna().any(axis=1) & (~data['covid19_timestamp'].isna())))
        print('%d incomplete PHQ9'
              % sum(data[depression_cols].isna().any(axis=1) & (~data['covid19_timestamp'].isna())))
        print('%d missing TST' % data['TST'].isna().sum())
        print('%d missing TIB' % data['TIB'].isna().sum())
        print('%d missing SE' % data['SE'].isna().sum())


def demo_QC(demo, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### DEMOGRAPHICS QC #####\n')
    
    
    #Survey time
    assert not demo['date_time'].isna().any()
    print('%d date_time after today'
          % sum(demo['date_time'] > pd.to_datetime('today')))
    print('Date range: %s to %s' % (demo['date_time'].min(), demo['date_time'].max()))
    
    #Age
    assert is_numeric_dtype(demo['age1'])
    assert demo.loc[demo['age1'].notna(), 'age1'].between(18, 118).all()
    print('Age: min = %d, max= %d' % (demo['age1'].min(), demo['age1'].max()))
    
    #Biological sex
    assert is_numeric_dtype(demo['bio_sex'])
    assert not demo['bio_sex'].isna().any()
    assert demo['bio_sex'].isin([1, 2]).all()
    
    #Gender identity
    assert is_numeric_dtype(demo['preferred_gender'])
    assert demo['preferred_gender'].isin([1,2,3,4,5,np.nan]).all()
    
    #Transgender identity
    assert is_numeric_dtype(demo['transgender2'])
    assert demo['transgender2'].isin([1,2,3,np.nan]).all()
    
    #Gender identity
    assert is_numeric_dtype(demo['sexual_orientation'])
    assert demo['sexual_orientation'].isin([1,2,3,4,5,np.nan]).all()
    
    #Ethnicity
    assert is_numeric_dtype(demo['ethnicity___1'])
    assert is_numeric_dtype(demo['ethnicity___2'])
    assert is_numeric_dtype(demo['ethnicity___3'])
    assert demo['ethnicity___1'].isin([0, 1]).all()
    assert demo['ethnicity___2'].isin([0, 1]).all()
    assert demo['ethnicity___3'].isin([0, 1]).all()
    
    #Race
    for i in range(1,10):
        assert is_numeric_dtype(demo['race1___%d'%i])
        assert demo['race1___%d'%i].isin([0, 1]).all()
        
    #Military status
    assert is_numeric_dtype(demo['military'])
    assert demo['military'].isin([1,2,3]).all()
    
    #Marital status
    assert is_numeric_dtype(demo['marital'])
    assert demo['marital'].isin([1,2,3,4,5]).all()
    
    #Disability status
    for i in range(1,7):
        assert is_numeric_dtype(demo['disability___%d'%i])
        assert demo['disability___%d'%i].isin([0, 1]).all()
    
    #Serious medical problems
    assert is_numeric_dtype(demo['medical'])
    
    #Dependents
    assert is_numeric_dtype(demo['dependents'])
    assert not demo['dependents'].isna().any()
    print('Max dependents: %d' % demo['dependents'].max())
    
    #Dependent children
    assert is_numeric_dtype(demo['dependent_children'])
    print('Max dependent children: %d' % demo['dependent_children'].max())
    
    #How many people are you living with
    assert is_numeric_dtype(demo['housing'])
    assert not demo['housing'].isna().any()
    print('Max housing: %d' % demo['housing'].max())
    
    #Income
    assert is_numeric_dtype(demo['income'])
    assert demo['income'].isin([1,2,3,4,5,6,7, np.nan]).all()
    
    #Education
    assert is_numeric_dtype(demo['education'])
    assert demo['education'].isin([1,2,3,4,5,6]).all()
    
    #Are you a student
    assert is_numeric_dtype(demo['student'])
    assert demo['student'].isin([0,1]).all()
    
    #Are you employed
    assert is_numeric_dtype(demo['employed'])
    assert demo['employed'].isin([0,1, np.nan]).all()
    
    #Are you working from home
    assert is_numeric_dtype(demo['working_home'])
    assert demo['working_home'].isin([1,2,3, np.nan]).all()
    
    #Has Covid impacted employment stats
    assert is_numeric_dtype(demo['employment_covid'])
    assert demo['employment_covid'].isin([0,1, np.nan]).all()
    
    #Has you institution taken measures in response to COVID
    assert is_numeric_dtype(demo['institution_measures'])
    assert demo['institution_measures'].isin([0, 1, np.nan]).all()
    
    #How long until you think things will return to normal
    assert is_numeric_dtype(demo['normal'])
    print('Max days until normal = %d' % demo.loc[demo['normal_units']==1, 'normal'].max())
    print('Max weeks until normal = %.1f' % demo.loc[demo['normal_units']==2, 'normal'].max())
    print('Max months until normal = %.1f' % demo.loc[demo['normal_units']==3, 'normal'].max())
    print('%d months until normal > 120' % sum(demo.loc[demo['normal_units']==3, 'normal'] > 120))
    
    #Units for normal
    assert is_numeric_dtype(demo['normal_units'])
    assert demo['normal_units'].isin([1,2,3, np.nan]).all()
    
    #Completion code
    assert is_numeric_dtype(demo['covid19_demographics_complete'])
    assert demo['covid19_demographics_complete'].isin([0,2]).all()
    
    
    if report_missing:
        
        print('\nMISSING DATA REPORT\n')
        
        print('%d missing sub_id' % demo['sub_id'].isna().sum())
        print('%d missing preferred_gender' % demo['preferred_gender'].isna().sum())
        print('%d missing gender_description' 
              % demo.loc[demo['preferred_gender']==4, 'gender_description'].isna().sum())
        print('%d missing transgender2' % demo['transgender2'].isna().sum())
        print('%d missing sexual_orientation' % demo['sexual_orientation'].isna().sum())
        print('%d missing so_description'
              % demo.loc[demo['sexual_orientation']==4, 'so_description'].isna().sum())
        print('%d missing medical_description'
              % demo.loc[demo['medical']==1, 'medical_description'].isna().sum())
        print('%d missing dependent_children' % demo['dependent_children'].isna().sum())
        print('%d missing income' % demo['income'].isna().sum())
        print('%d missing school' % demo.loc[demo['student']==1, 'school'].isna().sum())
        print('%d missing year_study' % demo.loc[demo['student']==1, 'year_study'].isna().sum())
        print('%d missing employed' % demo.loc[demo['student']==0, 'employed'].isna().sum())
        print('%d missing occupation' % demo.loc[demo['employed']==1, 'occupation'].isna().sum())
        print('%d missing working_home' % demo.loc[demo['employed']==1, 'working_home'].isna().sum())
        print('%d missing employment_covid' % demo.loc[demo['employed']==1, 'employment_covid'].isna().sum())
        print('%d missing institution_description' 
              % demo.loc[demo['institution_measures']==1, 'institution_describe'].isna().sum())
        print('%d missing normal' % demo['normal'].isna().sum())
        print('%d missing normal_units' % demo.loc[~demo['normal'].isna(), 'normal_units'].isna().sum())
    


def R1_QC(r1, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### ROUND 1 QC #####\n')
    
    #Timestample variables
    assert r1['round_1_timestamp'].max() < pd.to_datetime('today')
    assert r1['round_1_timestamp'].min() > pd.to_datetime('5/1/2020')
    assert r1['date_time_rd1'].max() < pd.to_datetime('today')
    assert r1['date_time_rd1'].min() > pd.to_datetime('5/1/2020')
    print('%d round_1_timestamp and date_time_rd1 greater than 24 hours different'
          % sum(abs(r1['round_1_timestamp'] - r1['date_time_rd1']) > pd.Timedelta(24, unit='hours')))
    
    #Clock time variables
    r1_clock_vars = ['psqi_1', 'psqi_3', 'mtq_p3', 'mtq_p4', 'mtq_p5', 'mtq_p6', 
                     'mtq_3', 'mtq_4', 'mtq_5', 'mtq_6']
    assert r1[r1_clock_vars].min().min() >= pd.Timedelta(0)
    assert r1[r1_clock_vars].max().max() <= pd.Timedelta(24, unit='hours')
    
    #PSQI
    print('PSQI 2 (minutes to fall asleep): min=%.0f, max=%.0f' % (r1['psqi_2'].min(), r1['psqi_2'].max()))
    print('PSQI 4 (hours sleep): min=%.0f, max=%.0f' % (r1['psqi_4'].min(), r1['psqi_4'].max()))
    #PSQI categorical    
    assert r1.loc[:, 'psqi_5a':'psqi_5j'].isin([np.nan, 0,1,2,3]).all().all()
    assert r1.loc[:, 'psqi_6':'psqi_9'].isin([np.nan, 0,1,2,3]).all().all()
    
    #Munich ChronoType Questionnaire
    assert is_numeric_dtype(r1['mtq_precovid_workdays'])
    assert all(r1['mtq_precovid_workdays'].between(0,7) | r1['mtq_precovid_workdays'].isna())
    assert is_numeric_dtype(r1['mtq_precovid_freedays'])
    assert all(r1['mtq_precovid_freedays'].between(0,7) | r1['mtq_precovid_freedays'].isna())
    assert all(r1['mtq_precovid_workday_sleeponset'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_workday_sleeponset'].isna())
    assert all(r1['mtq_precovid_workday_sleepend'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_workday_sleepend'].isna())
    assert all(r1['mtq_precovid_freeday_sleeponset'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_freeday_sleeponset'].isna())
    assert all(r1['mtq_precovid_freeday_sleepend'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_freeday_sleepend'].isna())
    assert is_numeric_dtype(r1['mtq_precovid_workday_sleepduration'])
    assert all(r1['mtq_precovid_workday_sleepduration'].between(0, 24)
               | r1['mtq_precovid_workday_sleepduration'].isna())
    print('%.1f%% of PreCOVID work day sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_precovid_workday_sleepduration'].between(4, 10)
              & r1['mtq_precovid_workday_sleepduration'].notna()).mean()*100))
    assert is_numeric_dtype(r1['mtq_precovid_freeday_sleepduration'])
    assert all(r1['mtq_precovid_freeday_sleepduration'].between(0, 24)
               | r1['mtq_precovid_freeday_sleepduration'].isna())
    print('%.1f%% of PreCOVID free day sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_precovid_freeday_sleepduration'].between(4, 10)
              & r1['mtq_precovid_freeday_sleepduration'].notna()).mean()*100))
    assert all(r1['mtq_precovid_workday_sleepmidpoint'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_workday_sleepmidpoint'].isna())
    assert all(r1['mtq_precovid_freeday_sleepmidpoint'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_freeday_sleepmidpoint'].isna())
    assert is_numeric_dtype(r1['mtq_precovid_avg_wk_sleepduration'])
    assert all(r1['mtq_precovid_avg_wk_sleepduration'].between(0, 24)
               | r1['mtq_precovid_avg_wk_sleepduration'].isna())
    print('%.1f%% of PreCOVID average sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_precovid_avg_wk_sleepduration'].between(4, 10)
              & r1['mtq_precovid_avg_wk_sleepduration'].notna()).mean()*100))
    assert is_numeric_dtype(r1['mtq_postcovid_workdays'])
    assert all(r1['mtq_postcovid_workdays'].between(0,7) | r1['mtq_postcovid_workdays'].isna())
    assert is_numeric_dtype(r1['mtq_postcovid_freedays'])
    assert all(r1['mtq_postcovid_freedays'].between(0,7) | r1['mtq_postcovid_freedays'].isna())
    assert all(r1['mtq_postcovid_workday_sleeponset'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_workday_sleeponset'].isna())
    assert all(r1['mtq_postcovid_workday_sleepend'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_workday_sleepend'].isna())
    assert all(r1['mtq_postcovid_freeday_sleeponset'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_freeday_sleeponset'].isna())
    assert all(r1['mtq_postcovid_freeday_sleepend'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_freeday_sleepend'].isna())
    assert is_numeric_dtype(r1['mtq_postcovid_workday_sleepduration'])
    assert all(r1['mtq_postcovid_workday_sleepduration'].between(0, 24)
               | r1['mtq_postcovid_workday_sleepduration'].isna())
    print('%.1f%% of postcovid work day sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_postcovid_workday_sleepduration'].between(4, 10)
              & r1['mtq_postcovid_workday_sleepduration'].notna()).mean()*100))
    assert is_numeric_dtype(r1['mtq_postcovid_freeday_sleepduration'])
    assert all(r1['mtq_postcovid_freeday_sleepduration'].between(0, 24)
               | r1['mtq_postcovid_freeday_sleepduration'].isna())
    print('%.1f%% of postcovid free day sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_postcovid_freeday_sleepduration'].between(4, 10)
              & r1['mtq_postcovid_freeday_sleepduration'].notna()).mean()*100))
    assert all(r1['mtq_postcovid_workday_sleepmidpoint'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_workday_sleepmidpoint'].isna())
    assert all(r1['mtq_postcovid_freeday_sleepmidpoint'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_freeday_sleepmidpoint'].isna())
    assert is_numeric_dtype(r1['mtq_postcovid_avg_wk_sleepduration'])
    assert all(r1['mtq_postcovid_avg_wk_sleepduration'].between(0, 24)
               | r1['mtq_postcovid_avg_wk_sleepduration'].isna())
    print('%.1f%% of postcovid average sleep duration outside of 4-10 hours' 
          % ((~r1['mtq_postcovid_avg_wk_sleepduration'].between(4, 10)
              & r1['mtq_postcovid_avg_wk_sleepduration'].notna()).mean()*100))
    assert all(r1['mtq_precovid_chronotype'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_precovid_chronotype'].isna())
    print('%.1f%% of pre-COVID chronotype between 8pm and 8am'
          % (r1['mtq_precovid_chronotype'].between(pd.Timedelta(0), pd.Timedelta(8, unit='hours')).mean()*100))
    assert all(r1['mtq_postcovid_chronotype'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r1['mtq_postcovid_chronotype'].isna())
    print('%.1f%% of post-COVID chronotype between 8pm and 8am'
          % (r1['mtq_postcovid_chronotype'].between(pd.Timedelta(0), pd.Timedelta(8, unit='hours')).mean()*100))
    
    #GAD-7
    gad_cols = [x for x in r1.columns if re.fullmatch('gad_\d', x)]
    assert r1[gad_cols].isin([np.nan, 0, 1, 2, 3]).all().all()
    
    #Cognitive Emotion Regulation Questionnaire_short
    cerq_cols = [x for x in r1.columns if 'cerq' in x]
    assert r1[cerq_cols].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    
    #Liebowitz Social Anxiety Scale
    assert r1.loc[:, 'telephone_fear':'salesperson2_avoid'].isin([np.nan, 0, 1, 2, 3]).all().all()
    
    #The Big Five Inventory–2 Short Form
    big5_cols = [x for x in r1.columns if 'big5' in x]
    assert r1[big5_cols].isin([np.nan, 1,2,3,4,5]).all().all()
    
    #completion code
    assert r1['round_1_complete'].isin([0,2]).all()
    
    
    if report_missing:
        
        print('\nR1 MISSING DATA REPORT\n')
        
        skip_vars = ['redcap_survey_identifier', 'psqi_5j', 'psqi_5j2']
        for col in r1.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r1[col].isna().sum(), col))
        


def R2_QC(r2, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### ROUND 2 QC #####\n')
    
    #Timestamp vars
    assert r2['round_2_timestamp'].min() > pd.to_datetime('6/15/2020')
    assert r2['round_2_timestamp'].max() < pd.to_datetime('today')
    assert r2['date_time_rd2'].min() > pd.to_datetime('6/14/2020')
    assert r2['date_time_rd2'].max() < pd.to_datetime('today')
    print('%d round_2_timestamp and date_time_rd2 greater than 24 hours different'
          % sum(abs(r2['round_2_timestamp'] - r2['date_time_rd2']) > pd.Timedelta(24, unit='hours')))
    
    #Dates
    r2_date_vars = ['stayhome_begin_us', 'stayhome_end_us', 'stayhome_begin', 
                    'stayhome_end', 'normal_date', 'mask_date',
                    'meetings_date', 'bigevents_date', 'shaking_hands_date']
    for col in r2_date_vars:
        print('%s: min=%s, max=%s' % (col, r2[col].min().date(), r2[col].max().date()))
    
    #COVID memory
    assert r2[['gen_1', 'gen_2']].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    yesno_qs = ['history', 'work_start', 'work_close', 'work_other', 
                'work_close_rem', 'school_kids', 'school_close', 'school_close_rem', 
                'neg_emo', 'us', 'stayhome_us', 'stayhome']
    assert r2[yesno_qs].isin([np.nan, 0, 1]).all().all()
    assert r2[[x for x in r2.columns 
               if any(y in x for y 
                      in ['vivid', 'reexp', 'occarousing', 'refarousing'])]].isin([np.nan, 1, 2, 3, 4]).all().all()
    assert r2[[x for x in r2.columns if 'thirdper' in x]].isin([np.nan, 1, 2]).all().all()
    for col in ['num_march', 'num_april', 'num_august', 'num_march_us', 'num_april_us']:
        assert is_numeric_dtype(r2[col])
        print('%s: min=%d, max=%d' % (col, r2[col].min(), r2[col].max()))
    assert r2['severity_state'].isin([np.nan, 1, 2, 3]).all().all()
    assert r2['warmer'].isin([np.nan, 1, 2, 3]).all().all()
    assert r2[[x for x in r2.columns
               if re.fullmatch('sp_mem_\d', x)]].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    assert r2[[x for x in r2.columns 
               if re.fullmatch('fut_\d', x)]].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    assert r2[['peak_pos', 'peak_neg']].isin([np.nan, 1, 2, 3, 4, 5, 6, 7, 8]).all().all()
    assert r2[['peak_pos_intense', 'peak_neg_intense']].isin([np.nan, 1, 2, 3]).all().all()
    assert r2[[x for x in r2.columns if 'sustained' in x]].isin([np.nan, 0, 1]).all().all()
    
    #Insomnia severity index
    isi_vars = [x for x in r2.columns if re.fullmatch('isi_\d', x)]
    assert r2[isi_vars].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    
    #Morningness-Eveningness Questionnaire
    assert r2['meq_1'].isin([np.nan, 0, 1, 2, 3, 4, 5]).all()
    assert r2['meq_2'].isin([np.nan, 1, 2, 3, 4]).all()
    assert r2['meq_3'].isin([np.nan, 1, 2, 3, 4, 5]).all()
    assert r2['meq_4'].isin([np.nan, 1, 2, 3, 4, 5]).all()
    assert r2['meq_5'].isin([np.nan, 0, 2, 4, 6]).all()
    
    #Perceived Stress Scale
    pss_vars = [x for x in r2.columns if re.match('pss_\d', x)]
    assert r2[pss_vars].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    
    #Toronto Empathy Questionnaire
    teq_vars = [x for x in r2.columns if re.match('teq_\d', x)]
    assert r2[teq_vars].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    
    #Completion code
    assert r2['round_2_complete'].isin([0,2]).all()
        
    
    if report_missing:
    
        print('\nR2 MISSING DATA REPORT\n')
        
        skip_vars = []
        for col in r2.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r2[col].isna().sum(), col))


def R3_QC(r3, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### ROUND 3 QC #####\n')
    
    #Timestamp vars
    assert r3['round_3_timestamp'].min() > pd.to_datetime('6/15/2020')
    assert r3['round_3_timestamp'].max() < pd.to_datetime('today')
    assert r3['date_time_rd3'].min() > pd.to_datetime('6/14/2020')
    assert r3['date_time_rd3'].max() < pd.to_datetime('today')
    print('%d round_2_timestamp and date_time_rd2 greater than 24 hours different'
          % sum(abs(r3['round_3_timestamp'] - r3['date_time_rd3']) > pd.Timedelta(24, unit='hours')))
    
    #Brief self-control scale
    bscs_vars = [x for x in r3.columns if re.fullmatch('bscs_\d*', x)]
    assert r3[bscs_vars].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    
    #Short impulsive behavior scale
    sibs_vars = [x for x in r3.columns if re.fullmatch('sibs_\d*', x)]
    assert r3[sibs_vars].isin([np.nan, 1, 2, 3, 4]).all().all()
    
    #Intolerance of uncertainty scale
    iu_vars = [x for x in r3.columns if re.fullmatch('iu_\d*', x)]
    assert r3[iu_vars].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    
    #Emotion regulation questionnaire
    erq_vars = [x for x in r3.columns if re.fullmatch('erq_\d*', x)]
    assert r3[erq_vars].isin([np.nan, 1, 2, 3, 4, 5, 6, 7]).all().all()
    
    #Age
    assert is_numeric_dtype(r3['age'])
    assert all(r3['age'].isna() | r3['age'].between(18, 120))
    print('Round 3 age: min=%d, max=%d' % (r3['age'].min(), r3['age'].max()))
    
    #High risk
    highrisk_vars = ['highrisk_self']
    highrisk_vars += [x for x in r3.columns if re.fullmatch('highrisk_check___\d', x)]
    highrisk_vars += ['highrisk_other_2', 'highrisk_other']
    assert r3[highrisk_vars].isin([np.nan, 0,1]).all().all()
    
    #YES(1) NO(0) variables
    yesno_vars = ['med_quar','shelter_quar','self_quar','covid_test','covid_doctor',
                  'covid_belief','covid_roommate','covid_roommate_2','covid_loved',
                  'covid_loved_2','perished','perished_2','night_shift','essential',
                  'homework','vol_self_iso','goods_scarcity','med_scarcity','ms_using',
                  'mental_health_2','pet','parent','children','mil_time','mistakes',
                  'dream_opt','covid_dream','covdream_scare','covdream_scare_2']
    yesno_vars += [x for x in r3.columns if re.fullmatch('job___\d', x)]
    yesno_vars += [x for x in r3.columns if re.fullmatch('med_history___\d*', x)]
    yesno_vars += [x for x in r3.columns if re.fullmatch('psych_history___\d*', x)]
    yesno_vars += [x for x in r3.columns if re.fullmatch('psych_history_2___\d*', x)]
    yesno_vars += [x for x in r3.columns if re.fullmatch('child_ages___\d*', x)]
    assert r3[yesno_vars].isin([np.nan, 0, 1]).all().all()
    
    #TO DO: Date variables
    # r3_date_vars = [x for x in r3.columns if x.endswith('_start') or x.endswith('_end')]
    # for col in r3_date_vars:
    #     print('%s: min=%s, max=%s' % (col, str(r3[col].min())[0:10], str(r3[col].max())[0:10]))
    
    #Some exit survey likert vars
    likert4_vars = ['pandemic_serious']
    assert r3[likert4_vars].isin([np.nan, 1, 2, 3, 4]).all().all()
    likert5_vars = ['severity_cov']
    likert5_vars += [x for x in r3.columns if re.fullmatch('covpos_\d', x)]
    likert5_vars += r3.loc[:, 'sleepaids':'sleep_change'].columns.to_list()
    assert r3[likert5_vars].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    likert7_vars = ['experience', 'job_impact', 'cov']
    likert7_vars += [x for x in r3.columns if re.fullmatch('mw_\d', x)]
    assert r3[likert7_vars].isin([np.nan, 1, 2, 3, 4, 5, 6, 7]).all().all()
    
    #Three option
    three_vars = ['exposure', 'financial_impact',
                  'no_gs_1', 'no_gs_2', 'no_gs_3',
                  'charity', 'bedtime_change', 'waketime_change_2', 'med_health',
                  'mental_health', 'mh_treatment']
    assert r3[three_vars].isin([np.nan, 1, 2, 3]).all().all()
    
    #Occupation categories
    assert r3['occupation'].isin([np.nan]+list(range(1,29))).all()
    
    #Some more exit survey variables
    sd_vars = [x for x in r3.columns if re.fullmatch('sd_\d*', x)]
    assert r3[sd_vars].isin([np.nan, 1, 2, 3, 4]).all().all()
    assert r3.loc[:, 'travel_air':'mask_serious'].isin([np.nan, 1, 2, 3, 4]).all().all()
    gs_vars = [x for x in r3.columns if re.fullmatch('gs_\d', x)]
    assert r3[gs_vars].isin([np.nan, 1, 2, 3, 4, 5, 6, 7, 8]).all().all()
    ms_vars = [x for x in r3.columns if re.fullmatch('ms\D*_\d', x)]
    assert r3[ms_vars].isin([np.nan, 1, 2, 3, 4, 5, 6]).all().all()
    
    #Isolation variables
    self_iso_vars = [x for x in r3.columns if re.fullmatch('self_iso_\d', x)]
    assert r3[self_iso_vars].isin([np.nan, 1, 2, 3, 4, 5]).all().all()
    no_iso_vars = [x for x in r3.columns if re.fullmatch('no_iso_\d', x)]
    assert r3[no_iso_vars].isin([np.nan, 1, 2, 3, 4, 5, 6, 7, 8]).all().all()
    
    #How many children with you at home?
    assert is_numeric_dtype(r3['how_many_kids'])
    assert r3['how_many_kids'].max() < 10
    print('Kids at home: min=%d, max=%d' % (r3['how_many_kids'].min(), r3['how_many_kids'].max()))
    
    #Fluency vars
    fluency_vars = ['fluency', 'fluency_diff']
    assert r3[fluency_vars].isin([np.nan, 1, 2, 3, 4]).all().all()
    
    #Lucid dreaming scale
    luc_vars = [x for x in r3.columns if re.fullmatch('luc_\d*', x)]
    assert r3[luc_vars].isin([np.nan, 0, 1, 2, 3, 4, 5]).all().all()
    
    #Dream PANAS scale
    pandr_vars = [x for x in r3.columns if re.fullmatch('pandr_\d*', x)]
    assert r3[pandr_vars].isin([np.nan, 0, 1, 2, 3, 4]).all().all()
    
    #completion code
    assert r3['round_3_complete'].isin([0,2]).all()
    
    for col in ['BSCS_Total','SUPPS_Neg_Urg','SUPPS_Lack_Pers','SUPPS_Lack_Premed',
                'SUPPS_Sen_Seek','SUPPS_Pos_Urg','IU_PA','IU_IA','IU_Total','ERQ_Cog_Reapp',
                'ERQ_Exp_Supp','COVID_Pos_Total','Pos_Social_Behavior_Total','Lucidity_Insight',
                'Lucidity_Control','Lucidity_Thought','Lucidity_realism','Lucidity_Memory',
                'Lucidity_Dissociation','Lucidity_Neg_emotion','Lucidity_Pos_emotion',
                'Dream_PANAS_PA','Dream_PANAS_NA','MW_Deliberate','MW_Spontaneous']:
        assert is_numeric_dtype(r3[col])
        
    
    if report_missing:
    
        print('\nR3 MISSING DATA REPORT\n')
        
        skip_vars = []
        for col in r3.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r3[col].isna().sum(), col))


def R4_QC(r4, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### ROUND 4 QC #####\n')
    
    #Timestamp vars
    assert r4['round_4_timestamp'].min() > pd.to_datetime('9/27/2020')
    assert r4['round_4_timestamp'].max() < pd.to_datetime('today')
    assert r4['date_time_rd4'].min() > pd.to_datetime('9/27/2020')
    assert r4['date_time_rd4'].max() < pd.to_datetime('today')
    print('%d round_4_timestamp and date_time_rd4 greater than 24 hours different'
          % sum(abs(r4['round_4_timestamp'] - r4['date_time_rd4']) > pd.Timedelta(24, unit='hours')))
    
    
    #PSQI
    assert all(r4['fall_psqi_1'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) 
               | r4['fall_psqi_1'].isna())
    assert is_numeric_dtype(r4['fall_psqi_2'])
    assert all(r4['fall_psqi_2'].between(0, 8*60) | r4['fall_psqi_2'].isna())
    assert all(r4['fall_psqi_3'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours')) 
               | r4['fall_psqi_3'].isna())
    assert all(r4['fall_psqi_4'].between(0, 24) | r4['fall_psqi_4'].isna())
    print('%d hours sleep (Fall PSQI 4) outisde 5-12 hours' % 
          np.sum(~r4['fall_psqi_4'].between(5, 12) & r4['fall_psqi_4'].notna()))
    assert r4.loc[:, 'fall_psqi_5a':'fall_psqi_5j'].isin([0,1,2,3,np.nan]).all().all()
    assert r4.loc[:, 'fall_psqi_6':'fall_psqi_9'].isin([0,1,2,3,np.nan]).all().all()
    assert all(r4['fall_PSQI_sleep_eff'].between(0,1) | r4['fall_PSQI_sleep_eff'].isna())
    
    #Insomnity severity index
    assert r4.loc[:, [x for x in r4.columns if '_isi_' in x]].isin([0, 1, 2, 3, 4, np.nan]).all().all()
    
    #Munich chronotype questionnaire
    assert r4['fall_mtq_1'].isin([0,1,np.nan]).all()
    assert is_numeric_dtype(r4['fall_mtq_2'])
    assert all(r4['fall_mtq_2'].between(0,7) | r4['fall_mtq_2'].isna())
    for col in r4.loc[:, 'fall_mtq_3':'fall_mtq_6']:
        assert all(r4[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
                   | r4[col].isna())
    assert all(r4['fall_mtq_workday_sleepduration'].between(0,24) 
               | r4['fall_mtq_workday_sleepduration'].isna())
    print('%d MTQ work sleep duration outside 6 to 10 hours'
          % np.sum(~r4['fall_mtq_workday_sleepduration'].between(5,11) 
                   & r4['fall_mtq_workday_sleepduration'].notna()))
    assert all(r4['fall_mtq_freeday_sleepduration'].between(0,24) 
               | r4['fall_mtq_freeday_sleepduration'].isna())
    print('%d MTQ free sleep duration outside 6 to 10 hours'
          % np.sum(~r4['fall_mtq_freeday_sleepduration'].between(5,11) 
                   & r4['fall_mtq_freeday_sleepduration'].notna()))
    assert all(r4['fall_mtq_workday_sleepmidpoint'].between(pd.Timedelta(0), pd.Timedelta(24, unit='days')) 
               | r4['fall_mtq_workday_sleepmidpoint'].isna())
    
    #GAD
    assert r4[[x for x in r4.columns if re.fullmatch('fall_gad_\d', x)]].isin([0, 1, 2, 3, np.nan]).all().all()
    
    #Memory questions
    assert r4[[col for col in r4.columns if col.startswith('gen_')]].isin([1,2,3,4,5,np.nan]).all().all()
    mem_rating_vars = [col for col in r4.columns 
                       if col.startswith('vivid_')
                       or col.startswith('occarousing_')
                       or col.startswith('refarousing_')]
    assert r4[mem_rating_vars].isin([1,2,3,4,np.nan]).all().all()
    mem_yesno_vars = ['history_fut', 'work_start_fut', 'work_close_fut', 
                      'work_other_fut', 'work_close_rem_fut', 'school_kids_fut',
                      'school_close_fut', 'school_close_rem_fut', 'neg_emo_fut',
                      'get_better_fut', 'us_fut', 'stayhome_us_fut', 'stayhome_fut']
    assert r4[mem_yesno_vars].isin([0,1,np.nan]).all().all()
    assert r4[[col for col in r4.columns if col.startswith('thirdper_')]].isin([1,2,np.nan]).all().all()
    assert is_numeric_dtype(r4['num_march_fut'])
    assert is_numeric_dtype(r4['num_april_fut'])
    assert is_numeric_dtype(r4['num_march_us_fut'])
    assert is_numeric_dtype(r4['num_april_us_fut'])
    assert (r4[['num_march_fut', 'num_april_fut', 'us_fut', 
                'num_march_us_fut', 'num_april_us_fut']].min().min() >= 0)
    assert (r4[['num_march_fut', 'num_april_fut', 'us_fut', 
                'num_march_us_fut', 'num_april_us_fut']].max().max() < 2e9)
    r4_date_vars = ['stayhome_begin_us_fut', 'stayhome_end_us_fut', 
                    'stayhome_begin_fut', 'stayhome_end_fut']
    for col in r4_date_vars:
        print('%d %s before 1/1/2020' % (sum(r4[col]<pd.Timestamp('1/1/2020')), col))
        print('%d %s after 12/31/2020' % (sum(r4[col]>pd.Timestamp('12/31/2020')), col))
    assert r4[['severity_state_fut', 'severity_fut']].isin([1,2,3,np.nan]).all().all()
    assert r4[[col for col in r4.columns if col.startswith('sp_mem')]].isin([0,1,2,3,4,np.nan]).all().all()
    assert r4[[col for col in r4.columns if col.endswith('_change')]].isin([1,2,3,np.nan]).all().all()
    assert r4[['peak_neg_fut', 'peak_pos_fut']].isin([1,2,3,4,5,6,7,8,np.nan]).all().all()
    assert r4[['peak_neg_intense_fut', 'peak_pos_intense_fut']].isin([1,2,3,np.nan]).all().all()
    assert r4.loc[:, 'sustained_neg_fut___1':'sustained_pos_fut___8'].isin([0,1,np.nan]).all().all()
    assert r4[[col for col in r4.columns if '_repeat_' in col]].isin([1,2,3,4,np.nan]).all().all()
    assert r4[[col for col in r4.columns if col.startswith('important_')]].isin([1,2,3,4,np.nan]).all().all()
    
    #Memory individual differences
    assert r4.loc[:, [x for x in r4.columns if re.match('ind_diff_\d', x)]].isin([1, 2, 3, 4, np.nan]).all().all()
    
    #Prosocial
    assert r4.loc[:, [x for x in r4.columns if re.fullmatch('prosoc_\d', x)]].isin([0,1,np.nan]).all().all()
    assert r4.loc[:, [x for x in r4.columns if re.match('prosoc_\d_', x)]].isin([1,2,3,4,5,np.nan]).all().all()
    
    #Tightness-looseness norms
    assert r4.loc[:, [x for x in r4.columns if re.match('soc_norm_\d', x)]].isin([1,2,3,4,5,6,np.nan]).all().all()
    
    #News
    assert r4.loc[:, [x for x in r4.columns if re.fullmatch('news_\d', x)]].isin([1,2,3,np.nan]).all().all()
    assert r4.loc[:, [x for x in r4.columns if re.match('news_\D', x)]].isin([1,2,3,4,5,np.nan]).all().all()
    
    #Moral circle
    assert r4['moral_circle'].isin(list(range(1,17))+[np.nan]).all()
    
    #COVID variables
    assert r4.loc[:, 'covid_test':'covid_belief'].isin([0,1,np.nan]).all().all()
    assert r4.loc[:, 'covid_roommate':'covid_loved_2'].isin([0,1,np.nan]).all().all()
    assert r4['severity_cov'].isin([1,2,3,4,5,np.nan]).all()
    assert r4['political'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    
    if report_missing:
    
        print('\nR4 MISSING DATA REPORT\n')
        
        skip_vars = []
        for col in r4.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r4[col].isna().sum(), col))



def R5_QC(r5, report_missing=False):
    """
    Check for data problems
    """
    
    print('\n\n##### ROUND 5 QC #####\n')
    
    #Timestamp vars
    assert r5['round_5_timestamp'].min() > pd.to_datetime('2/26/2021')
    assert r5['round_5_timestamp'].max() < pd.to_datetime('4/8/2021')
    assert r5['date_time_rd5'].min() > pd.to_datetime('2/26/2021')
    assert r5['date_time_rd5'].max() < pd.to_datetime('4/8/2021')
    print('%d round_5_timestamp and date_time_rd5 greater than 24 hours different'
          % sum(abs(r5['round_5_timestamp'] - r5['date_time_rd5']) > pd.Timedelta(24, unit='hours')))
    
    #Time vars
    r5_time_vars = ['psqi_1', 'psqi_3', 'mtq_3', 'mtq_p8', 'mtq_p9', 'mtq_p10']
    for col in r5_time_vars:
        assert all(r5[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
                   | r5[col].isna())
    
    #PSQI
    assert all(r5['psqi_1'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r5['psqi_1'].isna())
    assert is_numeric_dtype(r5['psqi_2'])
    assert all(r5['psqi_2'].between(0, 8*60) | r5['psqi_2'].isna())
    assert all(r5['psqi_3'].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
               | r5['psqi_3'].isna())
    assert is_numeric_dtype(r5['psqi_4'])
    assert all(r5['psqi_4'].between(0, 16) | r5['psqi_4'].isna())
    assert r5.loc[:, 'psqi_5a':'psqi_5j'].isin([0, 1, 2, 3, np.nan]).all().all()
    assert r5.loc[:, 'psqi_6':'psqi_9'].isin([0, 1, 2, 3, np.nan]).all().all()
    
    #MTQ
    assert all(r5['mtq_2'].between(0, 7) | r5['mtq_2'].isna())
    
    #ISI
    assert r5[['isi_%d' % i for i in range(1, 8)]].isin([0,1,2,3,4, np.nan]).all().all()
    
    #PROMIS
    assert r5[['promis_%d' % i for i in range(1, 8)]].isin([1,2,3,4,5, np.nan]).all().all()
    
    #GAD-7
    assert r5[['gad_%d' % i for i in range(1, 8)]].isin([0,1,2,3, np.nan]).all().all()
    
    #PSS
    assert r5[['pss_%d' % i for i in range(1, 11)]].isin([0,1,2,3,4, np.nan]).all().all()
    
    #MMQ
    assert r5[['mmq_%d' % i for i in range(1, 19)]].isin([0,1,2,3,4, np.nan]).all().all()
    
    #High risk conditions
    assert r5.loc[:, 'highrisk_self':'highrisk_check___8'].isin([0, 1, np.nan]).all().all()
    assert r5.loc[:, 'highrisk_other_2':'children_feb'].isin([0, 1, np.nan]).all().all()
    
    assert all(r5['how_many_kids_feb'].between(0, 20) | r5['how_many_kids_feb'].isna())
    assert r5.loc[:, 'child_ages_feb___1':'child_ages_feb___8'].isin([0, 1, np.nan]).all().all()
    
    #Yes-no questions
    r5_yesno = ['mtq_1', 'highrisk_self', 'highrisk_check___1', 'highrisk_check___2', 
                'highrisk_check___3', 'highrisk_check___4', 'highrisk_check___5', 
                'highrisk_check___6', 'highrisk_check___7', 'highrisk_check___8', 
                'highrisk_other_2', 'highrisk_other', 'parent_feb', 'children_feb', 
                'child_ages_feb___1', 'child_ages_feb___2', 'child_ages_feb___3', 
                'child_ages_feb___4', 'child_ages_feb___5', 'child_ages_feb___6', 
                'child_ages_feb___7', 'child_ages_feb___8', 'fall_babysit', 
                'sp2021_babysit', 'covid_test', 'covid_doctor', 'covid_belief', 
                'cov_long_haul', 'covid_roommate', 'covid_roommate_2', 'covid_loved', 
                'covid_loved_2', 'perished', 'perished_2', 'vaccination', 'vacc_safety', 
                'vacc_plan', 'novacc_safety', 'ace_1', 'ace_2', 'ace_3', 'ace_4', 
                'ace_5', 'ace_6', 'ace_7', 'ace_8', 'ace_9', 'ace_10', 'ace_11', 
                'ace_12', 'ace_13', 'ace_14', 'ace_15', 'ace_16', 'ace_17', 'ace_18']
    assert r5[r5_yesno].isin([0, 1, np.nan]).all().all()
    
    #Children
    assert is_numeric_dtype(r5['how_many_kids_feb'])
    assert all(r5['how_many_kids_feb'].between(0,15) | r5['how_many_kids_feb'].isna())
    #Fall
    print('%d fall_babysit_kids greater than how_many_kids' % 
          sum(r5['fall_babysit_kids'] > r5['how_many_kids_feb']))
    assert all(r5['bs_fall'].between(0,7) | r5['bs_fall'].isna())
    assert r5['babysit_format'].isin([0,1,2,3,np.nan]).all()
    #Spring
    print('%d sp2021_babysit_kids greater than how_many_kids' % 
          sum(r5['sp2021_babysit_kids'] > r5['how_many_kids_feb']))
    all(r5['bs_sp2021'].between(0,7) | r5['bs_sp2021'].isna())
    assert r5['sp2021_babysit_format'].isin([0,1,2,3,np.nan]).all()
    
    #COVID questions
    assert r5['severity_cov'].isin([1,2,3,4,np.nan]).all()
    print('Earliest reported COVID data: %s' % r5['date_cov'].min())
    r5_end_date = r5['round_5_timestamp'].max() + pd.Timedelta(1, unit='days')
    print('%d cov_date after data collection' % sum(r5['date_cov'] > r5_end_date))
    assert r5[[col for col in r5.columns if 'covpos_' in col]].isin([1,2,3,4,5,np.nan]).all().all()
    assert r5['experience'].isin([1,2,3,4,5,6,7,np.nan]).all()
    
    #COVID expectations
    for col in [x for x in r5.columns if '_date_feb' in x]:
        if any(r5[col] < pd.to_datetime('3/1/2020')):
            print('%d %s before 3/1/2020' % (any(r5[col] < pd.to_datetime('3/1/2020')), col))
        print('Max %s = %s' % (col, r5[col].max()))
    assert r5[['normal_feb', 'mask_feb', 'meetings_feb', 'bigevents_feb', 
               'shakinghands_feb']].isin([1,2,3,4,np.nan]).all().all()
    assert is_numeric_dtype(r5['num_march21'])
    assert not any(r5['num_march21'] < 0)
    print('%d num_march21 greater than any country population' % sum(r5['num_march21'] > 1.5e9))
    
    #Vaccination
    print('%d vacc_date before 12/14/20' % sum(r5['vacc_date'] < pd.to_datetime('12/14/20')))
    print('%d vacc_date after data collection' % sum(r5['vacc_date'] > r5_end_date))
    assert r5['vacc_doses'].isin([0,1,2, np.nan]).all()
    assert r5['vacc_side_effects'].isin([0,1,2,3,np.nan]).all()
    assert r5[['vacc_safe1','vacc_safe2','vacc_safe3','vacc_safe4','vacc_safe5']].isin([1,2,3,4,5,np.nan]).all().all()
    assert r5[['vacc_safe1','vacc_safe2','vacc_safe3','vacc_safe4','vacc_safe5']].sum(axis=1, skipna=False).isin([15, np.nan]).all()
    assert r5[[x for x in r5.columns if x.startswith('vacc_nosafe')]].isin([1,2,3,4,5,6,7,8,np.nan]).all().all()
    assert r5[[x for x in r5.columns if x.startswith('vacc_nosafe')]].sum(axis=1, skipna=False).isin([np.nan, 36]).all()
    
    #Interpersonal Reactivity Index
    assert r5[[x for x in r5.columns if x.startswith('iri_')]].isin([np.nan,0,1,2,3,4]).all().all()
    
    #Personality Inventory for DSM
    assert r5[[x for x in r5.columns if x.startswith('pid_')]].isin([np.nan,0,1,2,3]).all().all()
    
    #John Henryism Active Coping Scale
    assert r5[[x for x in r5.columns if x.startswith('jhacs_')]].isin([np.nan,0,1,2,3,4,5]).all().all()
    
    assert r5['round_5_complete'].isin([0,2]).all()
    
    
    if report_missing:
    
        print('\nR5 MISSING DATA REPORT\n')
        
        skip_vars = []
        for col in r5.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r5[col].isna().sum(), col))



def R6_QC(r6, report_missing=False):
    """
    Check for data problems and report missing data by variable
    """
    
    print('\n\n##### ROUND 6 QC #####\n')
    
    #Timestamp vars
    assert r6['april_18_timestamp'].min() > pd.to_datetime('4/17/2021')
    assert r6['april_18_timestamp'].max() < pd.to_datetime('4/27/2021')
    assert r6['todays_date'].min() > pd.to_datetime('4/17/2021')
    assert r6['todays_date'].max() < pd.to_datetime('4/21/2021')
    print('%d april_18_timestamp and todays_date greater than 24 hours different'
          % sum(abs(r6['april_18_timestamp'] - r6['todays_date']) > pd.Timedelta(24, unit='hours')))
    
    #LSAS
    assert r6[[x for x in r6.columns if x.endswith('_avoid') or x.endswith('_fear')]].isin([np.nan, 0, 1, 2, 3]).all().all()
    
    #Vaccination
    r6_yesno = ['vaccination', 'vacc_safety', 'vacc_plan', 'novacc_safety', 'covid_test', 
                'covid_doctor', 'covid_belief', 'cov_long_haul', 'covid_roommate', 
                'covid_roommate_2', 'covid_loved', 'covid_loved_2', 'perished', 'perished_2']
    assert r6[r6_yesno].isin([0,1,np.nan]).all().all()
    r6_end_date = r6['april_18_timestamp'].max() + pd.Timedelta(1, unit='days')
    print('%d vacc_date before 12/14/20' % sum(r6['vacc_date'] < pd.to_datetime('12/14/20')))
    print('%d vacc_date after data collection' % sum(r6['vacc_date'] > r6_end_date))
    assert r6['vacc_doses'].isin([0,1,2, np.nan]).all()
    assert r6['vacc_side_effects'].isin([0,1,2,3,np.nan]).all()
    
    #COVID
    assert r6['severity_cov'].isin([1,2,3,4,5,np.nan]).all()
    
    
    if report_missing:
    
        print('\nR6 MISSING DATA REPORT\n')
        
        skip_vars = []
        for col in r6.columns:
            if col in skip_vars:
                continue
            print('%d missing %s' % (r6[col].isna().sum(), col))
