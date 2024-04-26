# -*- coding: utf-8 -*-
"""
Create demographics table for BC COVID-19 data

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

import numpy as np
import pandas as pd

from BC_COVID19_import_data import import_covid19_data


def make_demo_table(demo, r4, subs=None):
    """
    Create detailed demographics table from demographics and Round 4 data
    subs input is a list of subject IDs to include; if it is omitted, all subjects will be used
    """
    
    demo = demo.copy()
    
    #Add US region to demographics
    demo.loc[demo['state'].isin(['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']),
             'US_region'] = 'Northeast'
    demo.loc[demo['state'].isin(['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 
                                 'MO', 'NE', 'ND', 'SD']), 'US_region'] = 'Midwest'
    demo.loc[demo['state'].isin(['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 
                                 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']), 'US_region'] = 'South'
    demo.loc[demo['state'].isin(['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 
                                 'AK', 'CA', 'HI', 'OR', 'WA']), 'US_region'] = 'West'
    
    #Add politics to demographics
    demo = demo.merge(r4[['sub_id', 'political']], how='outer', on='sub_id')
    
    #Update participants with more than one race
    idx = demo.loc[:, 'race1___1':'race1___9'].sum(axis='columns') > 1
    demo.loc[idx, 'race1___7'] = 1
    demo.loc[idx, 'race1___1':'race1___6'] = 0
    demo.loc[idx, 'race1___8':'race1___9'] = 0
    assert sum(demo.loc[:, 'race1___1':'race1___9'].sum(axis='columns') > 1) == 0
    
    #update ethnicity with more than one
    idx = demo.loc[:, 'ethnicity___1':'ethnicity___3'].sum(axis='columns') > 1
    demo.loc[idx, 'ethnicity___1':'ethnicity___3'] = np.nan
    assert sum(demo.loc[:, 'ethnicity___1':'ethnicity___3'].sum(axis='columns') > 1) == 0
    
    #Add informative columns labels for categories
    label_dict = {'race1___1': 'African American',
                  'race1___2': 'Asian',
                  'race1___3': 'White',
                  'race1___4': 'Hispanic/Latinx',
                  'race1___5': 'Native Hawaiian or other Pacific Islander',
                  'race1___6': 'American Indian/Alaska Native',
                  'race1___7': 'more than one race/prefer to self-describe',
                  'race1___8': 'unknown',
                  'race1___9': 'prefer not to say (race)',
                  'ethnicity___1': 'Hispanic',
                  'ethnicity___2': 'not Hispanic',
                  'ethnicity___3': 'prefer not to say (ethnicity)',
                  'disability___1': 'sensory impairment (vision/hearing)',
                  'disability___2': 'mobility impairment ',
                  'disability___3': 'learning disability',
                  'disability___4': 'mental Health Disorder',
                  'disability___5': 'disability or impairment not listed above',
                  'disability___6': 'prefer not to say (disability)'}
    demo.rename(label_dict, axis='columns', inplace=True)
    
    #Get demographics subset
    if subs is None:
        demo_subset = demo
    else:
        demo_subset = demo[demo['sub_id'].isin(subs)].copy()
    
    demo_table = pd.Series(dtype='object')
        
    demo_table.loc['N'] = len(demo_subset)
    
    #Age
    demo_table.loc['Age'] = ''
    for col in demo_subset['age1'].describe().index[1:]:
        demo_table.loc[col] = demo_subset['age1'].describe()[col]
    demo_table.loc['Age: nan'] = demo_subset['age1'].isna().mean()
        
    #Race and ethnicity
    demo_table.loc['Ethnicity'] = ''
    ethnicity_col = demo_subset.loc[:, 'Hispanic':'prefer not to say (ethnicity)'].columns
    assert demo_subset[ethnicity_col].notna().all(axis=1).sum()
    for col in ethnicity_col:
        demo_table.loc[col] = demo_subset[col].mean()
    demo_table.loc['Race'] = ''
    race_col = demo_subset.loc[:, 'African American':'prefer not to say (race)'].columns
    assert demo_subset[race_col].notna().all(axis=1).sum()
    for col in race_col:
        demo_table.loc[col] = demo_subset[col].mean()
    
    #Other variables
    for col in ['preferred_gender', 'bio_sex', 'transgender2', 'sexual_orientation',
                'education', 'marital', 'medical', 'income', 'student', 'employed',
                'political']:
        demo_table.loc[col] = ''
        for cat in demo[col].value_counts(dropna=False).sort_index().index:
            if sum(demo_subset[col]==cat) or (np.isnan(cat) and demo_subset[col].isna().any()):
                demo_table.loc[('%s: %s' % (col, cat))] = demo_subset[col].value_counts(normalize=True, sort=False, dropna=False)[cat]
            else:
                demo_table.loc[('%s: %s' % (col, cat))] = 0.0
    assert not sum(demo_subset['political'] == 7)
    demo_table.loc['political: 7.0'] = 0
    
    #Dependent children
    assert demo_subset['dependents'].notna().all()
    demo_table['How many dependent children live with you?'] = ''
    demo_table['children: 0']  = sum((demo_subset['dependents']==0) | (demo_subset['dependent_children']==0)) / len(demo_subset)
    demo_table['children: 1']  = sum(demo_subset['dependent_children']==1) / len(demo_subset)
    demo_table['children: 2']  = sum(demo_subset['dependent_children']==2) / len(demo_subset)
    demo_table['children: 3']  = sum(demo_subset['dependent_children']==3) / len(demo_subset)
    demo_table['children: 4']  = sum(demo_subset['dependent_children']==4) / len(demo_subset)
    demo_table['children: 5+'] = sum(demo_subset['dependent_children']>4) / len(demo_subset)
    assert not any(demo_subset['dependents'].isna() & demo_subset['dependent_children'].isna()) 
    
    #Housemates
    assert demo_subset['housing'].notna().all()
    demo_table['How many people do you live with?'] = ''
    demo_table['housemates: 0'] = sum(demo_subset['housing']==0) / len(demo_subset)
    demo_table['housemates: 1'] = sum(demo_subset['housing']==1) / len(demo_subset)
    demo_table['housemates: 2'] = sum(demo_subset['housing']==2) / len(demo_subset)
    demo_table['housemates: 3'] = sum(demo_subset['housing']==3) / len(demo_subset)
    demo_table['housemates: 4'] = sum(demo_subset['housing']==4) / len(demo_subset)
    demo_table['housemates: 5+'] = sum(demo_subset['housing']>4) / len(demo_subset)
    assert demo_subset['housing'].notna().all()
    
    #Country
    assert demo_subset['country'].notna().all()
    demo_table.loc['Country of Residence'] = ''
    country_counts = demo['country'].value_counts(dropna=False)
    for cat in country_counts.index:
        if country_counts[cat] >= 10:
            if sum(demo_subset['country']==cat) > 0:
                demo_table.loc[cat.title()] = demo_subset['country'].value_counts(normalize=True, sort=False, dropna=False)[cat]
            else:
                demo_table.loc[cat.title()] = 0
    demo_table['all other countries'] = np.mean(~demo_subset['country'].isin(country_counts[country_counts>=10].index))
    
    #US region
    US_demo = demo_subset[demo_subset['country']=='UNITED STATES']
    demo_table['U.S. Region (U.S. participants only)'] = ''
    for cat in demo.loc[demo['country']=='UNITED STATES', 'US_region'].value_counts().index:
        if US_demo.empty:
            demo_table.loc[('%s: %s' % ('US_region', cat))] = np.nan
        elif sum(US_demo['US_region']==cat) > 0:
            demo_table.loc[('%s: %s' % ('US_region', cat))] = US_demo['US_region'].value_counts(normalize=True, sort=False, dropna=False)[cat]
        else:
            demo_table.loc[('%s: %s' % ('US_region', cat))] = 0
    if US_demo.empty:
        demo_table.loc['US_region: not reported'] = np.nan
    else:
        demo_table.loc['US_region: not reported'] = US_demo['US_region'].isna().sum() / len(US_demo)
    
    #Rename rows
    row_substitutions = {'preferred_gender': 'Gender',
                         'preferred_gender: 1.0': 'female',
                         'preferred_gender: 2.0': 'male',
                         'preferred_gender: 3.0': 'non-binary/third gender',
                         'preferred_gender: 4.0': 'prefer to self-describe',
                         'preferred_gender: 5.0': 'prefer not to say',
                         'bio_sex': 'Biological Sex',
                         'bio_sex: 1.0': 'female',
                         'bio_sex: 2.0': 'male',
                         'transgender2': 'Gender Identity',
                         'transgender2: 2.0': 'cisgender',
                         'transgender2: 1.0': 'transgender',
                         'transgender2: 3.0': 'prefer not to say',
                         'sexual_orientation': 'Sexual Orientation',
                         'sexual_orientation: 3.0': 'straight/heterosexual',
                         'sexual_orientation: 2.0': 'bisexual',
                         'sexual_orientation: 1.0': 'gay/lesbian',
                         'sexual_orientation: 4.0': 'prefer to self-describe',
                         'sexual_orientation: 5.0': 'prefer not to say',
                         'education': 'Education',
                         'education: 6.0': 'graduate, medical, or professional degree',
                         'education: 4.0': "bachelor's degree",
                         'education: 3.0': 'some college',
                         'education: 5.0': 'some post-bachelor',
                         'education: 2.0': 'high school diploma or GED',
                         'education: 1.0': 'some high school',
                         'marital': 'Relationship Status',
                         'marital: 1.0': 'single',
                         'marital: 3.0': 'married',
                         'marital: 2.0': 'in a relationship',
                         'marital: 4.0': 'separated/divorced',
                         'marital: 5.0': 'widowed',
                         'medical': 'Serious medical problems?',
                         'medical: 0.0': 'no',
                         'medical: 1.0': 'yes',
                         'income': 'Income',
                         'income: 1.0': '$0 - 25,000',
                         'income: 2.0': '$25,001 - 50,000',
                         'income: 3.0': '$50,001 - 75,000',
                         'income: 4.0': '$75,001 - 100,000',
                         'income: 5.0': '$100,001 - 150,000',
                         'income: 6.0': '$150,001 - 250,000',
                         'income: 7.0': '$250,000+',
                         'student': 'Are you a full time student?',
                         'student: 0.0': 'no',
                         'student: 1.0': 'yes',
                         'employed': 'Are you currently employed?',
                         'employed: 1.0': 'yes',
                         'employed: 0.0': 'no',
                         'political': 'Political Ideology',
                         'political: 1.0': 'very liberal',
                         'political: 2.0': 'liberal',
                         'political: 3.0': 'slightly liberal',
                         'political: 4.0': 'moderate',
                         'political: 5.0': 'slightly conservative',
                         'political: 6.0': 'conservative',
                         'political: 7.0': 'very conservative',
                         'US_region': 'U.S. Region'}
    demo_table.rename(row_substitutions, inplace=True)
    demo_table.rename({'US_region: nan': 'not reported'}, inplace=True)
    demo_table.index = demo_table.index.str.replace('US_region: ', '')
    demo_table.index = demo_table.index.str.replace('children: ', '')
    demo_table.index = demo_table.index.str.replace('housemates: ', '')
    demo_table.index = ['not reported' if 'nan' in x else x for x in demo_table.index]
    
    return demo_table


def make_grouped_demo_table(demo, r4, subsets):
    """
    Create a demographics table with different columns for different subsets of participants
    subsets input is a dictionary with the name of the subset as the keys and the subject IDs as the values
    """
    
    #Create a demographics table for each subset
    subset_tables = {}
    for group in subsets:
        subset_tables[group] = make_demo_table(demo, r4, subsets[group])
    
    #Merge tables
    demo_table = pd.concat(subset_tables, axis=1)
    
    return demo_table


#%% EXAMPLE

if __name__ == '__main__':
    
    #Full path of of directory containing data csv files
    data_dir = r'D:\COVID19\export'
    
    #Date string at the end of the data csv files
    date_str = '2021-07-22_13_26'
    
    #Import data
    (data, demo, r1, r2, r3, r4, r5, r6) = import_covid19_data(data_dir, date_str)
    
    #Create demographics table
    demo_table = make_grouped_demo_table(demo, r4,
                                        {'All': None,
                                         'US': demo.loc[demo['country']=='UNITED STATES', 'sub_id'],
                                         'International': demo.loc[demo['country']!='UNITED STATES', 'sub_id'],
                                         'R1': r1['sub_id'],
                                         'R2': r2['sub_id'],
                                         'R3': r3['sub_id'],
                                         'R4': r4['sub_id'],
                                         'R5': r5['sub_id'],
                                         'R6': r6['sub_id']})
    
    #Output demographics table to csv file
    demo_table.to_csv('full_demographics.csv')
