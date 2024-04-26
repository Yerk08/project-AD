# -*- coding: utf-8 -*-
"""
Create descriptives table for 
Cunningham, Fields, & Kensinger, Scientific Data

Author: Eric Fields
Version Date: 20 October 2020
"""

from os.path import join
import numpy as np
import pandas as pd

main_dir = r'D:\COVID19'


#%% IMPORT DATA

data = pd.read_csv(join(main_dir, 'export', 'COVID19_combined_cleaned.csv'), index_col='unique_id')
demo = pd.read_csv(join(main_dir, 'export', 'COVID19_demographics_cleaned.csv'), index_col='record_id')
r1 = pd.read_csv(join(main_dir, 'export', 'COVID19_Round1_cleaned.csv'), index_col='record_id')
r2 = pd.read_csv(join(main_dir, 'export', 'COVID19_Round2_cleaned.csv'), index_col='record_id')
r3 = pd.read_csv(join(main_dir, 'export', 'COVID19_Round3_cleaned.csv'), index_col='record_id')


#%% FORMATTING

#Convert ref_dat to datetime
data['ref_date'] = pd.to_datetime(data['ref_date']).dt.date

#Calculate worry composite
data['worry_composite'] = data.loc[:, 'worry_health':'worry_finances'].sum(axis=1, skipna=False)


#%% MAKE DESCRIPTIVES TABLE

out_table = pd.DataFrame()

#Number of days with responses by participant
out_table.at['# daily responses', 'M (SD)'] = '%.2f (%.2f)' % (data['sub_id'].value_counts().mean(),
                                                               data['sub_id'].value_counts().std(ddof=1))
out_table.at['# daily responses', 'Min'] = data['sub_id'].value_counts().min()
out_table.at['# daily responses', '1st Quartile'] = data['sub_id'].value_counts().quantile(0.25)
out_table.at['# daily responses', 'Median'] = data['sub_id'].value_counts().median()
out_table.at['# daily responses', '3rd Quartile'] = data['sub_id'].value_counts().quantile(0.75)
out_table.at['# daily responses', 'Max'] = data['sub_id'].value_counts().max()
out_table.at['# daily responses', 'skew'] = data['sub_id'].value_counts().skew()
out_table.at['# daily responses', 'kurtosis'] = data['sub_id'].value_counts().kurtosis()

#Calculatte descriptives for various measures in the daily data
measures = {'stress': 'stress',
            'isolation': 'social isolation',
            'worry_composite': 'worry composite',
            'PANAS_PA': 'PANAS Positive',
            'PANAS_NA': 'PANAS Negative',
            'PHQ9': 'PHQ-9 (modified)',
            'TST': 'total sleep time',
            'TIB': 'time in bed',
            'SE': 'sleep efficiency'}
for col in measures:
    out_table.at[measures[col], 'M (SD)'] = '%.2f (%.2f)' % (data[col].mean(), data[col].std(ddof=1))
    out_table.at[measures[col], 'Min'] = data[col].min()
    out_table.at[measures[col], '1st Quartile'] = data[col].quantile(0.25)
    out_table.at[measures[col], 'Median'] = data[col].median()
    out_table.at[measures[col], '3rd Quartile'] = data[col].quantile(0.75)
    out_table.at[measures[col], 'Max'] = data[col].max()
    out_table.at[measures[col], 'skew'] = data[col].skew()
    out_table.at[measures[col], 'kurtosis'] = data[col].kurtosis()

#Calculatte descriptives for various measures in the round 1 assessement
measures = {'PSQI_TOTAL': 'Pittsburgh Sleep Quality Index',
            'gad_7_total': 'GAD-7'}
for col in measures:
    out_table.at[measures[col], 'M (SD)'] = '%.2f (%.2f)' % (r1[col].mean(), r1[col].std(ddof=1))
    out_table.at[measures[col], 'Min'] = r1[col].min()
    out_table.at[measures[col], '1st Quartile'] = r1[col].quantile(0.25)
    out_table.at[measures[col], 'Median'] = r1[col].median()
    out_table.at[measures[col], '3rd Quartile'] = r1[col].quantile(0.75)
    out_table.at[measures[col], 'Max'] = r1[col].max()
    out_table.at[measures[col], 'skew'] = r1[col].skew()
    out_table.at[measures[col], 'kurtosis'] = r1[col].kurtosis()

#Calculatte descriptives for various measures in the round 2 assessement    
measures = {'ISI_Total': 'Insomnia Severity Index',
            'PSS_TOTAL': 'Perceived Stress Scale'}
for col in measures:
    out_table.at[measures[col], 'M (SD)'] = '%.2f (%.2f)' % (r2[col].mean(), r2[col].std(ddof=1))
    out_table.at[measures[col], 'Min'] = r2[col].min()
    out_table.at[measures[col], '1st Quartile'] = r2[col].quantile(0.25)
    out_table.at[measures[col], 'Median'] = r2[col].median()
    out_table.at[measures[col], '3rd Quartile'] = r2[col].quantile(0.75)
    out_table.at[measures[col], 'Max'] = r2[col].max()
    out_table.at[measures[col], 'skew'] = r2[col].skew()
    out_table.at[measures[col], 'kurtosis'] = r2[col].kurtosis()

#Calculatte descriptives for various measures in the round 3 assessement
measures = {'ERQ_Cog_Reapp': 'ERQ Cognitive Reappraisal',
            'ERQ_Exp_Supp': 'ERQ Expressive Suppression',
            'BSCS_Total': 'Brief Self-Control Scale',
            'IU_Total': 'Intolerance of Uncertainty',
            'SUPPS_Neg_Urg': 'UPPS-P negative urgency',
            'SUPPS_Lack_Pers': 'UPPS-P lack of premediation',
            'SUPPS_Lack_Premed': 'UPPS-P lack of perseverance',
            'SUPPS_Sen_Seek': 'UPPS-P sensation seeking',
            'SUPPS_Pos_Urg': 'UPPS-P positive urgency'}
for col in measures:
    out_table.at[measures[col], 'M (SD)'] = '%.2f (%.2f)' % (r3[col].mean(), r3[col].std(ddof=1))
    out_table.at[measures[col], 'Min'] = r3[col].min()
    out_table.at[measures[col], '1st Quartile'] = r3[col].quantile(0.25)
    out_table.at[measures[col], 'Median'] = r3[col].median()
    out_table.at[measures[col], '3rd Quartile'] = r3[col].quantile(0.75)
    out_table.at[measures[col], 'Max'] = r3[col].max()
    out_table.at[measures[col], 'skew'] = r3[col].skew()
    out_table.at[measures[col], 'kurtosis'] = r3[col].kurtosis()
    
#Export table to .csv file
out_table.to_csv(join(main_dir, 'analysis', 'ScientificData', 'ScientificData_descriptives.csv'))


#%% MAKE DEMOGRAPHICS TABLE

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
              'race1___5': 'Native Hawaiian or Other Pacific Islander',
              'race1___6': 'American Indian/ Alaska Native',
              'race1___7': 'More than one race/ Prefer to self-describe',
              'race1___8': 'Unknown',
              'race1___9': 'Prefer not to say (race)',
              'ethnicity___1': 'Hispanic',
              'ethnicity___2': 'Not Hispanic',
              'ethnicity___3': 'Prefer not to say (ethnicity)',
              'disability___1': 'Sensory impairment (vision/hearing)',
              'disability___2': 'Mobility impairment ',
              'disability___3': 'Learning disability',
              'disability___4': 'Mental Health Disorder',
              'disability___5': 'Disability or impairment not listed above',
              'disability___6': 'Prefer not to say (disability)'}
demo.rename(label_dict, axis='columns', inplace=True)

#Get demographic subsets
demo_daily = demo[demo['sub_id'].isin(data['sub_id'].unique())]
demo_r1 = demo[demo['sub_id'].isin(r1['sub_id'].unique())]
demo_r2 = demo[demo['sub_id'].isin(r2['sub_id'].unique())]
demo_r3 = demo[demo['sub_id'].isin(r3['sub_id'].unique())]
col_map = {'All':demo, 'Daily':demo_daily, 'R1':demo_r1, 'R2':demo_r2, 'R3':demo_r3}

demo_table = pd.DataFrame()

for subset in col_map:
    
    demo_subset = col_map[subset]
    
    #Number of unique participants
    demo_table.loc['N', subset] = len(demo_subset)

    #Age
    demo_table.loc['Age', subset] = np.nan
    for col in demo['age1'].describe().index[1:]:
        demo_table.loc[col, subset] = demo_subset['age1'].describe()[col]

    #Country of residence
    demo_table.loc['Residence', subset] = np.nan
    demo_table.loc['USA', subset] = np.mean(demo_subset['country'] == 'UNITED STATES')
    demo_table.loc['Canada', subset] = np.mean(demo_subset['country'] == 'CANADA')
    demo_table.loc['Australia', subset] = np.mean(demo_subset['country'] == 'AUSTRALIA')
    demo_table.loc['United Kindgdom', subset] = np.mean(demo_subset['country'] == 'UNITED KINGDOM')
    demo_table.loc['India', subset] = np.mean(demo_subset['country'] == 'INDIA')
    demo_table.loc['other', subset] = 1 - demo_table.loc['USA':'India', subset].sum()

    #Race and ethnicity
    demo_table.loc['Ethnicity', subset] = np.nan
    race_col = demo.loc[:, 'Hispanic':'Prefer not to say (ethnicity)'].columns
    for col in race_col:
        demo_table.loc[col, subset] = demo_subset[col].mean()
    demo_table.loc['Race', subset] = np.nan
    race_col = demo.loc[:, 'African American':'Prefer not to say (race)'].columns
    for col in race_col:
        demo_table.loc[col, subset] = demo_subset[col].mean()

    #All other variables
    for col in ['preferred_gender', 'bio_sex', 'transgender2', 'sexual_orientation',
                'education', 'marital', 'medical', 'income', 'student', 'employed']:
        demo_table.loc[col, subset] = np.nan
        for cat in demo[col].value_counts().index:
            demo_table.loc[('%s: %s' % (col, cat)), subset] = demo_subset[col].value_counts(normalize=True, sort=False)[cat]
        
#Rename rows to clearer labels
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
                     'education: 3.0': "some college",
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
                     'employed: 0.0': 'no'}
demo_table.rename(row_substitutions, inplace=True)

#Output table to .csv file
demo_table.to_csv(join(main_dir, 'analysis', 'ScientificData', 'ScientificData_demographics_table.csv'))
