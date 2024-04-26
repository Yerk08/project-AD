# -*- coding: utf-8 -*-
"""
Import all data with some useful formatting

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
from os.path import join
import pandas as pd


def import_covid19_data(data_dir, date_str):
    """
    Import all COVID19 data with formatting
    """
    
    ##### Import Data #####
    
    data = pd.read_csv(join(data_dir, 'COVID19_combined_cleaned_%s.csv' % date_str), 
                       index_col='unique_id')
    demo = pd.read_csv(join(data_dir, 'COVID19_demographics_cleaned_%s.csv' % date_str), 
                       index_col='record_id')
    r1 = pd.read_csv(join(data_dir, 'COVID19_Round1_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    r2 = pd.read_csv(join(data_dir, 'COVID19_Round2_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    r3 = pd.read_csv(join(data_dir, 'COVID19_Round3_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    r4 = pd.read_csv(join(data_dir, 'COVID19_Round4_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    r5 = pd.read_csv(join(data_dir, 'COVID19_Round5_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    r6 = pd.read_csv(join(data_dir, 'COVID19_Round6_cleaned_%s.csv' % date_str), 
                     index_col='record_id')
    
    ##### Format Daily and Demographic Data #####
    
    #Convert date format in daily data
    daily_date_vars = ['ref_date', 'covid19_timestamp', 'todays_date', 'redcap_timestamp',
                       'covid19_short_survey_timestamp']
    for col in daily_date_vars:
        data[col] = pd.to_datetime(data[col])
    
    #Reverse score and shift to 0
    worry_vars = ['worry_health', 'family_health', 'community_1health', 
                  'national_health', 'worry_finances']
    for col in ['stress', 'isolation', *worry_vars]:
        data[col] = 8 - data[col] - 1
    data['worry_scale'] = 40 - data['worry_scale'] - 5
    
    #Convert sleepdiary times to timedelta
    daily_time_vars = ['sleepdiary_bedtime', 'sleepdiary_fallasleep', 
                       'sleepdiary_waketime', 'sleepdiary_outofbed']
    for col in daily_time_vars:
        data[col] = pd.to_datetime(data[col]) - pd.Timestamp.today().normalize()
    
    #Convert date variables
    demo['date_time'] = pd.to_datetime(demo['date_time'])
    
    ##### Format Round 1-4 Data #####
    
    #Convert timestamps to dates
    r1['round_1_timestamp'] = pd.to_datetime(r1['round_1_timestamp'])
    r1['date_time_rd1'] = pd.to_datetime(r1['date_time_rd1'])
    r2['round_2_timestamp'] = pd.to_datetime(r2['round_2_timestamp'])
    r2['date_time_rd2'] = pd.to_datetime(r2['date_time_rd2'])
    r3['round_3_timestamp'] = pd.to_datetime(r3['round_3_timestamp'])
    r3['date_time_rd3'] = pd.to_datetime(r3['date_time_rd3'])
    r4['round_4_timestamp'] = pd.to_datetime(r4['round_4_timestamp'])
    r4['date_time_rd4'] = pd.to_datetime(r4['date_time_rd4'])
    r5['round_5_timestamp'] = pd.to_datetime(r5['round_5_timestamp'])
    r5['date_time_rd5'] = pd.to_datetime(r5['date_time_rd5'])
    r6['april_18_timestamp'] = pd.to_datetime(r6['april_18_timestamp'])
    r6['todays_date'] = pd.to_datetime(r6['todays_date'])
    
    #Other date variables
    r2_date_vars = ['stayhome_begin_us', 'stayhome_end_us', 'stayhome_begin', 
                    'stayhome_end', 'normal_date', 'mask_date', 'meetings_date', 
                    'bigevents_date', 'shaking_hands_date']
    for col in r2_date_vars:
        r2[col] = pd.to_datetime(r2[col])
    r4_date_vars = ['stayhome_begin_us_fut', 'stayhome_end_us_fut', 
                    'stayhome_begin_fut', 'stayhome_end_fut']
    for col in r4_date_vars:
        r4[col] = pd.to_datetime(r4[col])
    r5_date_vars = (['round_5_timestamp', 'date_time_rd5', 'date_cov', 'vacc_date'] 
            + [col for col in r5.columns if '_date_' in col])
    for col in r5_date_vars:
        r5[col] = pd.to_datetime(r5[col])
    r6_date_vars = ['april_18_timestamp', 'todays_date', 'vacc_date', 'date_cov']
    for col in r6_date_vars:
        r6[col] = pd.to_datetime(r6[col])
        
    #Convert clock times to timedelta
    r1_clock_vars = ['psqi_1', 'psqi_3', 'mtq_p3', 'mtq_p4', 'mtq_p5', 'mtq_p6',
                     'mtq_3', 'mtq_4', 'mtq_5', 'mtq_6']
    r1_clock_vars += [col for col in r1.columns if col.endswith('_sleepmidpoint') 
                      or col.endswith('_chronotype')]
    for col in r1_clock_vars:
        r1[col] = pd.to_datetime(r1[col]) - pd.Timestamp.today().normalize()
    r4_clock_vars = ['fall_psqi_1', 'fall_psqi_3', 'fall_mtq_3', 'fall_mtq_4', 
                     'fall_mtq_5', 'fall_mtq_6']
    r4_clock_vars += [col for col in r4.columns if col.endswith('_sleepmidpoint') 
                      or col.endswith('_chronotype')]
    for col in r4_clock_vars:
        r4[col] = pd.to_datetime(r4[col]) - pd.Timestamp.today().normalize()
    r5_clock_vars = ['psqi_1', 'psqi_3', 'mtq_3', 'mtq_p8', 'mtq_p9', 'mtq_p10']
    r5_clock_vars += [col for col in r5.columns if col.endswith('_sleepmidpoint') 
                      or col.endswith('_chronotype')]
    for col in r5_clock_vars:
        assert r5.loc[r5[col].notna(), col].apply(lambda x: bool(re.search('\d:\d\d', x))).all()
        r5[col] = pd.to_datetime(r5[col]) - pd.Timestamp.today().normalize()
        assert all(r5[col].between(pd.Timedelta(0), pd.Timedelta(24, unit='hours'))
                   | r5[col].isna())
    
    return (data, demo, r1, r2, r3, r4, r5, r6)


if __name__ == '__main__':
    
    #Full path of of directory containing data csv files
    data_dir = r'D:\COVID19\export'
    
    #Date string at the end of the data csv files
    date_str = '2021-07-22_13_26'
    
    #Import data
    (data, demo, r1, r2, r3, r4, r5, r6) = import_covid19_data(data_dir, date_str)
