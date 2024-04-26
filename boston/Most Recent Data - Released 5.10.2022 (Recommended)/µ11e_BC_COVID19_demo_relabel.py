# -*- coding: utf-8 -*-
"""
Change numbers in demographics to labels

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

from BC_COVID19_import_data import import_covid19_data


def relabel_demo(demo):
    """
    Change labelling in demographic data from numbers to informative labels
    """
    
    demo = demo.copy()
    
    demo.rename({'age1': 'age'}, axis='columns', inplace=True)
    
    demo['bio_sex'].replace({1: 'female', 2: 'male'},
                            inplace=True)
    
    demo['preferred_gender'].replace({1: 'female', 2: 'male', 3: 'non-binary/third gender', 
                                      4: 'prefer to self-describe', 5: 'prefer not to say'},
                                     inplace=True)
    
    demo['transgender2'].replace({1: 'yes', 2: 'no', 3: 'prefer not to say'},
                                 inplace=True)
    demo.rename({'transgender2': 'transgender'}, axis='columns', inplace=True)
    
    demo['sexual_orientation'].replace({1: 'gay/lesbian', 2: 'bisexual', 3: 'straight/heterosexual', 
                                        4: 'prefer to self-describe', 5: 'prefer not to say'},
                                       inplace=True)
    
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
    
    demo['military'].replace({1: 'civilian', 2: 'active military', 3: 'veteran'},
                             inplace=True)
    
    demo['marital'].replace({1: 'single', 2: 'in a relationship', 3: 'married', 
                             4: 'separated/divorced', 5: 'widowed'},
                            inplace=True)
    
    demo['medical'].replace({1: 'yes', 0: 'no'}, inplace=True)
    
    demo['income'].replace({1: '$0-$25,000', 2: '$25,001-$50,000', 3: '$50,001 - $75,000', 
                            4: '$75,001 - $100,000', 5: '$100,001 - $150,000', 
                            6: '$150,001 - $250,000', 7: '$250,000+'},
                           inplace=True)
    
    demo['education'].replace({1: 'some high school', 2: 'high school diploma or GED', 
                               3: 'some college', 4: 'college degree', 5: 'some post-bacc education', 
                               6: 'graduate, medical, or professional degree'},
                              inplace=True)
    
    demo['student'].replace({1: 'yes', 0: 'no'}, inplace=True)
    
    demo['employed'].replace({1: 'yes', 0: 'no'}, inplace=True)
    
    demo['working_home'].replace({1: 'no', 2: 'part-time', 3: 'full-time'}, 
                                 inplace=True)
    
    demo['employment_covid'].replace({1: 'yes', 0: 'no'}, inplace=True)
    
    demo['institution_measures'].replace({1: 'yes', 0: 'no'}, inplace=True)
    
    demo['normal_units'].replace({1: 'days', 2: 'weeks', 3: 'months'},
                                 inplace=True)
    
    return demo


if __name__ == '__main__':
    
    #Full path of of directory containing data csv files
    data_dir = r'D:\COVID19\export'
    
    #Date string at the end of the data csv files
    data_str = '2021-07-22_13_26'
    
    #Import data
    (_, demo, _, _, _, _, _, _) = import_covid19_data(data_dir, data_str)
    
    #Relabel
    demo = relabel_demo(demo)
