This readme is a guide to using some helpful Python functions and scripts shared with the BC COVID-19 dataset.

For additional information on the dataset, see:
Cunningham, T. J., Fields, E. C., & Kensinger, E. A. (2021). Boston College daily sleep and well-being survey data during early phase of the COVID-19 pandemic. Scientific Data, 8(110). https://doi.org/10.1038/s41597-021-00886-y


############################### BC_COVID19_import_data.py ###############################

This script contains a one function: import_covid19_data

This function takes care of some data import tasks that need to be done for any work on the data in Python. It imports the data from each source (demographics, daily surveys, and one time assessments) and applies some useful formatting to various variables (e.g., converting dates and clock times from strings to Python types that can be worked with mathematically). It also does some data transformations that we have done in our own analysis, such as re-scaling the stress and worry variables so that higher numbers mean more worry and stress (in the raw data, it is the opposite).

You can copy or import this function to any script where you need to import the data into Python.

Usage of the function is demonstrated at the bottom of the script. The inputs for the function are:
data_dir - The full path to the directory where all the data .csv files are
date_str - The date string at the end of each .csv file (this changes with each new release of the data)




############################### BC_COVID19_merge.py ###############################

This script facilitates the potentially difficult task of merging data from the various sources in the dataset (demographics, daily surveys, and one time assessments). The user specifies the variables they want from each source, and the script outputs a csv file with the merged data. For variables in the daily data, each day from each subject will be output as a different row (with one-time assessment and demographic variables replicated across each row). The user can also specify time bins, in which case variables from the daily data will be averaged across each time bin and there will be one row per time bin per subject.

*** USAGE ***

1. Scroll down to the section labelled SET-UP. This section is clearly demarcated, and you should not need to change anything outside this section.

2. The first two variables tell Python how to find the COVID-19 data
a. You can find the date string at the end of each csv file name. It should be the same for all the data files.
b. The data directory is just the full path on your computer (or network you are connected to) containing the data csv files

3. The output file is the full path and name of a csv file to output the merged data to. If you don't want to create a file (see below), you can specify False for the output_file input.

4. The time_bins variable is a Python dictionary specifying names and time ranges for time bins to average across. If this is an empty dictionary (or None of False), un-averaged daily data will be return in the csv file.

5. For each data source, indicate which variables you want using the column names from the relevant csv file in single quotes. An empty list (square brackets) means no data from that source will be included.

6. Run the script. This should produce the csv file you've requested.

Experienced Python programmers can copy or import the two functions in this script into their own script. The merge function will output the merged data to a Python data frame for those who wish to further work with the data in Python.



############################### BC_COVID19_make_demo_table.py ###############################

This script will create a demographics table for a given subset of participants, or for multiple subsets of participants.

The script contains two functions: make_demo_table and make_grouped_demo_table. You can use this script or copy or import these functions into your own script.

make_demo_table takes the demographics data and the r4 data as pandas data frame as inputs. These can be easily produced using the import_covid19_data function (see above). An optional third input specifies a subset of participants to include in the table as a list of subject IDs (the default is all participants). The function returns a demographics table as a pandas data frame.

make_grouped_demo_table is very similar, but allows you to make a demographics table where each column is a different subset of participants. Here, the third input to the function is a Python dictionary, where the keys are the column headings and the values are lists of the subsets.

The bottom of the script gives an example of usage.

The returned demographics table can be output to a .csv or .xlsx file using pandas to_csv and to_excel methods.

Here are some examples of how to use pandas to get the list of subject IDs you need for different subsets:

#US participants only
subs = demo.loc[demo['country'] == 'UNITED STATES', 'sub_id]

#Participants 60 and older
subs = demo.loc[demo['age1'] >=60, 'sub_id]

#Participants who completed round 1
subs = r1['sub_id']


############################### BC_COVID19_make_demo_relabel.py ###############################

Many variables in the demographics file have numerical responses that correspond to categories (e.g., for bio_sex, 1=female, 2=male). In addition, some column names also contain numbers representing categories (e.g., the race___1 column contains a 0 or 1 indicating whether the participant is African-American). The relabel_demo function in this script relabels all these numbers to strings for each category.
