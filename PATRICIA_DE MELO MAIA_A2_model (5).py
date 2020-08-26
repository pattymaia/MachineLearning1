
# timeit - used to measure the execution time
import timeit
elapsed_time =  """

#!/usr/bin/env python
# coding: utf-8

# Created on Fri March 09 14:42:35 2020
# 
# @author:Patricia de Melo Maia cohort: 5 - Valencia
# 

################################################################################
# Import Packages
################################################################################

# Importing libraries
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix        
from sklearn.metrics import roc_auc_score           
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.tree import export_graphviz            
from IPython.display import Image 
import pydotplus 


###############################################################################
#Load the Data 
###############################################################################

# loading data
apprendice = pd.read_excel('Apprentice_Chef_Dataset.xlsx')


################################################################################
# Feature Engineering 
################################################################################

###########  FAMILY_NAME  ##############
#Step 1: Flagging missing values

for col in apprendice:

# creating columns with 1s if missing and 0 if not
    if apprendice[col].isnull().astype(int).sum() > 0:    
        apprendice['m_'+col] = apprendice[col].isnull().astype(int)


#STEP 2:splitting names and checking name that shows relationship.

# placeholder list
placeholder_lst = []

# looping over each NAME
for index, col in apprendice.iterrows():
    
    # splitting NAME domain at ' '
    split_name = apprendice.loc[index, 'NAME'].split(sep = ' ')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_name)
    
# converting placeholder_lst into a DataFrame 
name_df = pd.DataFrame(placeholder_lst)


#STEP3: Creating a column to flag relatives 

# creating column: relative
name_df['relative'] = 0

# looping to NAME and checking for words that shows relationship between users.

for index, val in name_df.iterrows():
    if name_df.loc[ index, 2] =='(son' or name_df.loc[ index, 2] =='(daughter'or name_df.loc[ index, 1] =='(brother'or name_df.loc[ index, 1] =='(father)'or name_df.loc[ index, 2] =='(wife' or name_df.loc[ index, 1] =='Wife':
        name_df.loc[index, 'relative'] = 1
        
    elif name_df.loc[ index, 2]  !='(son' or name_df.loc[ index, 2] !='(daughter'or name_df.loc[ index, 1] !='(brother'or name_df.loc[ index, 1] !='(father)'or name_df.loc[ index, 2] !='(wife' or name_df.loc[ index, 1] !='Wife':
        name_df.loc[index, 'relative'] = 0
        
    else:
        Print(error)
        

#Concatenating name_df to apprendice
apprendice= pd.concat([apprendice, name_df], axis=1)

#Dropping colums 
apprendice = apprendice.drop(labels = [0,1,2,3,4,5 ], axis   = 1)


#STEP4: drop FAMILY_NAME
apprendice = apprendice.drop(labels = ['FAMILY_NAME'], axis   = 1)



###########  EMAIL  ##############

#Step 1: Splitting

# Creating an empty list
placeholder_lst = []

# looping over each email address

for index, col in apprendice.iterrows():
    
    # splitting email domain at '@'
    split_email = apprendice.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)

# Converting placeholder_lst into a DataFrame 
email = pd.DataFrame(placeholder_lst)


# renaming column to concatenate
email.columns = ['0' , 'PERSONAL_MAIL']

# concatenating 'PERSONAL_MAIL' with apprendice DataFrame
apprendice = pd.concat([apprendice, email['PERSONAL_MAIL']], axis = 1)


#Step2: Indicating if the email is professional, junk or personal

# Testing if all other emails than junk and personal are professional
personal_email = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_email = ['@me.com', '@aol.com', '@hotmail.com', '@live.com', '@msn.com', '@passport.com']
professional_email = ['@mmm.com', '@amex.com', '@apple.com', '@boeing.com', 
                              '@caterpillar.com', '@chevron.com', '@cisco.com', '@cocacola.com',
                              '@disney.com', '@dupont.com', '@exxon.com', '@ge.org', '@goldmansacs.com', 
                              '@homedepot.com', '@ibm.com', '@intel.com', '@jnj.com', '@jpmorgan.com', 
                              '@mcdonalds.com', '@merck.com', '@microsoft.com', '@nike.com', 
                              '@pfizer.com', '@pg.com', '@travelers.com', '@unitedtech.com', 
                              '@unitedhealth.com', '@verizon.com', '@visa.com', '@walmart.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in apprendice['PERSONAL_MAIL']:
    
    if '@' + domain in personal_email:
        placeholder_lst.append('personal')
        
    elif '@' + domain in junk_email:
        placeholder_lst.append('junk')

    elif '@' + domain in professional_email:
        placeholder_lst.append('professional')

    else:
            print('Unknown')

# concatenating with original DataFrame
apprendice['domain_group'] = pd.Series(placeholder_lst)


#STEP 3: Creating dummies for domain_group
# one hot encoding categorical variables
one_hot_domain= pd.get_dummies(apprendice['domain_group'])

# dropping categorical variables after they've been encoded
apprendice = apprendice.drop('domain_group', axis = 1)

# joining codings together
apprendice = apprendice.join([one_hot_domain])


# Step 4: droping columns EMAIL, PERSONAL_MAIL 
apprendice = apprendice.drop(['EMAIL','PERSONAL_MAIL'], axis = 1)



###########  FOLLOWED_RECOMMENDATION_PCT  ##############

#STEP 1: Creating variable NUMBER OF RECOMMENDATIONS

##### NUMBER OF RECOMMENDATION
apprendice["NUMBER_OF_RECOMMENDATION"]= apprendice['FOLLOWED_RECOMMENDATIONS_PCT']/100 * apprendice['TOTAL_MEALS_ORDERED']

#STEP 2: Drop FOLLOWED_RECOMMENDATIONS_PCT
apprendice = apprendice.drop(labels = ['FOLLOWED_RECOMMENDATIONS_PCT'], axis   = 1)

#Dropping Irrelevant Variables: post-event

apprendice = apprendice.drop(labels = ['CANCELLATIONS_BEFORE_NOON', 
                                       'CANCELLATIONS_AFTER_NOON',
                                       'PC_LOGINS',
                                       'MOBILE_LOGINS', 
                                       'EARLY_DELIVERIES', 
                                       'LATE_DELIVERIES', 
                                       'PACKAGE_LOCKER', 
                                       'REFRIGERATED_LOCKER'], 
                                         axis   = 1)


#Creating new variable ###########################################################

##### Dividing the avg time per site visit by total orders.
apprendice["AVG_TIME_PER_SITE_VISIT_PER_ORDER"]= apprendice['AVG_TIME_PER_SITE_VISIT']/apprendice['TOTAL_MEALS_ORDERED']

##### Dividing the avg clicks per visit by total orders.
apprendice['AVG_CLICKS_PER_VISIT_PER_ORDER'] =apprendice['AVG_CLICKS_PER_VISIT']/apprendice['TOTAL_MEALS_ORDERED']

##### Dividing PRODUCT CATEGORIES VIEWED by total orders.
apprendice['PRODUCT_CATEGORIES_VIEWED_PER_ORDER'] = apprendice['PRODUCT_CATEGORIES_VIEWED']/apprendice['TOTAL_MEALS_ORDERED']


# Copying the file
apprend = apprendice.copy()

#################################################################################
#Set up train-test split
#################################################################################

# declaring explanatory variables
apprend_data = apprend.drop(['CROSS_SELL_SUCCESS'], axis = 1)

#Dropping categorical variables
apprend_data = apprend_data.drop(columns= ['NAME','FIRST_NAME'])

# declaring response variable
apprend_target = apprend.loc[ : ,'CROSS_SELL_SUCCESS']

# train-test split
X_train, x_test, y_train, y_test = train_test_split(
            apprend_data,
            apprend_target,
            test_size = 0.25,
            random_state = 222,
            stratify = apprend_target)

# merging training data for statsmodels
apprend_train = pd.concat([X_train, y_train], axis = 1)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a classification tree object
tree_pruned      = DecisionTreeClassifier(max_depth = 4,
                                          min_samples_leaf = 25,
                                          random_state = 802)


# FITTING the training data
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# PREDICTING on new data
tree_pred = tree_pruned_fit.predict(x_test)

################################################################################
# Final Model Score (score)
################################################################################

training_score = tree_pruned_fit.score(X_train, y_train).round(4)
test_score = tree_pruned_fit.score(x_test, y_test).round(4)
AUC_score = roc_auc_score(y_true  = y_test, y_score = tree_pred).round(4)

# SCORING the model

print('Training ACCURACY:', training_score)
print('Testing  ACCURACY:', test_score)
print('AUC Score        :', AUC_score)


"""
# calculating execution time
elapsed_time = timeit.timeit(elapsed_time, number=3)/3
print(elapsed_time)