# Created on Fri March 02 14:42:35 2020
# 
# @author:Patricia de Melo Maia
# cohort: 5 - Valencia
#     
#     A) Introduction:
#     Apprentice Chef, Inc. is an innovative company with a unique spin on cooking at home. 
#     After three years serving customers across the San Francisco Bay Area, the executives 
#     at Apprentice Chef have come to realize that over 90% of their revenue comes from customers
#     that have been ordering meal sets for 12 months or less. Given this information, they would
#     like to better understand how much revenue to expect from each customer within their first 
#     year of orders. Thus, they have hired you on a full-timecontract to analyze their data, 
#     develop your top insights, and build a machine learning model to predict revenue over the 
#     first year of each customer’s life cycle.

# In[1]:


##########################################################################################################
#Inicialization #########################################################################################
##########################################################################################################
# importing libraries
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
import sklearn.linear_model # linear models
import gender_guesser.detector as gender # gender detection
import sklearn.neighbors # KNN for Regression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
          
# setting pandas print options
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'

# reading the file 
df = pd.read_excel(file)


# In[ ]:


##########################################################################################################
#Dataset Exploration ####################################################################################
##########################################################################################################
#First lines
#df.head(n = 5)

# Information about each variable
#df.info()

# descriptive statistics
#df.describe().round(2)

#Checking Missing Value Analysis and Imputation
#df.isnull().sum()


# First I imported the libraries, after, uploaded the file and then, and, then, check the information in the dataset. Checking the first 5 lines of dataset we see there are more information in the column name, like the relation between some users. Checking the info of the dataset we can see that most of the information are interger(22), there are three float and four objects. Checking the descrictive statistic information, we can see that all the user mande contact with customer service. Also, most people is registered with a mobile phone.

# In[2]:



# I runned a linear regression with all the intergir (22) and float(3) to check how are the p-values between them.
# I found p-values to some of theses variables. Instead of drop them, I will flag them as binary (1-use the feature , 0-did not use it) to check if the information existence is relevante to my model. Unfortunately, this approch did not improve my model. Now I will create new variables. 

# In[3]:


##########################################################################################################
#Creating new variable #######################################################################################
##########################################################################################################

##### Dividing the avg time per site visit by total orders.
df["AVG_TIME_PER_SITE_VISIT_PER_ORDER"]= df['AVG_TIME_PER_SITE_VISIT']/df['TOTAL_MEALS_ORDERED']

##### Dividing the avg prep vid time by total orders.
df['AVG_PREP_VID_TIME_PER_ORDER'] = df['AVG_PREP_VID_TIME']/df['TOTAL_MEALS_ORDERED']

##### Dividing the avg clicks per visit by total orders.
df['AVG_CLICKS_PER_VISIT_PER_ORDER'] = df['AVG_CLICKS_PER_VISIT']/df['TOTAL_MEALS_ORDERED']

##### Dividing PRODUCT CATEGORIES VIEWED by total orders.
df['PRODUCT_CATEGORIES_VIEWED_PER_ORDER'] = df['PRODUCT_CATEGORIES_VIEWED']/df['TOTAL_MEALS_ORDERED']

##### NUMBER OF RECOMMENDATION
df["NUMBER_OF_RECOMMENDATION"]= df['FOLLOWED_RECOMMENDATIONS_PCT']/100 * df['TOTAL_MEALS_ORDERED']



# In[5]:


##########################################################################################################
#Feature Engineering ####################################################################################
##########################################################################################################

#thresholds - Base on Histogram analysis
#REVENUE
#CROSS_SELL_SUCCESS
TOTAL_MEALS_ORDERED_HI = 200
UNIQUE_MEALS_PURCH_HI = 9
CONTACTS_W_CUSTOMER_SERVICE_HI= 12
CONTACTS_W_CUSTOMER_SERVICE_LO= 3
PRODUCT_CATEGORIES_VIEWED_HI = 10
PRODUCT_CATEGORIES_VIEWED_LO = 2
AVG_TIME_PER_SITE_VISIT_HI= 240
#MOBILE_NUMBER
CANCELLATIONS_BEFORE_NOON_HI = 5
CANCELLATIONS_AFTER_NOON_HI = 2
#TASTES_AND_PREFERENCES 
PC_LOGINS_HI= 7
PC_LOGINS_LO= 4
MOBILE_LOGINS_HI= 2
WEEKLY_PLAN_HI= 15
EARLY_DELIVERIES_HI = 4
LATE_DELIVERIES_HI= 7
#PACKAGE_LOCKER => is already binary
#REFRIGERATED_LOCKER  => is already binary
#FOLLOWED_RECOMMENDATIONS_PCT
AVG_PREP_VID_TIME_HI= 280
LARGEST_ORDER_SIZE_HI = 7.5
MASTER_CLASSES_ATTENDED_HI = 2
#MEDIAN_MEAL_RATING 
AVG_CLICKS_PER_VISIT_HI=18
AVG_CLICKS_PER_VISIT_LO=8
TOTAL_PHOTOS_VIEWED_HI= 200
AVG_TIME_PER_SITE_VISIT_PER_ORDER_HI= 8
AVG_PREP_VID_TIME_PER_ORDER_HI = 7.5
AVG_CLICKS_PER_VISIT_PER_ORDER_HI = 0.7
PRODUCT_CATEGORIES_VIEWED_PER_ORDER_HI= 0.25
NUMBER_OF_RECOMMENDATION_HI= 80


#TOTAL_MEALS_ORDERED ############################################################################################

df['T_TOTAL_MEALS_ORDERED'] = 0
condition_hi = df.loc[0:,'T_TOTAL_MEALS_ORDERED'][df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

df['T_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,value = 1,inplace = True)


#UNIQUE_MEALS_PURCH ##############################################################################################

df['T_UNIQUE_MEALS_PURCH'] = 0
condition_hi = df.loc[0:,'T_UNIQUE_MEALS_PURCH'][df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HI]

df['T_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,value = 1,inplace = True)

#CONTACTS_W_CUSTOMER_SERVICE #####################################################################################

df['T_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = df.loc[0:,'T_CONTACTS_W_CUSTOMER_SERVICE'][df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]
condition_lo = df.loc[0:,'T_CONTACTS_W_CUSTOMER_SERVICE'][df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_LO]


df['T_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,value = 1,inplace = True)
df['T_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,value = 1,inplace = True)

#PRODUCT_CATEGORIES_VIEWED - HIGHER AND LOWER LIMITS ##############################################################

df['T_PRODUCT_CATEGORIES_VIEWED'] = 0
condition_hi = df.loc[0:,'T_PRODUCT_CATEGORIES_VIEWED'][df['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_HI]
condition_lo = df.loc[0:,'T_PRODUCT_CATEGORIES_VIEWED'][df['PRODUCT_CATEGORIES_VIEWED'] < PRODUCT_CATEGORIES_VIEWED_LO]

df['T_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_hi,value = 1,inplace = True)
df['T_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_lo,value = 1,inplace = True)

# AVG_TIME_PER_SITE_VISIT #########################################################################################

df['T_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = df.loc[0:,'T_AVG_TIME_PER_SITE_VISIT'][df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HI]

df['T_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,value = 1,inplace = True)


#CANCELLATIONS_BEFORE_NOON #########################################################################################

df['T_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = df.loc[0:,'T_CANCELLATIONS_BEFORE_NOON'][df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_HI]

df['T_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,value = 1,inplace = True)


#CANCELLATIONS_AFTER_NOON ###########################################################################################

df['T_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = df.loc[0:,'T_CANCELLATIONS_AFTER_NOON'][df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

df['T_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,value = 1,inplace = True)

#PC_LOGINS - HIGHER AND LOWER LIMITS #############################################################################

df['T_PC_LOGINS'] = 0
condition_hi = df.loc[0:,'T_PC_LOGINS'][df['PC_LOGINS'] > PC_LOGINS_HI]
condition_lo = df.loc[0:,'T_PC_LOGINS'][df['PC_LOGINS'] < PC_LOGINS_LO]

df['T_PC_LOGINS'].replace(to_replace = condition_hi,value = 1,inplace = True)
df['T_PC_LOGINS'].replace(to_replace = condition_lo,value = 1,inplace = True)

#MOBILE_LOGINS #####################################################################################################
df['T_MOBILE_LOGINS'] = 0
condition = df.loc[0:,'T_MOBILE_LOGINS'][df['MOBILE_LOGINS'] > MOBILE_LOGINS_HI]

df['T_MOBILE_LOGINS'].replace(to_replace = condition_hi,value = 1,inplace = True)


# WEEKLY_PLAN ######################################################################################################
df['T_WEEKLY_PLAN'] = 0
condition = df.loc[0:,'T_WEEKLY_PLAN'][df['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]

df['T_WEEKLY_PLAN'].replace(to_replace = condition_hi,value = 1,inplace = True)


#EARLY_DELIVERIES ##################################################################################################
df['T_EARLY_DELIVERIES'] = 0
condition = df.loc[0:,'T_EARLY_DELIVERIES'][df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]

df['T_EARLY_DELIVERIES'].replace(to_replace = condition_hi,value = 1,inplace = True)


#LATE_DELIVERIES ##################################################################################################
df['T_LATE_DELIVERIES'] = 0
condition_hi = df.loc[0:,'T_LATE_DELIVERIES'][df['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

df['T_LATE_DELIVERIES'].replace(to_replace = condition_hi,value = 1,inplace = True)


#AVG_PREP_VID_TIME ###########################################################################################
df['T_AVG_PREP_VID_TIME'] = 0
condition_hi = df.loc[0:,'T_AVG_PREP_VID_TIME'][df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]

df['T_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,value = 1,inplace = True)

#LARGEST_ORDER_SIZE ###########################################################################################
df['T_LARGEST_ORDER_SIZE'] = 0
condition_hi = df.loc[0:,'T_LARGEST_ORDER_SIZE'][df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]

df['T_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,value = 1,inplace = True)


#MASTER_CLASSES_ATTENDED ###########################################################################################
df['T_MASTER_CLASSES_ATTENDED'] = 0
condition_HI = df.loc[0:,'T_MASTER_CLASSES_ATTENDED'][df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

df['T_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,value = 1,inplace = True)


# AVG_CLICKS_PER_VISIT - HIGHER AND LOWER LIMITS ###################################################################
df['T_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = df.loc[0:,'T_AVG_CLICKS_PER_VISIT'][df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]
condition_lo = df.loc[0:,'T_AVG_CLICKS_PER_VISIT'][df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_LO]

df['T_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,value = 1,inplace = True)
df['T_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,value = 1,inplace = True)


# TOTAL_PHOTOS_VIEWED ##############################################################################################

df['T_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = df.loc[0:,'T_TOTAL_PHOTOS_VIEWED'][df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_HI]

df['T_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)

#AVG_TIME_PER_SITE_VISIT_PER_ORDER #################################################################################


df['T_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = df.loc[0:,'T_TOTAL_PHOTOS_VIEWED'][df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_HI]

df['T_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)

#AVG_PREP_VID_TIME_PER_ORDER #######################################################################################

df['T_AVG_PREP_VID_TIME_PER_ORDER'] = 0
condition_hi = df.loc[0:,'T_AVG_PREP_VID_TIME_PER_ORDER'][df['AVG_PREP_VID_TIME_PER_ORDER'] > AVG_PREP_VID_TIME_PER_ORDER_HI]

df['T_AVG_PREP_VID_TIME_PER_ORDER'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)
#AVG_CLICKS_PER_VISIT_PER_ORDER ####################################################################################

df['T_AVG_CLICKS_PER_VISIT_PER_ORDER'] = 0
condition_hi = df.loc[0:,'T_AVG_CLICKS_PER_VISIT_PER_ORDER'][df['AVG_CLICKS_PER_VISIT_PER_ORDER'] > AVG_CLICKS_PER_VISIT_PER_ORDER_HI]

df['T_AVG_CLICKS_PER_VISIT_PER_ORDER'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)
#PRODUCT_CATEGORIES_VIEWED_PER_ORDER ###############################################################################

df['T_PRODUCT_CATEGORIES_VIEWED_PER_ORDER'] = 0
condition_hi = df.loc[0:,'T_PRODUCT_CATEGORIES_VIEWED_PER_ORDER'][df['PRODUCT_CATEGORIES_VIEWED_PER_ORDER'] > PRODUCT_CATEGORIES_VIEWED_PER_ORDER_HI]

df['T_PRODUCT_CATEGORIES_VIEWED_PER_ORDER'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)
#NUMBER_OF_RECOMMENDATION ##########################################################################################

df['T_NUMBER_OF_RECOMMENDATION'] = 0
condition_hi = df.loc[0:,'T_NUMBER_OF_RECOMMENDATION'][df['NUMBER_OF_RECOMMENDATION'] > NUMBER_OF_RECOMMENDATION_HI]

df['T_NUMBER_OF_RECOMMENDATION'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)


# In[8]:


##########################################################################################################
#Feature Engineering ####################################################################################
##########################################################################################################
#thresholds - Base on scatterplot

#REVENUE
#CROSS_SELL_SUCCESS
TOTAL_MEALS_ORDERED_CHANGE = 120
UNIQUE_MEALS_PURCH_CHANGE = 9 
CONTACTS_W_CUSTOMER_SERVICE_CHANGE= 10
#PRODUCT_CATEGORIES_VIEWED
AVG_TIME_PER_SITE_VISIT_CHANGE= 190
#MOBILE_NUMBER
CANCELLATIONS_BEFORE_NOON_CHANGE = 8
CANCELLATIONS_AFTER_NOON_CHANGE = 2
#TASTES_AND_PREFERENCES 
#PC_LOGINS
#MOBILE_LOGINS
WEEKLY_PLAN_CHANGE= 15
#EARLY_DELIVERIES_HI = 4
LATE_DELIVERIES_CHANGE= 9
#PACKAGE_LOCKER => is already binary
#REFRIGERATED_LOCKER  => is already binary
#FOLLOWED_RECOMMENDATIONS_PCT
AVG_PREP_VID_TIME_CHANGE= 210
LARGEST_ORDER_SIZE_HI = 6
MASTER_CLASSES_ATTENDED_HI = 2
#MEDIAN_MEAL_RATING 
AVG_CLICKS_PER_VISIT_CHANGE= 10
TOTAL_PHOTOS_VIEWED_CHANGE= 300
AVG_TIME_PER_SITE_VISIT_PER_ORDER_CHARGE= 7.5
AVG_PREP_VID_TIME_PER_ORDER_CHARGE = 4
AVG_CLICKS_PER_VISIT_PER_ORDER_CHARGE= 0.6
PRODUCT_CATEGORIES_VIEWED_PER_ORDER_CHARGE= 0.3
NUMBER_OF_RECOMMENDATION_CHARGE= 60


#TOTAL_MEALS_ORDERED ###########################################################################################
df['TOTAL_MEALS_ORDERED_C'] = 0
condition = df.loc[0:,'TOTAL_MEALS_ORDERED_C']                  [df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_CHANGE]

df['TOTAL_MEALS_ORDERED_C'].replace(to_replace = condition,value = 1, inplace= True)

#UNIQUE_MEALS_PURCH ###########################################################################################
df['UNIQUE_MEALS_PURCH_C'] = 0
condition = df.loc[0:,'UNIQUE_MEALS_PURCH_C']                  [df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_CHANGE]

df['UNIQUE_MEALS_PURCH_C'].replace(to_replace = condition,value = 1, inplace= True)


#AVG_TIME_PER_SITE_VISIT ###########################################################################################
df['AVG_TIME_PER_SITE_VISIT_C'] = 0
condition = df.loc[0:,'AVG_TIME_PER_SITE_VISIT_C']                  [df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_CHANGE]

df['AVG_TIME_PER_SITE_VISIT_C'].replace(to_replace = condition,value = 1, inplace= True)

##########################################################################################################
#WEEKLY_PLAN
#df['WEEKLY_PLAN_C'] = 0
#condition = df.loc[0:,'WEEKLY_PLAN_C']\
#                  [df['WEEKLY_PLAN'] > WEEKLY_PLAN_CHANGE]

#df['WEEKLY_PLAN_C'].replace(to_replace = condition,value = 1, inplace= True)
##########################################################################################################
#AVG_CLICKS_PER_VISIT
df['AVG_CLICKS_PER_VISIT_C'] = 0
condition = df.loc[0:,'AVG_CLICKS_PER_VISIT_C']                  [df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_CHANGE]

df['AVG_CLICKS_PER_VISIT_C'].replace(to_replace = condition,value = 1, inplace= True)

##########################################################################################################
#AVG_PREP_VID_TIME
df['AVG_PREP_VID_TIME_C'] = 0
condition = df.loc[0:,'AVG_PREP_VID_TIME_C']                  [df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_CHANGE]

df['AVG_PREP_VID_TIME_C'].replace(to_replace = condition,value = 1, inplace= True)

##########################################################################################################
#TOTAL_PHOTOS_VIEWED
df['TOTAL_PHOTOS_VIEWED_C'] = 0
condition = df.loc[0:,'TOTAL_PHOTOS_VIEWED_C']                  [df['TOTAL_PHOTOS_VIEWED'] >TOTAL_PHOTOS_VIEWED_CHANGE]

df['TOTAL_PHOTOS_VIEWED_C'].replace(to_replace = condition,value = 1, inplace= True)


# In[9]:


# Copying the file
df_explanatory = df.copy()



# In[40]:


######################################################################################################################
# Create the standardized file to use in linear models ###############################################################
######################################################################################################################


#Creating df with x_variables
x_variables = ['CROSS_SELL_SUCCESS',
               'TOTAL_MEALS_ORDERED',
               'UNIQUE_MEALS_PURCH',
               'CONTACTS_W_CUSTOMER_SERVICE',
               'PRODUCT_CATEGORIES_VIEWED',
               'CANCELLATIONS_AFTER_NOON',
               'MOBILE_LOGINS',
               'LATE_DELIVERIES',
               'FOLLOWED_RECOMMENDATIONS_PCT',
               'AVG_PREP_VID_TIME',
               'LARGEST_ORDER_SIZE',
               'MASTER_CLASSES_ATTENDED',
               'MEDIAN_MEAL_RATING',
               'AVG_CLICKS_PER_VISIT',
               'AVG_PREP_VID_TIME_PER_ORDER',
               'AVG_CLICKS_PER_VISIT_PER_ORDER',
               'NUMBER_OF_RECOMMENDATION',
               'T_UNIQUE_MEALS_PURCH',
               'T_PRODUCT_CATEGORIES_VIEWED',
               'T_AVG_TIME_PER_SITE_VISIT',
               'T_PC_LOGINS',
               'T_MOBILE_LOGINS',
               'T_EARLY_DELIVERIES',
               'T_LATE_DELIVERIES',
               'T_AVG_PREP_VID_TIME',
               'T_AVG_CLICKS_PER_VISIT',
               'T_TOTAL_PHOTOS_VIEWED',
               'T_AVG_PREP_VID_TIME_PER_ORDER',
               'T_AVG_CLICKS_PER_VISIT_PER_ORDER',
               'UNIQUE_MEALS_PURCH_C',
               'AVG_CLICKS_PER_VISIT_C',
               'AVG_PREP_VID_TIME_C',
               'TOTAL_PHOTOS_VIEWED_C']     
                
#dropping

#df_explanatory= df_explanatory.drop(['REVENUE'],axis=1)

# preparing explanatory variable data
appen_data   = df_explanatory.loc[ : , x_variables]

# preparing response variable data
appen_target = df_explanatory.loc[:, 'REVENUE']

# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            appen_data,
            appen_target,
            test_size = 0.25,
            random_state = 222)


# In[27]:


######################################################################################################################
# Create the standardized file to use in linear models ###############################################################
######################################################################################################################

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()

# FITTING the scaler with housing_data
scaler.fit(appen_data)

# TRANSFORMING our data after fit
X_scaled = scaler.transform(appen_data)

# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# checking the results
X_scaled_df.describe().round(2)

X_scaled_df.columns = appen_data.columns

X_scaled_df

# this is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            appen_target,
            test_size = 0.25,
            random_state = 222)


# In[28]:


# adding labels to the scaled DataFrame
X_scaled_df.columns = appen_data.columns


# In[41]:


######################################################################################################################
## gradient boosting model object ####################################################################################
######################################################################################################################

# INSTANTIATING
elastic_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=150, max_depth=2, min_samples_leaf=120)

# FITTING 
elastic_fit = elastic_model.fit(X_train, y_train)


# PREDICTING 
elastic_pred = elastic_model.predict(X_test)

print('Training Score:', elastic_model.score(X_train, y_train).round(4))
print('Testing Score:',  elastic_model.score(X_test, y_test).round(4))

# saving
elastic_train_score = elastic_model.score(X_train, y_train).round(4)
elastic_test_score  = elastic_model.score(X_test, y_test).round(4)


# Comparing linear regression, lasso, SGD Regression and Elastic Net CV model object, Lasso provided the best testing score.
