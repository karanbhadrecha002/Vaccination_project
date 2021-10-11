# Project Summary

- The Data science project which is given here is an analysis of Vaccination. The goal of this project is to predict whether a person will/will not take the H1N1 vaccine or the Seasonal vaccine from each feature of their data such as h1n1_concern, doctor_recc_h1n1, doctor_recc_seasonal, opinion_h1n1_vacc_effective, opinion_h1n1_risk, opinion_h1n1_sick_from_vacc, opinion_seas_vacc_effective etc. The Goal and Insights of the project as follows,

  1. A trained model which can predict whether a person will/will not take the H1N1 vaccine based on factors as inputs.
  2. A trained model which can predict whether a person will/will not take the Seasonal vaccine based on factors as inputs.
  
- The given data of Vaccination has the 26707 data to perform a higher level machine learning where it is well structured. The features present in the feature data are 36 and in label data are 3 in total. The Shape of the feature data is 26707x36 and label is 26707x3. 'respondent_id' is common feature b/w these two, so we combined feature data and label and make single dataset. The 38 features are classified into quantitative and qualitative where 10 features are qualitative and 26 features are quantitative. 

- The dataset is a complete labelled data and categorical which decides the machine learning algorithm to be used. The important aspects of the data are depending on the correlation of data between features and h1n1_vaccine, seasonal_vaccine. The analysis of the project has gone through the stage of distribution analysis, correlation analysis and analysis by each department to satisfy the project goal.

- The machine learning model which is used in this project is XGB classifier which predicted the nearby higher accuracy of 81% to 85% for h1n1 vaccine and 73% to 77% for seasonal vaccine. Since it is categorical labelled data, it has to go through the classifier machine learning techniques which will be suitable for this structured data. The numerical features are the most relevant in the model according to correlation technique.

### 1. Requirement
The data was given from the Datamites for this project where the collected source is Datamites. It is one of the leading data analytics and automation solutions. The data is not from the real organization. The whole project was done in Jupiter notebook with python platform.

### 2. Analysis
Data were analyzed by describing the features present in the data. the features play the bigger part in the analysis. the features tell the relation between the dependent and independent variables. Pandas also help to describe the datasets answering following questions early in our project. The futures present in the data are divided into numerical and categorical data.

##### Categorical Features
These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based. The categorical features as follows,

- age_group
- education
- race
- sex
- income_poverty
- marital_status
- rent_or_own
- employment_status
- hhs_geo_region
- census_msa
- employment_industry
- employment_occupation

##### Numerical Features
These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based. The Numerical Features as follows,
- respondent_id
- h1n1_concern
- h1n1_knowledge
- behavioral_antiviral_meds
- behavioral_avoidance
- behavioral_face_mask
- behavioral_wash_hands
- behavioral_large_gatherings
- behavioral_outside_home
- behavioral_touch_face
- doctor_recc_h1n1
- doctor_recc_seasonal
- chronic_med_condition
- child_under_6_months
- health_worker
- health_insurance
- opinion_h1n1_vacc_effective
- opinion_h1n1_risk
- opinion_h1n1_sick_from_vacc
- opinion_seas_vacc_effective
- opinion_seas_risk
- opinion_seas_sick_from_vacc
- household_adults
- household_children
- h1n1_vaccine
- seasonal_vaccine

##### Alphanumeric Features
Numerical, alphanumeric data within same feature. These are candidates for correcting goal. No alphanumeric data types are present in dataset.

##### Data Clean Check
The Data cleaning and wrangling is the part of the Data science project where the workflow the project go through this stage. because the damaged and missing data will lead to the disaster in the accuracy and quality of the model. If the data is already structured and cleaned, there is no need for the data cleaning. In this case, the given data have some outliers, we dectected and treated outliers by replacing with mean values of respective featuresand and make data cleaned and there are missing data present in this data.

##### Analysis by Visualization
we can able to perform the analysis by the visualisation of the data in two forms here in this project. One is by distributing the data and visualize using the density plotting. The other one is nothing but the correlation method which will visualise the correlation heat map and we can able to achieve the correlation values between the numerical features.
1. Distribution Plot
   - In general, one of the first few steps in exploring the data would be to have a rough idea of how the features are distributed with one another. To do so, we shall invoke the familiar kdeplot function from the Seaborn plotting library. The distribution has been done by both numerical and categorical features. it will show the overall idea about the density and majority of data present in a different level.

2. Correlation Plot
   - The next tool in a data explorer's arsenal is that of a correlation matrix. By plotting a correlation matrix, we have a very nice overview of how the features are related to one another. For a Pandas data frame, we can conveniently use the call .corr which by default provides the Pearson Correlation values of the columns pairwise in that data frame. The correlation works bet for numerical data where we are going to use all the numerical features present in the data.

From the above Pearson correlation heat plot, we can be to see that correlation between features with numerical values in the dataset. The heat signatures show the level of correlation from 0 to 1. from this distribution we can derive the facts as follows,
The most important features selected are h1n1_concern, doctor_recc_h1n1, doctor_recc_seasonal, opinion_h1n1_vacc_effective, opinion_h1n1_risk, opinion_h1n1_sick_from_vacc, opinion_seas_vacc_effective, opinion_seas_risk, opinion_seas_sick_from_vacc.

##### Machine Learning Model
The machine learning models used in this project is XGB classifier

Both machine learning algorithms are best for classification and labelled data. The train and test data are divided and fitted into the model and passed through the machine learning. Since we have already noted the severe imbalance in the values within the target variable, we implement the SMOTE method in the dealing with this skewed value via the learn Python package. The predicted data and test data achieved the accuracy rate of 81.91% using XGB classifier for H1N1 and accuracy rate of 74.87% using XGB classifier for Seasoanl.

From the above model,We select Random Forest classifier for fitting the model and than Evaluted the model.
In model Evalution part we calculate,
1. accuracy score
2. confusion matric
3. MSE and RMSE values
4. Precision
5. Recall
6. F1 score
7. Classification Report


### 3. Summary
The machine learning model has been fitted and predicted with the accuracy score. The goal of this project is nothing but the results from the analysis and machine learning model.

##### Goal 1:  A trained model which can predict whether a person will/will not take the H1N1 vaccine based on factors as inputs.
The trained model is created using the XGB classifier algorithm as follows, 
1. accuracy score is 81.95%
2. confusion matric 
                   
                   col_0         0      1
                 h1n1_vaccine            
                            0   5555   764
                            1    682  1012

3. MSE value =  0.180456757768626
4. RMSE value = 0.42480202185091587
5. Precision = 57%
6. Recall = 59.7%
7. F1 score = 58.3%
8. Classification Report
                       
                       precision    recall  f1-score   support

                   0       0.89      0.88      0.89      6319
                   1       0.57      0.59      0.58      1694

            accuracy                           0.82      8013
           macro avg       0.73      0.74      0.73      8013
        weighted avg       0.82      0.82      0.82      8013


##### Goal 1:  A trained model which can predict whether a person will/will not take the Seasonal vaccine based on factors as inputs.
The trained model is created using the XGB classifier algorithm as follows, 
1. accuracy score is 74.87%
2. confusion matric 
                  
                  col_0    0     1   
       seasonal_vaccine
                      0   3324   946   
                      1   1067  2676  
                                
3. MSE value =  0.25121677274429055
4. RMSE value = 0.501215295800408
5. Precision = 73.9%
6. Recall = 71.5%
7. F1 score = 72.7%
8. Classification Report
                       
                       precision    recall  f1-score   support

                   0       0.76      0.78      0.77      4270
                   1       0.74      0.71      0.73      3743

            accuracy                           0.75      8013
           macro avg       0.75      0.75      0.75      8013
        weighted avg       0.75      0.75      0.75      8013

