# AI_BOOTCAMP_Group_Project_3_Amazon

## Optimization Goals

### The objectives of model optimization were to: <br> 
Assess optimization capabilities: Ensure the transformer model can run efficiently on a local machine and optionally on a virtual environment like Google Colab.<br> 
Improve model efficiency: Identify steps to enhance performance and scalability.<br> 

### Further Optimization Steps to Consider
1) Expand dataset: Initially ran a subset of the dataset for feasibility; next step is to incorporate the full dataset. <br> 
2) Enhance data preprocessing <br> 
Further clean the dataset. <br> 
Add a column indicating whether Amazon flagged the data as spam. <br> 
3) Refine model implementation: Optimize both the transformer and optimization models to run seamlessly on local and virtual machines.

## Current Performance Metrics
XGBoost:
Review Body Accuracy: 0.467
Review Headlines Accuracy: 0.481

Random Forest:
Review Body Accuracy: 0.462
Review Headlines Accuracy: 0.483

Logistic Regression:
Review Body Accuracy: 0.475
Review Headlines Accuracy: 0.493

## Current Best Parameters Metrics 
### CV=3 with 20 candidates totaling 60 fits 

XGBoost: <br>
Best parameters for review body: subsample= 0.8, n_estimators= 700, min_child_weight= 5, max_depth= 9, learning_rate= 0.0944, gamma= 0.4, colsample_bytree 0.8 <br>
Best parameters for headlines: subsample= 0.7, n_estimators= 500, min_child_weight= 1, max_depth= 3, learning_rate= 0.0944, gamma= 0.2, colsample_bytree= 0.7 <br>
Further parameters to test: reg_alpha, reg_lambda, scale_pos_weight <br>

Random Forest: <br>
Best parameters for review body: n_estimators= 900, min_samples_split= 5, min_samples_leaf= 2, max_dept= None, bootstrap= False <br>
Best parameters for headlines: n_estimators= 900, min_samples_split= 5, min_samples_leaf= 2, max_depth= None, bootstrap= False <br>
Further parameters to test: max_features <br>

Logistic Regression: <br>
Best parameters for review body: solver= saga, penalty= l2, C= 0.46416 <br>
Best parameters for headlines: solver= saga, penalty= l2, C= 0.46416 <br>
Further parameters to test: max_iter <br>







