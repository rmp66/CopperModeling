#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Copper_Set.csv')
df.head


# In[3]:


df.dtypes


# In[4]:


# Convert 'quantity tons' to float, setting errors='coerce' converts invalid parsing to NaN
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')


# In[5]:


# Convert 'status' column to string datatype
df['status'] = df['status'].astype('string')


# In[6]:


# Convert 'item type' column to string datatype
df['item type'] = df['item type'].astype('string')


# In[7]:


# Convert 'status' column to string datatype
df['material_ref'] = df['material_ref'].astype('string')


# In[8]:


# Function to convert date from float to datetime, handling nulls and already available timetsamps
def convert_date(x):
    if pd.isna(x):
        return np.nan
    elif isinstance(x, pd.Timestamp):
        return x
    try:
        # Convert float to int, then to string, and finally to datetime
        date_str = str(int(x))
        return pd.to_datetime(date_str, format='%Y%m%d')
    except ValueError:
        # Return NaN for any conversion errors encountered
        return np.nan

# Apply the conversion function to the columns
df['item_date'] = df['item_date'].apply(convert_date)
df['delivery date'] = df['delivery date'].apply(convert_date)


# In[9]:


df.dtypes


# In[10]:


df.isnull().sum()


# In[11]:


# Remove leading zeros in material ref
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['material_ref'] = df['material_ref'].fillna('1000A')
# Replace empty strings with '1000A'
df['material_ref'] = df['material_ref'].replace('', '1000A')


# In[12]:


df.head


# In[13]:


#Check if for null/empty value in 'material_ref' the field is filled with '1000A'
# Filter the DataFrame for rows where 'Material Reference' equals '1000A'
filtered_df = df[df['material_ref'] == '1000A']
# Select the first 5 records
first_5_records = filtered_df.head(5)
print(first_5_records)


# In[18]:


# Find null or empty 'id' items
null_or_empty_ids = df['id'].isnull() | (df['id'] == '')
# Indexes of the first two null or empty 'id' items
indexes_to_replace = np.where(null_or_empty_ids)[0][:2]
# Replace the first two null/empty 'id' items with 'A1' and 'A2'
if len(indexes_to_replace) >= 1:
    df.at[indexes_to_replace[0], 'id'] = 'A1'
if len(indexes_to_replace) >= 2:
    df.at[indexes_to_replace[1], 'id'] = 'A2'



# In[19]:


# Check if 'id' column has any null or empty values
if df['id'].isnull().any() or (df['id'] == '').any():
    print("The 'id' column still contains null or empty values.")
else:
    print("The 'id' column does not contain any null or empty values.")


# In[20]:


# Calculate the mode of the 'item_date' column. Mode returns a Series, so we take the first. Fill mode for nulls
item_date_mode_value = df['item_date'].mode()[0]
#print(item_date_mode_value)
df['item_date'].replace({None: item_date_mode_value, '': item_date_mode_value}, inplace=True)


# In[21]:


#do same for delivery date
delivery_date_mode_value = df['delivery date'].mode()[0]
#print(item_date_mode_value)
df['delivery date'].replace({None: delivery_date_mode_value, '': delivery_date_mode_value}, inplace=True)


# In[22]:


#Fill quantity null with mean. Wrong. Should have been done after droping outliers
quantity_tons_mean_value = df['customer'].mean()
print(quantity_tons_mean_value)
df['quantity tons'].replace({None: quantity_tons_mean_value, '': quantity_tons_mean_value}, inplace=True)


# In[23]:


#Replace customer null with mode
customer_mode_value = df['customer'].mode()[0]
print(customer_mode_value)
df['customer'].replace({None: customer_mode_value, '': customer_mode_value}, inplace=True)


# In[24]:


#do same for country
country_mode_value = df['country'].mode()[0]
print(country_mode_value)
df['country'].replace({None: country_mode_value, '': country_mode_value}, inplace=True)


# In[25]:


# Delete rows where 'status' column is null
df = df.dropna(subset=['status'])


# In[26]:


df.shape


# In[27]:


#Replace application null with mode
application_mode_value = df['application'].mode()[0]
print(application_mode_value)
df['application'].replace({None: application_mode_value, '': application_mode_value}, inplace=True)


# In[28]:


#Fill thickness null with median
thickness_median_value = df['thickness'].median()
print(thickness_median_value)
df['thickness'].replace({None: thickness_median_value, '': thickness_median_value}, inplace=True)


# In[29]:


# Check if 'material_ref' column has any null or empty values
if df['material_ref'].isnull().any() or (df['material_ref'] == '').any():
    print("The 'material_ref' column still contains null or empty values.")
else:
    print("The 'material_ref' column does not contain any null or empty values.")


# In[30]:


# Delete rows where 'status' column is null
df = df.dropna(subset=['selling_price'])


# In[31]:


df.shape


# In[32]:


df.isnull().sum()


# In[33]:


df['quantity tons'] = df['quantity tons'].fillna(df['quantity tons'].mean())


# In[34]:


df.isnull().sum()


# In[35]:


#Still there are null values


# In[36]:


df['customer'] = df['customer'].fillna(df['customer'].mode()[0])


# In[37]:


df['country'] = df['country'].fillna(df['country'].mode()[0])


# In[38]:


df['application'] = df['application'].fillna(df['application'].mode()[0])


# In[39]:


df['thickness'] = df['thickness'].fillna(df['thickness'].median())


# In[40]:


df.isnull().sum()


# In[41]:


df.shape


# In[39]:


# percentile list
perc = [.20, .40, .60, .80]
# list of dtypes to include
include = ['object', 'float', 'int', 'string', 'datetime']
# calling describe method
Desc = df.describe(percentiles=perc, include=include)
# display
Desc


# In[40]:


# All the 181670 ids are unique - object
# The item date is between dates 2020-07-02  to 2021-04-01  - datetime
# In quantity There are -ve min which need to be treated. MAx value may not be error compared to coorelation with selling price. 
#Mean is around 5875. 20-80% values make sense. # Post treatment skewness to be checked - Float 
# Customer does not have uniques alone and so may have repetitions. - Float
# Is country max 113 a wrong entry - NO - Float
# Status is categorical(9) variable with 'Won' repeated 116010/181670 times - Nominal - String
# Item type is categorical(7) variable with 'W' repeated 105615/181670 times - Nominal - String
# Application looks bit skewed with MAX value of 99- NO - Float
# Thickness is also skewed with max.value - YES- Float
# Width min is error? Other values look fine - YES - Float
# Material ref has no meaning/use and there were 16500+ missing values? Can be dropped - String
# Product ref may be combination of features but does not show much coorelation - Integer
# The delivery date is between dates 2019-04-01 to 2022-01-01  - datetime
# IN selling price there are -ve values requiring treatment. Max value is high and so correlations to be studied.
#overall looks like there are products which are different in top end with high quatity and specs with high sellin pric?
#They may not be wrong even though outliers.???


# In[42]:


#As per problem statement records which are not Status = won or lost can be removed
# Keep only rows where 'status' is either 'Won' or 'Lost'
df = df[(df['status'] == 'Won') | (df['status'] == 'Lost')]


# In[43]:


df.shape


# In[44]:


#Lets drop useless columns and convert everything into numbers
# Drop - item date, material ref, delivery date)
df.drop('item_date', axis=1, inplace=True)
df.drop('material_ref', axis=1, inplace=True)
df.drop('delivery date', axis=1, inplace=True)


# In[45]:


df.shape


# In[46]:


#Inspection of data reveals selling prices > 8000(10 records), <= 10(9)  are outliers with typo errors/very low price than peers 
# Remove rows where 'selling_price' is greater than 8000 
df = df[df['selling_price'] <= 8000]
df = df[df['selling_price'] >= 10]
df.shape


# In[47]:


#Lets look for outliers in selling price
sns.boxplot(y=df['selling_price'])
plt.show()


# In[48]:


#Remove quantity -ves
df = df[df['quantity tons'] >= 0]
df.shape


# In[49]:


#Lets look for outliers in thickness
sns.boxplot(y=df['thickness'])
plt.show()


# In[50]:


#Remove thickness > 50(2)
df = df[df['thickness'] <= 50]
df.shape


# In[51]:


#Lets look for outliers in thickness
sns.boxplot(y=df['thickness'])
plt.show()


# In[52]:


#Remove width <= 10(1)
df = df[df['width'] >= 10]
df.shape


# In[53]:


#Convert status & item type into numbers/encode
df.status.value_counts()


# In[54]:


#Refer Module 10 visualization
#Lets usderstand correlation between major features and selling price using heatmap
#Pick up numerical columns
num_cols = ['selling_price', 'quantity tons', 'customer', 'country', 'application', 'thickness', 'width', 'product_ref', ]
corrl = df[num_cols].corr()
corrl


# In[55]:


sns.heatmap(corrl, annot=True, cmap='coolwarm')
plt.show()


# In[56]:


df.dtypes


# In[57]:


#Since status is a target binary encoding used with Won = 1 and Lost = 0
df['status'] = df['status'].map({'Won': 1, 'Lost': 0})
print(df)


# In[58]:


#Convert item type into numbers/encode
# Rename the 'item type' column to 'item_type'
df.rename(columns={'item type': 'item_type'}, inplace=True)
df.item_type.value_counts()


# In[59]:


# for item_type lets go for dummy encoding
status_dummies = pd.get_dummies(df['item_type'], prefix='itype')
status_dummies = status_dummies.astype(int)
print(status_dummies)

# Join dummy variables back to the original DataFrame:
df_encoded = pd.concat([df, status_dummies], axis=1)
# Remove the original 'status' column
df = df_encoded.drop('item_type', axis=1)
# See df with the new dummy encoded columns
print(df)


# In[60]:


df.dtypes


# In[61]:


#Feature engineering
# Convert coded features to strings for combining 
df['customer'] = df['customer'].astype(str)
df['country'] = df['country'].astype(str)
df['application'] = df['application'].astype(str)
df['product_ref'] = df['product_ref'].astype(str)
df.dtypes


# In[62]:


# Create the combined feature
df['customer_application_product'] = df['customer'].astype(str) + '_' + df['application'].astype(str) + '_' + df['product_ref'].astype(str)
# Drop the original columns
df.drop(['customer', 'application', 'product_ref'], axis=1, inplace=True)
df.dtypes


# In[63]:


df.dtypes


# In[64]:


df.shape


# In[65]:


print(df)


# In[66]:


# Find no. of unques in 'customer_application_product' column
unique_count = df['customer_application_product'].nunique()
print(unique_count)


# In[68]:


#Calculate selling price range
min_price = df['selling_price'].min()
max_price = df['selling_price'].max()
price_range = max_price - min_price
print(price_range)


# In[69]:


#Lets encode 'country' using Binary encoder to minimize columns compared to dummy encoder
from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols=['country'])
df_encoded = encoder.fit_transform(df['country'])
print(df_encoded.head())
# Concatenate the encoded DataFrame with the original DataFrame
df = pd.concat([df, df_encoded], axis=1)
# Drop original 'country' column
df.drop('country', axis=1, inplace=True)
print(df.head())


# In[70]:


df.head


# In[71]:


df.dtypes


# In[72]:


#drop id
df.drop(['id'], axis=1, inplace=True)
df.dtypes


# In[73]:


# Check for missing values in the target column
print(df['status'].isnull().sum())
print(df['selling_price'].isnull().sum())


# In[74]:


#split the train/test data for classification and regression and apply Target encoding to  'customer_application_product' 
#column of  train and test data separately so that there is no leak


from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# Split data into features and targets
X_class = df.drop(['status', 'selling_price'], axis=1) 
y_class = df['status']

X_regress = df.drop(['status', 'selling_price'], axis=1)  
y_regress = df['selling_price']

# Train Test data split 
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_regress, X_test_regress, y_train_regress, y_test_regress = train_test_split(X_regress, y_regress, test_size=0.2, random_state=42)
#print(X_train_class.head())
#Save X_train_class to use in 'topredict_df' as we need 'customer_application_product' column
# Make a copy of X_train_class for prediction purposes
#X_train_class_topredict_df = X_train_class.copy()
X_train_class.to_csv('X_train_class_topredict.csv', index=False)
#print(X_train_class_topredict.head())

# Initialize the target encoder
encoder = TargetEncoder(smoothing=5)
# Fit and transform the encoder on the training sets
X_train_class['customer_application_product_encoded'] = encoder.fit_transform(X_train_class['customer_application_product'], y_train_class)
X_test_class['customer_application_product_encoded'] = encoder.transform(X_test_class['customer_application_product'])

X_train_regress['customer_application_product_encoded'] = encoder.fit_transform(X_train_regress['customer_application_product'], y_train_regress)
X_test_regress['customer_application_product_encoded'] = encoder.transform(X_test_regress['customer_application_product'])

# Drop original 'customer_application_product' column from train and test sets
X_train_class.drop('customer_application_product', axis=1, inplace=True)
X_test_class.drop('customer_application_product', axis=1, inplace=True)
X_train_regress.drop('customer_application_product', axis=1, inplace=True)
X_test_regress.drop('customer_application_product', axis=1, inplace=True)

#print(X_train_class_topredict_df.head())
print(X_train_class.head()) 
print(X_test_class.head()) 
print(X_train_regress.head()) 
print(X_test_regress.head()) 


# In[75]:


X_train_class.dtypes 


# In[76]:


X_train_regress.dtypes 


# In[77]:


#Use GridSearchCV to help on LogisticRegression hyper parameter tunning 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Provide the model and parameters to determine the best
log_reg = LogisticRegression()
param_grid = {
    'C': [0.1, 1, 10, 100],  # Example values
    'max_iter': [100, 500, 1000],   # Example values
    'solver': ['lbfgs', 'saga', 'liblinear']  # Depending on the compatibility with the penalty
}

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_class, y_train_class)

#Print best parameters and model
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Initialize the Logistic Regression model
#model = LogisticRegression()
model = LogisticRegression(C=10.0, max_iter=500, solver='lbfgs')
# Fit on the training data
model.fit(X_train_class, y_train_class)
# Predict the 'status' on the test data
y_pred = model.predict(X_test_class)
y_pred_proba = model.predict_proba(X_test_class)[:, 1]  # Probabilities for the positive class

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_class, y_pred)
precision = precision_score(y_test_class, y_pred)
recall = recall_score(y_test_class, y_pred)
f1 = f1_score(y_test_class, y_pred)
roc_auc = roc_auc_score(y_test_class, y_pred_proba)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


# In[79]:


confusion_matrix(y_test_class, y_pred)
#High false positives(predicted 'won' when actual is 'lost') is a concern


# In[80]:


import joblib

# Assuming X_train and y_train are already defined
# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train_class, y_train_class)

# Save the logistic regression model
joblib.dump(model, 'logistic_regression_model.pkl')


# In[81]:


#Now lets look at the head of what is predicted by the model using in test data
# Creating a DataFrame with all features from X_test_class
predicted_df = pd.DataFrame(X_test_class)
# Adding actual and predicted results to above features DataFrame
predicted_df['Actual Status'] = y_test_class  
predicted_df['Predicted Status'] = y_pred
predicted_df['Prediction Probability'] = y_pred_proba
# Display the head of the DataFrame
print(predicted_df.head())


# In[82]:


# Lets provide a set of features and ask the available Logistic Regression model predict
# Sample DataFrame to predict
data = {
    'quantity tons': [101.74289, 73.30, 5.144, 228.7, 5.1, 6.9, 51.9],
    'thickness': [3.1, 3.5, 4.0, 12.0, 0.68, 0.45, 15.0],
    'width': [1250.0, 1300.0, 1350.0, 1500.0, 2025.0, 830.0, 1014.0],
    'itype': ['S', 'IPL', 'Others', 'PL', 'SLAWR', 'W', 'WI'],
    'country': [25.0, 26.0, 30.0, 40.0, 77.0, 78.0, 78.0],
    'customer': [30147616.0, 30147800.0, 30210087.0, 30210753.0, 30267125.0, 30354200.0, 30210087.0],
    'application': [04.0, 10.0, 15.0, 41.0, 42.0, 59.0, 65.0],
    'product': [611993, 640665, 1671863738, 1668701718, 640405, 628377, 611993 ]
}
topredict_df = pd.DataFrame(data)
#print(topredict_df.shape)
#print(topredict_df.isnull().sum())

# for itype do dummy encoding as done in model. Since it was showing too many nulll the code was modified to limit to 7 records
if 'itype' in data:
    topredict_df = pd.get_dummies(topredict_df, columns=['itype'], prefix='itype')
    #print(topredict_df.head())
#print(topredict_df.shape)
#print(topredict_df.isnull().sum())

#Feature engineering
# Convert coded features to strings for combining 
topredict_df['customer'] = topredict_df['customer'].astype(str)
topredict_df['country'] = topredict_df['country'].astype(str)
topredict_df['application'] = topredict_df['application'].astype(str)
topredict_df['product'] = topredict_df['product'].astype(str)
#topredict_df.dtypes

#Lets encode 'country' using Binary encoder to minimize columns compared to dummy encoder as done in model
binary_encoder = BinaryEncoder(cols=['country'])
topredict_df_encoded = binary_encoder.fit_transform(topredict_df['country'])
#print(topredict_df_encoded.head())
#below code added as no. of countries in topredict df is less and so resulted in only 3 columns. Model had five columns 
expected_cols = ['country_0', 'country_1', 'country_2', 'country_3', 'country_4']  
for col in expected_cols:
    if col not in topredict_df_encoded.columns:
        topredict_df_encoded[col] = 0  # Add missing columns with default value of 0
# Concatenate the encoded DataFrame with the original DataFrame
topredict_df = pd.concat([topredict_df, topredict_df_encoded], axis=1)
# Drop original 'country' column
topredict_df.drop('country', axis=1, inplace=True)
#print(topredict_df.head())

#combine 'customer_application_product' on topredict-df 
topredict_df['customer_application_product'] = topredict_df['customer'].astype(str) + '_' + topredict_df['application'].astype(str) + '_' + topredict_df['product'].astype(str)
# Drop the original columns
topredict_df.drop(['customer', 'application', 'product'], axis=1, inplace=True) 
# Load the X_trin_class DataFrame with 'customer_application_product' column for training 
X_train_class_topredict_df = pd.read_csv('X_train_class_topredict.csv')
# To address index issue reset the index on both:
X_train_class_topredict_df.reset_index(drop=True, inplace=True)
y_train_class.reset_index(drop=True, inplace=True)
# Initialize and fit the Target Encoder
encoder = TargetEncoder(smoothing=5)
#Train on X_class
X_train_class_topredict_df['customer_application_product_encoded'] = encoder.fit_transform(
    X_train_class_topredict_df[['customer_application_product']],
    y_train_class
)
# Drop the original 'customer_application_product' column
X_train_class_topredict_df.drop('customer_application_product', axis=1, inplace=True)
topredict_df['customer_application_product_encoded'] = encoder.transform(topredict_df[['customer_application_product']])
topredict_df.drop('customer_application_product', axis=1, inplace=True)
print(topredict_df.head())

# Check for NaN values
print(topredict_df.isnull().sum())
#Load the saved model for prediction
model = joblib.load('logistic_regression_model.pkl')  # Adjust path if necessary

# Predict 'status'
predicted_status = model.predict(topredict_df)
predicted_probability = model.predict_proba(topredict_df)[:, 1]  # Probability of positive class

print("Predicted Status:", predicted_status)
print("Probability of 'Won' Status:", predicted_probability)


# In[83]:


import numpy as np

#Feature importance of trained Logistic Regression model
feature_names = X_train_class.columns  
coefficients = model.coef_[0]

# Pair feature names with coefficients
features_importance = zip(feature_names, coefficients)
sorted_features = sorted(features_importance, key=lambda x: x[1], reverse=True)

# Print sorted features by importance
print("Feature importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance}")


# In[84]:


#Lets use ExtraTreesClassifier for Status(Won/Lost) classification 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the ExtraTrees Classifier
# param_grid = {
#     'n_estimators': [100, 200,500],
#     'max_depth': [None, 10, 20, 30],
#     'max_features': ['sqrt', 'log2'],  # Removed 'auto' and using 'sqrt' explicitly
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# Since above hyper parameter findng code takes long time it was run once and adopted as below:

param_grid = {
    'n_estimators': [200], 
    'max_features': ['sqrt'], 
    'max_depth': [30, None],  
    'min_samples_split': [5],  
    'min_samples_leaf': [1],  
    'bootstrap': [False]  
}

# Setup GridSearchCV
ext_clf = ExtraTreesClassifier(random_state=42)
grid_search = GridSearchCV(estimator=ext_clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=0)
grid_search.fit(X_train_class, y_train_class)

# Get best estimator after hyperparameter tuning
best_ext_clf = grid_search.best_estimator_
#print("Best estimator:", grid_search.best_estimator_)

# Predict the 'status' on the test data
y_pred = best_ext_clf.predict(X_test_class)
y_pred_proba = best_ext_clf.predict_proba(X_test_class)[:, 1]  # Probabilities for the positive class

# Calculate and print metrics
accuracy = accuracy_score(y_test_class, y_pred)
precision = precision_score(y_test_class, y_pred)
recall = recall_score(y_test_class, y_pred)
f1 = f1_score(y_test_class, y_pred)
roc_auc = roc_auc_score(y_test_class, y_pred_proba)
conf_matrix = confusion_matrix(y_test_class, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)



# In[85]:


# Lets make a single prediction
row = [[5.1, 0.68, 2025.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0.689034 ]]
yhat = best_ext_clf.predict(row)
print('Predicted Class: %d' % yhat[0])


# In[86]:


#Lets use XGB Classifier for Status classification
from xgboost import XGBClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Initialize the classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Setup GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
grid_search_xgb.fit(X_train_class, y_train_class)

# Get the best model
best_xgb_clf = grid_search_xgb.best_estimator_
#print("Best estimator:", grid_search_xgb.best_estimator_)

# Predict on the test set
y_pred_xgb = best_xgb_clf.predict(X_test_class)
y_pred_proba_xgb = best_xgb_clf.predict_proba(X_test_class)[:, 1]  # Probabilities for the positive class

# Calculate evaluation metrics
accuracy_xgb = accuracy_score(y_test_class, y_pred)
precision_xgb = precision_score(y_test_class, y_pred)
recall_xgb = recall_score(y_test_class, y_pred)
f1_xgb = f1_score(y_test_class, y_pred)
roc_auc_xgb = roc_auc_score(y_test_class, y_pred_proba)
conf_matrix_xgb = confusion_matrix(y_test_class, y_pred)

# Output the results
print("Accuracy:", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1 Score:", f1_xgb)
print("ROC AUC Score:", roc_auc_xgb)
print("Confusion Matrix:\n", conf_matrix_xgb)


# In[87]:


# Lets make a single prediction
row_xgb = [[5.1, 0.68, 2025.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0.689034 ]]
yhat_xgb = best_xgb_clf.predict(row)
print('Predicted Class: %d' % yhat_xgb[0])


# In[88]:


#Lets do Random tree regression model for 'selling_price'
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#import numpy as np
# param_grid = {
#     'n_estimators': [100, 200],  # Reduced number of options
#     'max_features': ['sqrt'],  # Choose one based on prior knowledge about the dataset
#     'max_depth': [None, 20],  # Only None and a reasonably high depth
#     'min_samples_split': [2, 10],  # Reduced number of options
#     'min_samples_leaf': [1, 4],  # Reduced number of options
#     'bootstrap': [True, False]  # Simplify by choosing only one method
# }
#Best estimator: RandomForestRegressor(max_depth=20, max_features='sqrt', min_samples_split=10,
#n_estimators=200, oob_score=True, random_state=42)
#Based on above calc the param_grid minimised to save time:
param_grid = {
    'n_estimators': [200],  # Reduced number of options
    'max_features': ['sqrt'],  # Choose one based on prior knowledge about the dataset
    'max_depth': [20],  # Only None and a reasonably high depth
    'min_samples_split': [10],  # Reduced number of options
    'min_samples_leaf': [1, 4],  # Reduced number of options
    'bootstrap': [True, False]  # Simplify by choosing only one method
}

# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=42, oob_score=True)

# Setup GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
grid_search_rf.fit(X_train_regress, y_train_regress)

# Best estimator after hyperparameter tuning
best_rf_regressor = grid_search_rf.best_estimator_
#print("Best estimator:", grid_search_rf.best_estimator_)

# Predict the selling price on the test data
y_pred_regress = best_rf_regressor.predict(X_test_regress)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_regress, y_pred_regress)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_regress, y_pred_regress)
oob_score = best_rf_regressor.oob_score_  # Retrieve OOB score from the best model

# Print metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
print("OOB Score:", oob_score)


# In[89]:


# Lets make a single prediction of selling price
row_rf_reg = [[5.1, 0.68, 2025.0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0.789034 ]]
yhat_rf_reg = best_rf_regressor.predict(row_rf_reg)
print('Predicted Selling Price: %d' % yhat_rf_reg[0])


# In[90]:


#Lets do XGBoost regression model for 'selling_price'
from xgboost import XGBRegressor

# param_grid = {
#     'n_estimators': [500, 700, 1000],  # Number of gradient boosted trees. Equivalent to the number of boosting rounds
#     'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting. Range is [0,1]
#     'max_depth': [3, 5, 7],  # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
#     'subsample': [0.7, 0.9],  # Subsample ratio of the training instances. Setting it to 0.7 means that XGBoost would randomly sample 70% of the training data prior to growing trees. and this will prevent overfitting.
#     'colsample_bytree': [0.7, 0.9]  # Subsample ratio of columns when constructing each tree
# }
#Best estimator: XGBRegressor(colsample_bytree=0.9, learning_rate=0.1,  max_depth=7,  n_estimators=200, random_state=42, ...)
#Based on above calc the param_grid minimised to save time:
param_grid = {
    'n_estimators': [700],  # Number of gradient boosted trees. Equivalent to the number of boosting rounds
    'learning_rate': [0.1],  # Step size shrinkage used to prevent overfitting. Range is [0,1]
    'max_depth': [7],  # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
    'subsample': [0.7, 0.9],  # Subsample ratio of the training instances. Setting it to 0.7 means that XGBoost would randomly sample 70% of the training data prior to growing trees. and this will prevent overfitting.
    'colsample_bytree': [0.9]  # Subsample ratio of columns when constructing each tree
}


xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search_xgbre = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_search_xgbre.fit(X_train_regress, y_train_regress)

# Get the best estimator
best_xgb_regressor = grid_search_xgbre.best_estimator_
#print("Best estimator:", grid_search_xgbre.best_estimator_)
# Predict the selling price on the test data
y_pred_regress = best_xgb_regressor.predict(X_test_regress)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_regress, y_pred_regress)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_regress, y_pred_regress)

# Output the metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)



# In[91]:


# Lets make a single prediction of selling price unsing the XGB Regressor model
row_xgb_reg = [[5.1, 0.68, 2025.0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0.789034 ]]
yhat_xgb_reg = best_xgb_regressor.predict(row_xgb_reg)
print('Predicted Selling Price: %d' % yhat_xgb_reg[0])


# In[ ]:




