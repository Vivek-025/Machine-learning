#!/usr/bin/env python
# coding: utf-8

# <div style="
#     background-color:#0D47A1;
#     border-left:7px solid #0D47A1;
#     padding:10px;
#     border-radius:6px;
#     width:50%;
#     margin:auto;
#     text-align:center;
#     font-size:32px;
# ">
# <b>Machine Learning</b>
# </div>

# In[ ]:


import sklearn # sklearn means Scikit-learn, a Python library used for Machine Learning.sklearn contains ML algorithms.

print(sklearn.__version__)  
# Shows installed sklearn version


# In[4]:


import pandas as pd
#Import pandas for handling tables.

df = pd.read_csv("fruits.csv")  
# Reads fruits.csv and stores complete table in df
# Now df becomes:
#    Diameter   Weight   Fruit
# 0    3.0       30     Banana
# 1    6.0      100     Apple
# 2    6.1       95     Apple
# 3    3.2       35     Banana
# 4    5.5       80     Apple

'''
loc and iloc

loc  → df.loc[row_name , column_name]
iloc → df.iloc[row_no , column_no]
'''

X = df.iloc[:,:-1]  
# Take all rows and all columns except last column
# X contains inputs/features
# Example:
# Diameter   Weight
# 3.0          30
# 6.0         100

Y = df.iloc[:,-1]  
# Take only last column
# Y contains Target/outputs
# Example:
# Banana
# Apple
# Apple

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier() 
# Create a model that uses KNN algorithm", KNN full form is K-Nearest Neighbors.
# Creates an empty KNN model (student created,No Learning yet.)

model.fit(X,Y)  # fit() method is used to train the model using the given data
# Train model using fruit data
# Student learns:
# Diameter=3.0 Weight=30 → Banana
# Diameter=6.0 Weight=100 → Apple

#model.predict([[5.7,90]])  
# Predict fruit for:
# Diameter=5.7
# Weight=90

# Store predicted answers in variable a
a = model.predict(X) # Model trained on the same data and predicting on the same data
print(a)

# Print actual answers from dataset
print(list(Y))

# Calculating model accuracy manually
c = 0  
# Counter for correct predictions

for i,j in zip(a,Y):  
    # Combine predicted and actual answers together

    if(i==j):  # Check if prediction matches actual answer
        c = c + 1  # Increase correct count

print("Correct predictions:",c)  # Print number of correct predictions

# Accuracy = correct predictions / total rows × 100
print("Accuracy of model calculating manually:",(c/len(Y))*100)  

#Calculating model accuracy using built-in accuracy_score() method/function
from sklearn.metrics import accuracy_score
print("Accuracy of model using method:",accuracy_score(a,Y))


# In[21]:


s=[1,20,3,41,5,6,70,81,92,10,20,30,4,5,8,9,12,45,67]
from sklearn.model_selection import train_test_split
print(train_test_split(s,train_size=0.7,test_size=0.3,random_state=10))
train,test=train_test_split(s,train_size=0.7,test_size=0.3,random_state=10)
print(train)
print(test)


# # <span style="color:#2563eb">Train-Test Split Understanding</span>
# 
# ---
# 
# ## <span style="color:#16a34a">X_train</span>
# 
# **Data the machine learns from.**
# 
# ### Example
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 6.0      | 100    |
# | 3.2      | 35     |
# | 5.5      | 80     |
# 
# ---
# 
# ## <span style="color:#16a34a">Y_train</span>
# 
# **Correct answers for X_train.**
# 
# *(Y_train contains the correct answers that we give to the machine while learning.)*
# 
# ### Example
# 
# ```text
# Apple
# Banana
# Apple
# ```
# 
# ---
# 
# ## <span style="color:#16a34a">So together:</span>
# 
# | X_train | Y_train |
# | ------- | ------- |
# | 6.0,100 | Apple   |
# | 3.2,35  | Banana  |
# | 5.5,80  | Apple   |
# 
# ---
# 
# ## <span style="color:#16a34a">During:</span>
# 
# ```python
# model.fit(X_train, Y_train)
# ```
# 
# the machine learns:
# 
# ```text
# If diameter and weight are like this,
# the answer is Apple.
# 
# If diameter and weight are like this,
# the answer is Banana.
# ```
# 
# ---
# 
# ### <span style="color:#dc2626">Without Y_train</span>
# 
# Without Y_train, the machine would not know whether a fruit is Apple or Banana, so it would have nothing to learn.
# 
# ---
# 
# # <span style="color:#2563eb">X_test</span>
# 
# **New data that the machine did NOT learn from.**
# 
# ### Example
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 2.8      | 30     |
# | 6.1      | 95     |
# 
# The model has never seen these rows during training.
# 
# ---
# 
# # <span style="color:#2563eb">Y_test</span>
# 
# **Correct answers for X_test data.**
# 
# ### Example
# 
# ```text
# X_test          Y_test
# 
# 2.8,30    ->    Banana
# 6.1,95    ->    Apple
# ```
# 
# ---
# 
# # <span style="color:#2563eb">Why do we need them?</span>
# 
# After learning from:
# 
# ```text
# X_train + Y_train
# ```
# 
# we give:
# 
# ```text
# X_test
# ```
# 
# to the model and ask:
# 
# ```text
# What fruit is this?
# ```
# 
# The model predicts:
# 
# ```text
# Banana
# Apple
# ```
# 
# Then we compare those predictions with:
# 
# ```text
# Y_test
# ```
# 
# which contains the actual answers.
# 

# In[15]:


import pandas as pd
df = pd.read_csv("fruits.csv")
X = df.iloc[:,:-1] # Feature 
Y = df.iloc[:,-1]  # Target 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier() 

model.fit(X_train, Y_train)
# Train the model using X_train as input data
# and Y_train as the correct answers.

train=model.predict(X_train) 
# The machine is being tested on the same data it learned from.
# X_train → used for training
# X_train → used again for testing
# This gives training accuracy.

test=model.predict(X_test)
# The machine is being tested on data it did NOT learn from.
# X_train → used for training
# X_test → used for testing
# This gives testing accuracy.

print(train)
print(test)

from sklearn.metrics import accuracy_score

print("Training Accuracy :", accuracy_score(train,Y_train)) 
# Training Accuracy
# This means the machine learned on X_train,
# then we predict on the same X_train data,
# and compare those predictions with Y_train
# (the correct answers).

print("Testing Accuracy :",accuracy_score(test,Y_test))
# Testing Accuracy
# The machine predicts on unseen test data (X_test),
# and those predictions are compared with the correct answers (Y_test).


# ## <span style="color:yellow">Variance is average squared distance of data points from the mean.
# 
# ## <span style="color:cyan">An outlier is a data point that is very different from the rest of the data.
# 
# ---

# # <span style="color:#2563eb">📊 Model Evaluation</span>
# 
# ---
# 
# ## <span style="color:#16a34a">What is Model Evaluation?</span>
# 
# Model evaluation means:
# 
# > **Checking how good or bad a machine learning model performs.**
# 
# ---
# 
# # <span style="color:#2563eb">🎯 Bias Error</span>
# 
# **Bias Error = Error on the training data.**
# 
# ### High Bias
# 
# * Poor performance on training data.
# * The model is unable to learn the patterns properly.
# 
# **Example:**
# 
# ```text
# Training Accuracy = 55%
# Training Error = 45%
# ```
# 
# ---
# 
# # <span style="color:#2563eb">📈 Variance</span>
# 
# **Variance = Difference between training score and testing score.**
# 
# ---
# 
# ## <span style="color:#16a34a">✅ Low Variance</span>
# 
# ```text
# Training and testing performance is almost the same.
# ```
# 
# **Example:**
# 
# ```text
# Training Accuracy = 90%
# Testing Accuracy  = 88%
# ```
# 
# Difference:
# 
# ```text
# 2%
# ```
# 
# ---
# 
# ## <span style="color:#dc2626">⚠️ High Variance</span>
# 
# ```text
# Training and testing performance differs significantly.
# ```
# 
# **Example:**
# 
# ```text
# Training Accuracy = 95%
# Testing Accuracy  = 60%
# ```
# 
# Difference:
# 
# ```text
# 35%
# ```
# 
# ---
# 
# #  <span style="color:#dc2626">Overfitting
# ## A model with high variance.
# 
# ---
# 
# #  <span style="color:#2563eb">Underfitting
# # A model with high bias.
# 
# ---
# 
# #  <span style="color:#16a34a">Best Fit
# ## Low Bias + Low Variance
# 

# # <span style="color:#2563eb">🎯 Techniques Used to Make a Model Best Fit</span>
# 
# ### <span style="color:#16a34a">Goal</span>
# 
# ```text
# Low Bias + Low Variance = Best Fit Model
# ```
# 
# ---
# 
# ## <span style="color:#dc2626">1. Remove Outliers</span>
# 
# **Outliers** are data points that are very far from the rest of the data.
# 
# ### Purpose
# 
# ```text
# Remove unusual values that may negatively affect model learning.
# ```
# 
# ### Example
# 
# ```text
# 50
# 55
# 60
# 58
# 53
# 1000   ← Outlier
# ```
# 
# ---
# 
# ## <span style="color:#ea580c">2. Feature Scaling</span>
# 
# Bring all feature values to a similar range.
# 
# 
# ### Example
# 
# #### Before Scaling
# 
# ```text
# Age    = 20
# Salary = 500000
# ```
# 
# #### After Scaling
# 
# ```text
# Age    = 0.3
# Salary = 0.5
# ```
# 
# ---
# 
# ## <span style="color:#9333ea">3. Increase / Decrease Features</span>
# 
# ### Increase Features
# 
# ```text
# Add useful columns that help the model learn better.
# ```
# 
# ### Decrease Features
# 
# ```text
# Remove unnecessary columns that do not help prediction.
# ```
# 
# ---
# 
# ## <span style="color:#0891b2">4. Increase Samples</span>
# 
# Collect more data.
# 
# ### Sample
# 
# A sample is:
# 
# ```text
# One complete row containing features and target.
# ```
# 
# ### Example
# 
# | Diameter | Weight | Fruit |
# | -------- | ------ | ----- |
# | 6.0      | 100    | Apple |
# 
# ```text
# This entire row is one sample.
# ```
# 
# ### Purpose
# 
# ```text
# More samples help the model learn better patterns.
# ```
# 
# ---
# 
# ## <span style="color:#16a34a">5. Hyperparameter Tuning</span>
# 
# Change algorithm settings to find the best performance.
# 
# ### Example (KNN)
# 
# ```python
# KNeighborsClassifier(n_neighbors=3)
# 
# KNeighborsClassifier(n_neighbors=5)
# 
# KNeighborsClassifier(n_neighbors=7)
# ```
# 
# ### Purpose
# 
# ```text
# Find the settings that produce the best results.
# ```
# 
# ---
# 
# ## <span style="color:#2563eb">6. Change Algorithm</span>
# 
# If one algorithm performs poorly, try another algorithm.
# 
# ### Example
# 
# ```text
# KNN
# Decision Tree
# Random Forest
# Logistic Regression
# ```
# 
# ### Purpose
# 
# ```text
# Different algorithms perform differently on different datasets.
# ```
# 
# ---
# 
# # <span style="color:#16a34a">📌 Quick Revision</span>
# 
# | Technique                    | Purpose                                  |
# | ---------------------------- | ---------------------------------------- |
# | Remove Outliers              | Remove unusual values                    |
# | Feature Scaling              | Bring features to similar ranges         |
# | Increase / Decrease Features | Add useful features, remove useless ones |
# | Increase Samples             | Collect more data                        |
# | Hyperparameter Tuning        | Change algorithm settings                |
# | Change Algorithm             | Try a different ML algorithm             |
# 
# ---
# 
# # <span style="color:#dc2626">🏆 Final Goal</span>
# 
# ```text
# Low Bias + Low Variance
#           =
#       Best Fit Model
# ```
# 

# In[22]:


import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("fruits.csv")  
x=df.iloc[:,:-1] 
y=df.iloc[:,-1]


# In[23]:


sb.scatterplot(x=df.diameter,y=df.weight,hue=df.FruitName)
# Create a scatter plot.

# X-axis:
# diameter

# Y-axis:
# weight

# Each dot means One fruit
# Example:
# Diameter = 6
# Weight = 100
# becomes one point(one fruit) on the graph.

#hue = df.Fruitname means Color points based on this column.
#means Use different colors for different fruit names.

plt.show()
# Display the graph.

print(df[df.FruitName=="Banana"])
# Show all rows where FruitName is Banana.

df.drop(5,axis=0,inplace=True)
# Remove row index 5
# Because the diameter of row 5 is very different from the other Banana rows,
# it is considered an outlier. Therefore, we removed that outlier.

# axis=0 means: Row operation
# axis=0 -> rows
# axis=1 -> columns

# inplace=True means:
# Modify the original DataFrame directly.

print(df[df.FruitName=="Banana"])
#Show Banana rows again after deleting row 5.


# ### 1. Load the fruits dataset.
#  
# ### 2. Separate features and target.
# 
# ### 3. Visualize Apples and Bananas using a scatter plot.
# 
# ### 4. Checking Banana rows.
# 
# ### 5. Remove row number 5 as it is outlier.
# 
# ### 6. Check the Banana rows again to confirm deletion.
# 
# ---

# In[28]:


from sklearn.preprocessing import StandardScaler  # standard scaler is a class
obj=StandardScaler()  # object created to use class 

ss=obj.fit_transform(X) #giving X which contains diameter and weight values
#This does 2 things together:

# fit()
# Study the feature data X.
# For each feature column:
# Find Mean and Standard Deviation
# Diameter Mean = 4.6045
# Diameter Std  = 1.6286
# Weight Mean = 78.8182
# Weight Std  = 29.9206

# transform() does scaling means
# Apply:
# (Value - Mean) / Standard Deviation
# to every value.

print(ss) # ss conatins scaled values 


# # <span style="color:#2563eb">📏 Feature Scaling</span>
# 
# ## <span style="color:#16a34a">Definition</span>
# 
# **Feature Scaling** means transforming feature values so that all features are on a similar scale/range.
# 
# ---
# 
# ## <span style="color:#2563eb">Why Do We Need Feature Scaling?</span>
# 
# Consider the features from the fruits dataset:
# 
# | Feature  | Range     |
# | -------- | --------- |
# | Diameter | 2 to 7    |
# | Weight   | 25 to 120 |
# 
# Notice:
# 
# ```text
# Diameter ≈ 2 to 7
# 
# Weight ≈ 25 to 120
# ```
# 
# The **Weight** values are much larger than the **Diameter** values.
# 
# ---
# 
# ## <span style="color:#dc2626">Problem Without Scaling</span>
# 
# Example:
# 
# ```text
# Diameter = 3
# Weight   = 120
# ```
# 
# The machine learning algorithm may think:
# 
# ```text
# Weight is much more important than Diameter
# ```
# 
# simply because its numerical values are larger.
# 
# This can negatively affect the model's learning process.
# 
# ---
# 
# ## <span style="color:#16a34a">Solution: Feature Scaling</span>
# 
# Convert all features to a similar range.
# 
# ### Before Scaling
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 3.0      | 30     |
# | 7.1      | 120    |
# | 2.5      | 50     |
# 
# ---
# 
# ### After Scaling
# 
# | Diameter | Weight |
# | -------- | ------ |
# | -0.8     | -1.1   |
# | 1.7      | 2.0    |
# | -1.0     | -0.4   |
# 
# ---
# 
# ## <span style="color:#2563eb">Result After Scaling</span>
# 
# ```text
# Diameter ≈ -2 to 2
# 
# Weight ≈ -2 to 2
# ```
# 
# Now both features are in a similar range.
# 
# ---
# 
# ## <span style="color:#16a34a">Benefits</span>
# 
# ✅ No feature gets unfair importance.
# 
# ✅ Distance-based algorithms work better.
# 
# ✅ Model performance can improve.
# 
# ---
# 

# # <span style="color:#2563eb">📏 StandardScaler Working (Using Fruits Dataset)</span>
# 
# ---
# 
# # <span style="color:#16a34a">Step 1: StandardScaler Calculates Statistics</span>
# 
# For the **Diameter** feature:
# 
# ```text
# Mean (Diameter) = 4.6045
# Standard Deviation (Diameter) = 1.6286
# ```
# 
# For the **Weight** feature:
# 
# ```text
# Mean (Weight) = 78.8182
# Standard Deviation (Weight) = 29.9206
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 2: Take the First Row</span>
# 
# ### Original Row
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 3.0      | 30     |
# 
# ---
# 
# # <span style="color:#2563eb">Step 3: Scale Diameter</span>
# 
# ### Formula
# 
# ```text
# (Value - Mean) / Standard Deviation
# ```
# 
# ### Calculation
# 
# ```text
# (3.0 - 4.6045) / 1.6286
# =
# -1.6045 / 1.6286
# =
# -0.985
# ```
# 
# ### Scaled Diameter
# 
# ```text
# -0.985
# ```
# 
# ---
# 
# # <span style="color:#2563eb">Step 4: Scale Weight</span>
# 
# ### Formula
# 
# ```text
# (Value - Mean) / Standard Deviation
# ```
# 
# ### Calculation
# 
# ```text
# (30 - 78.8182) / 29.9206
# =
# -48.8182 / 29.9206
# =
# -1.632
# ```
# 
# ### Scaled Weight
# 
# ```text
# -1.632
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 5: Result</span>
# 
# ### Before Scaling
# 
# ```text
# 3.0 , 30
# ```
# 
# ### After Scaling
# 
# ```text
# -0.985 , -1.632
# ```
# 
# This matches the StandardScaler output:
# 
# ```text
# [-0.98524588 -1.63159341]
# ```
# 
# ---
# 
# # <span style="color:#2563eb">What Happens for Every Feature?</span>
# 
# For each feature column separately:
# 
# ```text
# 1. Find Mean
# 
# 2. Find Standard Deviation
# 
# 3. Apply
# 
# (Value - Mean) / Standard Deviation
# ```
# 
# ---
# 
# # <span style="color:#dc2626">Important Note</span>
# 
# You do NOT manually perform these calculations.
# 
# The **StandardScaler** class automatically:
# 
# ```text
# 1. Finds the Mean
# 
# 2. Finds the Standard Deviation
# 
# 3. Applies
# 
# (Value - Mean) / Standard Deviation
# 
# to every feature value
# ```
# 
# ---
# 
# 

# # <span style="color:#2563eb">📊 How Standard Deviation is Calculated</span>
# 
# Standard Deviation measures how spread out the data is from its mean (average).
# 
# ---
# 
# ## <span style="color:#16a34a">Steps</span>
# 
# ```text id="ztg7jq"
# 1. Find the Mean
# 
# 2. Compute (Value - Mean)
# 
# 3. Square the Differences
# 
# 4. Take the average of the squared differences (variance).
# 
# 5. Take the Square Root of Variance(Standard deviation = √Variance)
# 
# 6. Result = Standard Deviation
# ```
# 

# ## <span style="color:#16a34a">StandardScaler
# 
# → Uses Mean and Standard Deviation
# 
# → No fixed range
# 
# ## <span style="color:#2563eb">MinMaxScaler
# 
# → Uses Minimum and Maximum values
# 
# → Scales everything between 0 and 1 (default)

# In[17]:


from sklearn.neighbors import KNeighborsClassifier as kn
import pandas as pd
df = pd.read_csv("fruits.csv")  
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3)
from sklearn.preprocessing import StandardScaler        
obj=StandardScaler()

x_train_new=obj.fit_transform(X_train)
# fit()
# Find mean and standard deviation for diameter and weight.

# transform()
# Apply (Value - Mean) / Standard Deviation to every value.
# Result = Scaled training data, stored in x_train_new

x_test_new = obj.transform(X_test)
# Scale Testing Features
# Uses the SAME Mean and Std learned from X_train
# We DO NOT use: obj.fit_transform(X_test) because that would calculate a NEW Mean and Std from the test data. 
# Test data must remain unseen. Therefore we only use transform() which uses the SAME Mean and Std learned from X_train.

# Example:
# Original Features:
# Diameter   Weight
# 3.0        30
# 6.0        100
# After Scaling:
# -0.98     -1.63
#  0.85      0.70
# We train KNN using these scaled values.


model = kn()
# Create KNN Model

model.fit(x_train_new, Y_train)
# Train Model 
#x_train_new contains: 
# Scaled Diameter 
# Scaled Weight  
# Y_train contains: 
# Fruit Names  
# Example: 
# -0.98 -1.63 -> Banana 
# 0.85 0.70 -> Apple 
# The model learns this relationship.


pre = model.predict(x_train_new)
# Prediction on Training Data

prep = model.predict(x_test_new)
# Prediction on Testing Data
# We are predicting on unseen data.

# Accuracy
from sklearn.metrics import accuracy_score

print("Training Accuracy :", accuracy_score(pre, Y_train))
print("Testing Accuracy  :", accuracy_score(prep, Y_test))

# Important Realization We are NOT training KNN on Diameter = 3.0, Weight = 30 
# We ARE training KNN on Scaled Diameter = -0.98, Scaled Weight = -1.63 
# because KNN performs better when features are on a similar scale.


# In[19]:


from sklearn.neighbors import KNeighborsClassifier as kn
import pandas as pd
df = pd.read_csv("fruits.csv")  
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3)
from sklearn.preprocessing import MinMaxScaler
obj=MinMaxScaler()

x_train_new=obj.fit_transform(X_train)
# fit()
# Find Minimum and Maximum values for Diameter and Weight.

# transform()
# Apply: (Value - Minimum) / (Maximum - Minimum) to every feature value.

# Result:
# Values are scaled between 0 and 1.

x_test_new = obj.transform(X_test)
# Scale Testing Features
# Uses the SAME Minimum and Maximum learned from X_train.

# Example:
# Original Features:
# Diameter   Weight
# 3.0        30
# 6.0        100
# After Scaling:
# 0.15      0.05
# 0.77      0.82
# We train KNN using these scaled values.


model = kn()
# Create KNN Model

model.fit(x_train_new, Y_train)
# Train Model 
#x_train_new contains: 
# Scaled Diameter 
# Scaled Weight  
# Y_train contains: 
# Fruit Names  
# Example: 
# -0.98 -1.63 -> Banana 
# 0.85 0.70 -> Apple 
# The model learns this relationship.


pre = model.predict(x_train_new)
# Prediction on same Training Data

prep = model.predict(x_test_new)
# Prediction on Testing Data
# We are predicting on unseen data.

# Accuracy
from sklearn.metrics import accuracy_score

print("Training Accuracy :", accuracy_score(pre, Y_train))
print("Testing Accuracy  :", accuracy_score(prep, Y_test))

#Important Realization We are NOT training KNN on: 
# We ARE training KNN on scaled values.
# Example:
# Scaled Diameter = 0.15
# Scaled Weight   = 0.05


# # <span style="color:#2563eb">📏 MinMaxScaler Working (Using Fruits Dataset)</span>
# 
# ---
# 
# # <span style="color:#16a34a">Step 1: MinMaxScaler Calculates Statistics</span>
# 
# For the **Diameter** feature:
# 
# ```text
# Minimum (Diameter) = 2.3
# Maximum (Diameter) = 7.1
# ```
# 
# For the **Weight** feature:
# 
# ```text
# Minimum (Weight) = 25
# Maximum (Weight) = 120
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 2: Take the First Row</span>
# 
# ### Original Row
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 3.0      | 30     |
# 
# ---
# 
# # <span style="color:#2563eb">Step 3: Scale Diameter</span>
# 
# ### Formula
# 
# ```text
# (Value - Minimum) / (Maximum - Minimum)
# ```
# 
# ### Calculation
# 
# ```text
# (3.0 - 2.3) / (7.1 - 2.3)
# 
# =
# 0.7 / 4.8
# 
# =
# 0.1458
# ```
# 
# ### Scaled Diameter
# 
# ```text
# 0.1458
# ```
# 
# ---
# 
# # <span style="color:#2563eb">Step 4: Scale Weight</span>
# 
# ### Formula
# 
# ```text
# (Value - Minimum) / (Maximum - Minimum)
# ```
# 
# ### Calculation
# 
# ```text
# (30 - 25) / (120 - 25)
# 
# =
# 5 / 95
# 
# =
# 0.0526
# ```
# 
# ### Scaled Weight
# 
# ```text
# 0.0526
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 5: Result</span>
# 
# ### Before Scaling
# 
# ```text
# 3.0 , 30
# ```
# 
# ### After Scaling
# 
# ```text
# 0.1458 , 0.0526
# ```
# 
# ---
# 
# # <span style="color:#2563eb">What Happens for Every Feature?</span>
# 
# For each feature column separately:
# 
# ```text
# 1. Find Minimum Value
# 
# 2. Find Maximum Value
# 
# 3. Apply
# 
# (Value - Minimum)
# ----------------------
# (Maximum - Minimum)
# ```
# 
# ---
# 
# # <span style="color:#dc2626">Important Note</span>
# 
# You do NOT manually perform these calculations.
# 
# The **MinMaxScaler** class automatically:
# 
# ```text
# 1. Finds the Minimum Value
# 
# 2. Finds the Maximum Value
# 
# 3. Applies
# 
# (Value - Minimum)
# ----------------------
# (Maximum - Minimum)
# 
# to every feature value
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Result of MinMaxScaler</span>
# 
# ```text
# Minimum Value → 0
# 
# Maximum Value → 1
# ```
# 
# Therefore:
# 
# All scaled values always will lie between:
# 
# ```text
# 0 and 1
# ```
# 
# ---
# 
# 

# In[4]:


from sklearn.neighbors import KNeighborsClassifier as kn
import pandas as pd

df = pd.read_csv("fruits.csv")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3)

from sklearn.preprocessing import MaxAbsScaler

obj = MaxAbsScaler()

x_train_new = obj.fit_transform(X_train)

# fit()
# Find Maximum Absolute Value for Diameter and Weight.

# Example:
# Diameter values: 3.0, 4.5, 7.1
# Maximum Absolute Diameter = 7.1

# Weight values: 30, 70, 120
# Maximum Absolute Weight = 120

# transform()
# Apply the formula:
# (Value / Maximum Absolute Value) to every feature value.

# Example:
# Diameter = 3.0
# 3.0 / 7.1 = 0.4225

# Weight = 30
# 30 / 120 = 0.25

# Result:
# Values are scaled between -1 and 1.
# (For positive data like fruits, values lie between 0 and 1.)

# Scaled training data is stored in x_train_new.


x_test_new = obj.transform(X_test)

# Scale Testing Features
# Uses the SAME Maximum Absolute Values
# learned from X_train.
# We DO NOT use:
# obj.fit_transform(X_test)
# because that would calculate NEW Maximum Absolute Values from the test data.
# Test data must remain unseen.
# Therefore we only use transform() which uses the SAME Maximum Absolute Values learned from X_train.

# Example:
# Original Features:
# Diameter   Weight
# 3.0        30
# 6.0        100

# After Scaling:
# 0.4225     0.25
# 0.8450     0.8333

# We train KNN using these scaled values.

model = kn()
# Create KNN Model

model.fit(x_train_new, Y_train)

# Train Model

# x_train_new contains:
# Scaled Diameter
# Scaled Weight

# Y_train contains:
# Fruit Names

# Example:
# 0.4225  0.25   -> Banana
# 0.8450  0.8333 -> Apple

# The model learns this relationship.

pre = model.predict(x_train_new)

# Prediction on same Training Data

prep = model.predict(x_test_new)

# Prediction on Testing Data
# We are predicting on unseen data.

from sklearn.metrics import accuracy_score

# Accuracy

print("Training Accuracy :", accuracy_score(pre, Y_train))
print("Testing Accuracy  :", accuracy_score(prep, Y_test))


# # <span style="color:#2563eb">📏 MaxAbsScaler Working (Using Fruits Dataset)</span>
# 
# ---
# 
# # <span style="color:#16a34a">Step 1: MaxAbsScaler Calculates Statistics</span>
# 
# For each feature column, it finds:
# 
# ```text
# Maximum Absolute Value
# ```
# 
# ### Formula
# 
# ```text
# Scaled Value
# =
# Value / Maximum Absolute Value
# ```
# 
# ---
# 
# ### Suppose the Training Data Contains
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 3.0      | 30     |
# | 4.5      | 70     |
# | 7.1      | 120    |
# 
# ---
# 
# For the **Diameter** feature:
# 
# ```text
# Maximum Absolute Diameter = 7.1
# ```
# 
# For the **Weight** feature:
# 
# ```text
# Maximum Absolute Weight = 120
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 2: Take the First Row</span>
# 
# ### Original Row
# 
# | Diameter | Weight |
# | -------- | ------ |
# | 3.0      | 30     |
# 
# ---
# 
# # <span style="color:#2563eb">Step 3: Scale Diameter</span>
# 
# ### Formula
# 
# ```text
# Value / Maximum Absolute Value
# ```
# 
# ### Calculation
# 
# ```text
# 3.0 / 7.1
# 
# =
# 0.4225
# ```
# 
# ### Scaled Diameter
# 
# ```text
# 0.4225
# ```
# 
# ---
# 
# # <span style="color:#2563eb">Step 4: Scale Weight</span>
# 
# ### Formula
# 
# ```text
# Value / Maximum Absolute Value
# ```
# 
# ### Calculation
# 
# ```text
# 30 / 120
# 
# =
# 0.25
# ```
# 
# ### Scaled Weight
# 
# ```text
# 0.25
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Step 5: Result</span>
# 
# ### Before Scaling
# 
# ```text
# 3.0 , 30
# ```
# 
# ### After Scaling
# 
# ```text
# 0.4225 , 0.25
# ```
# 
# ---
# 
# # <span style="color:#2563eb">What Happens For Every Feature?</span>
# 
# For each feature column separately:
# 
# ```text
# 1. Find Maximum Absolute Value
# 
# 2. Apply
# 
# Value
# -----------------------
# Maximum Absolute Value
# ```
# 
# ---
# 
# # <span style="color:#dc2626">Important Note</span>
# 
# You do NOT manually perform these calculations.
# 
# The **MaxAbsScaler** class automatically:
# 
# ```text
# 1. Finds the Maximum Absolute Value
# 
# 2. Divides every value by that maximum absolute value
# ```
# 
# ---
# 
# # <span style="color:#16a34a">Result of MaxAbsScaler</span>
# 
# The largest absolute value becomes:
# 
# ```text
# 1
# ```
# 
# All other values become:
# 
# ```text
# Between -1 and 1
# ```
# 
# If your data contains only positive values (like the Fruits Dataset):
# 
# ```text
# Between 0 and 1
# ```
# 
# ---
# 
# # <span style="color:#2563eb">Example with Negative Values</span>
# 
# Suppose a feature contains:
# 
# ```text
# -100
# -50
#  20
#  80
# ```
# 
# Maximum Absolute Value:
# 
# ```text
# |-100| = 100
# |80|   = 80
# 
# Maximum Absolute Value = 100
# ```
# 
# After Scaling:
# 
# ```text
# -100 / 100 = -1.0
# 
# -50 / 100 = -0.5
# 
# 20 / 100 = 0.2
# 
# 80 / 100 = 0.8
# ```
# 
# Result:
# 
# ```text
# Values lie between -1 and 1
# ```
# 
# 
# 
# ---
# 
# # <span style="color:#16a34a">Key Takeaway</span>
# 
# ```text
# MaxAbsScaler
# 
# 1. Finds Maximum Absolute Value
# 
# 2. Divides every value by it
# 
# 3. Largest absolute value becomes 1
# 
# 4. Output lies between -1 and 1
# 
# 5. Works feature-wise (column-wise)
# ```
# 

# # 📊 Feature Scaling Comparison
# 
# | Scaler | Formula | Uses (Learned During `fit()`) | Scaled Value Range |
# |---------|---------|---------|---------|
# | **StandardScaler** | `(Value - Mean) / Standard Deviation` | Mean, Standard Deviation | No fixed range |
# | **MinMaxScaler** | `(Value - Minimum) / (Maximum - Minimum)` | Minimum, Maximum | **[0, 1]** |
# | **MaxAbsScaler** | `Value / Maximum Absolute Value` | Maximum Absolute Value | **[-1, 1]** |
# 

# In[15]:


from sklearn.neighbors import KNeighborsClassifier as kn
import pandas as pd
df = pd.read_csv("fruits.csv")  
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3)
from sklearn.preprocessing import Binarizer
obj=Binarizer(threshold=4.0)
# Create Binarizer Object
# Threshold = 4.0
# Rule:
# Value > 4.0  -> 1
# Value <= 4.0 -> 0

x_train_new=obj.fit_transform(X_train)
# fit()
# Store Threshold = 4.0

# transform()
# Apply:
# Value > 4.0  -> 1
# Value <= 4.0 -> 0

# Example:
# Original Row:
# Diameter Weight
# 3.0      30

# After Transformation:
# 0        1

# x_train_new contains transformed features consisting only of 0s and 1s

x_test_new = obj.transform(X_test)
# Transform Testing Features
# Uses the SAME Threshold (4.0) stored during fit()
# We do NOT use:  obj.fit_transform(X_test) because test data must remain unseen
# x_test_new contains transformed testing features (0s and 1s)


model = kn()

model.fit(x_train_new, Y_train)
# Train KNN Model
# KNN stores:
# x_train_new
# Y_train

# Example:
# 1 1 -> Apple
# 1 1 -> Banana
# 0 1 -> Apple

# KNN mainly stores these examples
# and later uses nearest neighbors
# to make predictions

pre = model.predict(x_train_new)
# Predict on same Training Data

prep = model.predict(x_test_new)
# Predict on Testing Data
# Predictions for unseen data

from sklearn.metrics import accuracy_score

print("Training Accuracy :", accuracy_score(pre, Y_train))
print("Testing Accuracy  :", accuracy_score(prep, Y_test))


# # 🔲 Binarizer
# 
# | Transformer | Formula / Rule | Uses (Learned During `fit()`) | Range of Scaled Values |
# |------------|------------|------------|------------|
# | **Binarizer** | `Value > Threshold → 1`<br>`Value ≤ Threshold → 0` | Threshold | **0 or 1** |
# 
# ---
# 
# ## 📝 How It Works
# 
# Binarizer converts numerical values into only:
# 
# ```text
# 0 or 1
# ```
# 
# using a specified threshold.
# 
# ### Rule
# 
# ```text
# Value > Threshold  → 1
# 
# Value ≤ Threshold → 0
# ```
# 
# ---
# 
# ## 🎯 Example
# 
# Suppose:
# 
# ```python
# threshold = 4
# ```
# 
# Original Values:
# 
# ```text
# 3.0
# 4.5
# 7.1
# 2.8
# ```
# 
# After Binarization:
# 
# ```text
# 0
# 1
# 1
# 0
# ```
# 
# ---
# 
# ## ⚙️ What Does `fit()` Do?
# 
# Unlike:
# 
# ```text
# StandardScaler
# MinMaxScaler
# MaxAbsScaler
# ```
# 
# Binarizer does not calculate:
# 
# ```text
# Mean
# 
# Standard Deviation
# 
# Minimum
# 
# Maximum
# 
# Absolute max
# ```
# 
# It simply stores the threshold value.
# 
# Example:
# 
# ```python
# obj = Binarizer(threshold=4)
# ```
# 
# Stored:
# 
# ```text
# Threshold = 4
# ```
# 
# ---

# # <span style="color:yellow">Feature Encoding is the process of converting categorical data (text labels/categories) into numerical data so that Machine Learning algorithms can work with it.

# In[7]:


l=['male','female','male','female','male','male','female']
import pandas as pd
result=pd.Series(l).map({'male':0,"female":1}).values
print(result)


# # 📘 Pandas Series and DataFrame Explained
# 
# 
# ---
# 
# # 1️⃣ What is a Pandas Series?
# 
# A **Series** is a 1D data structure consisting of a sequence of values.
# 
# Example:
# 
# | Index | Value |
# |---------|---------|
# | 0 | male |
# | 1 | female |
# | 2 | male |
# | 3 | female |
# 
# ### Creating a Series
# 
# ```python
# import pandas as pd
# 
# s = pd.Series(['male', 'female', 'male'])
# 
# print(s)
# ```
# 
# ### Output
# 
# ```text
# 0      male
# 1    female
# 2      male
# dtype: object
# ```
# 
# ### Understanding the Output
# 
# - Left side → **Index**
# - Right side → **Values**
# 
# ---
# 
# # 2️⃣ What is a DataFrame?
# 
# A **DataFrame** is a 2D data structure (or table) consisting of rows and columns.
# 
# ### Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     'Name': ['Ram', 'Shyam', 'Mohan'],
#     'Age': [20, 21, 22]
# })
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#     Name  Age
# 0    Ram   20
# 1  Shyam   21
# 2  Mohan   22
# ```
# 
# ### Understanding the Output
# 
# - Each column is a **Series**.
# - A DataFrame is basically **multiple Series combined together**.
# 
# ---
# 
# # 🔄 Relationship Between Series and DataFrame
# 
# ```text
# DataFrame
# 
# +-------+-----+
# | Name  | Age |
# +-------+-----+
# | Ram   | 20  |
# | Shyam | 21  |
# | Mohan | 22  |
# +-------+-----+
# 
# Name Column → Series
# Age Column  → Series
# ```
# 
# ---
# 
# # 3️⃣ Understanding Your Code
# 
# ```python
# l = ['male', 'female', 'male', 'female', 'male', 'male', 'female']
# 
# import pandas as pd
# 
# a = pd.Series(l).map({'male': 0, 'female': 1}).values
# 
# print(a)
# ```
# 
# Let's break it down step by step.
# 
# ---
# 
# # Step 1: Create a Python List
# 
# ```python
# l = ['male', 'female', 'male', 'female', 'male', 'male', 'female']
# ```
# 
# 
# ```python
# 
# ```
# 
# ---
# 
# # Step 2: Convert the List into a Series
# 
# ```python
# pd.Series(l)
# ```
# 
# ### Output
# 
# ```text
# 0      male
# 1    female
# 2      male
# 3    female
# 4      male
# 5      male
# 6    female
# dtype: object
# ```
# 
# ### Table Representation
# 
# | Index | Value |
# |---------|---------|
# | 0 | male |
# | 1 | female |
# | 2 | male |
# | 3 | female |
# | 4 | male |
# | 5 | male |
# | 6 | female |
# 
# ---
# 
# # Step 3: Apply Mapping Using `.map()`
# 
# ```python
# .map({'male': 0, 'female': 1})
# ```
# 
# The dictionary used for mapping is:
# 
# ```python
# {
#     'male': 0,
#     'female': 1
# }
# ```
# 
# ### What `.map()` Does
# 
# It replaces each value according to the dictionary.
# 
# | Original Value | New Value |
# |---------------|-----------|
# | male | 0 |
# | female | 1 |
# | male | 0 |
# | female | 1 |
# | male | 0 |
# | male | 0 |
# | female | 1 |
# 
# ### Resulting Series
# 
# ```text
# 0    0
# 1    1
# 2    0
# 3    1
# 4    0
# 5    0
# 6    1
# dtype: int64
# ```
# 
# ---
# 
# # Step 4: Extract Raw Values Using `.values`
# 
# 
# The result before `.values` is:
# 
# ```text
# 0    0
# 1    1
# 2    0
# 3    1
# 4    0
# 5    0
# 6    1
# dtype: int64
# ```
# 
# A Series contains:
# 
# - Values
# - Index
# - Data Type Information
# 
# Many Machine Learning algorithms only need the actual values.
# 
# So:
# 
# ```python
# .values
# ```
# 
# extracts only the data.
# 
# ### Result
# 
# ```python
# array([0, 1, 0, 1, 0, 0, 1])
# ```
# 
# This is a **NumPy Array**.
# 
# ---
# 
# 
# ---
# 
# # 🎯 Why Is This Used in Machine Learning?
# 
# Most Machine Learning algorithms cannot directly understand text values such as:
# 
# ```text
# male
# female
# ```
# 
# They require numerical values.
# 
# Therefore, we convert:
# 
# | Category | Encoded Value |
# |-----------|---------------|
# | male | 0 |
# | female | 1 |
# 
# ---
# 
# # ✅ Final Output
# 
# ```python
# array([0, 1, 0, 1, 0, 0, 1])
# ```
# 
# This numeric representation can now be used by Machine Learning algorithms.

# ### 📊 1D vs 2D Data Structures
# 
# - A **DataFrame** is a **2D data structure** because it consists of **rows and columns**. To access a value, we need **two coordinates**: a **row** and a **column**.
# 
# - A **List** and a **Pandas Series** are **1D data structures** because they store values in a **single sequence**. To access a value, only **one index** is required.

# In[3]:


from sklearn.preprocessing import OrdinalEncoder  
import pandas as pd                             
l = ['male', 'female', 'male', 'female', 'male', 'male', 'female']
# list of categorical values

obj = OrdinalEncoder(categories=[['male', 'female']])
# Create an OrdinalEncoder object and stores in obj
# The parameter categories=[['male','female']] tells:
# male   → 0
# female → 1

# If categories are NOT provided, OrdinalEncoder assigns numbers
# based on lexicographical (alphabetical) order, similar to LabelEncoder.
# Example:
# f comes before m
# female -> 0
# male   -> 1

a = obj.fit_transform(pd.DataFrame(l))
# pd.DataFrame(l) converts the list into a 2D DataFrame:
#       0(Default column name)
# 0    male
# 1  female
# 2    male
# 3  female
# 4    male
# 5    male
# 6  female
# fit() learns the mapping:
# male   -> 0
# female -> 1
# transform() replaces categories with their encoded values

print(a)


# | Encoding | Short Definition |
# |-----------|------------------|
# | **Label Encoding** | Assigns numbers to categories based on lexicographical (alphabetical) order. |
# | **One-Hot Encoding** | Creates separate binary columns for each category. |
# | **Binary Encoding** | First Label Encodes, then converts the labels to binary and stores each binary digit in a separate column. |
# | **Ordinal Encoding** | Assigns numbers to categories according to a user-defined order/ranking. |

# In[ ]:




