#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# ## Features_dataset

# In[228]:


df1 = pd.read_csv("dense_train.csv")


# In[229]:


new_name = {'Unnamed: 0' : 'new_label'}


# In[230]:


df1.rename(columns=new_name,inplace=True)


# In[231]:


unique_values = df1['new_label'].unique()


# In[232]:


unique_values


# In[239]:


df1.columns


# In[ ]:


#df1.drop(columns='new_label',inplace=True)


# ## Label dataset

# In[240]:


df2 = pd.read_csv("labels_train.csv")


# In[241]:


new_name = {'Unnamed: 0' : 'new_label'}


# In[242]:


df2.rename(columns=new_name,inplace=True)


# In[243]:


df2


# ## Removing coorelated features

# In[234]:


correlation_matrix = df1.corr()
correlation_threshold = 0.7  # Set your correlation threshold value
correlated_features = set()
for i in range(1,len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
X = df1.drop(columns=correlated_features)


# In[235]:


X.columns


# ## Concaetenating labels and feature dataset

# In[244]:


merged_df = pd.merge(X, df2, on='new_label', how='inner')


# In[245]:


merged_df


# In[246]:


merged_df.fillna(0,inplace=True)


# In[250]:


merged_df.drop(columns='new_label',inplace=True)


# ## Predicting NR.AhR

# In[251]:


columns_to_remove = ['NR.AR','NR.AR.LBD','NR.Aromatase','NR.ER','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_ahr = merged_df.drop(columns=columns_to_remove)


# In[252]:


df_ahr 


# In[253]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


# In[254]:


X = df_ahr.drop(columns=['NR.AhR'])  # FeaturesX = df_ahr.drop(columns=['new_label','NR.AhR'])  # Features
y = df_ahr['NR.AhR']  # Target variable


# In[255]:


y


# In[256]:


X


# In[271]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.4,random_state=42)


# In[272]:


Y_train


# In[273]:


from sklearn.ensemble import RandomForestClassifier


# In[278]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=150, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[279]:


predictions = rf_classifier.predict(X_test)


# In[280]:


predictions


# In[281]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# ## Predicting  NR.AR

# In[282]:


columns_to_remove = ['NR.AhR','NR.AR.LBD','NR.Aromatase','NR.ER','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_ar = merged_df.drop(columns=columns_to_remove)


# In[283]:


df_ar


# In[284]:


X = df_ar.drop(columns=['NR.AR'])
y = df_ar['NR.AR']  # Target variable


# In[285]:


y


# In[286]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[287]:


from sklearn.ensemble import RandomForestClassifier


# In[288]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[290]:


predictions = rf_classifier.predict(X_test)


# In[291]:


predictions


# In[293]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting NR.AR.LBD

# In[304]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_ar_lbd = merged_df.drop(columns=columns_to_remove)


# In[305]:


X = df_ar_lbd.drop(columns=['NR.AR.LBD'])
y = df_ar_lbd['NR.AR.LBD']  # Target variable


# In[306]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[307]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[308]:


predictions = rf_classifier.predict(X_test)


# In[309]:


predictions


# In[310]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting NR.Aromatase

# In[311]:


columns_to_remove = ['NR.AhR','NR.AR','NR.AR.LBD','NR.ER','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_aromatase = merged_df.drop(columns=columns_to_remove)


# In[312]:


X = df_aromatase.drop(columns=['NR.Aromatase'])
y = df_aromatase['NR.Aromatase']  # Target variable


# In[313]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[314]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[315]:


predictions = rf_classifier.predict(X_test)


# In[316]:


predictions


# In[317]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting NR.ER

# In[318]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.AR.LBD','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_er= merged_df.drop(columns=columns_to_remove)


# In[319]:


X = df_er.drop(columns=['NR.ER'])
y = df_er['NR.ER']  # Target variable


# In[320]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[321]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[322]:


predictions = rf_classifier.predict(X_test)


# In[323]:


predictions


# In[324]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# 

# ## Predicting NR.ER.LBD

# In[325]:


columns_to_remove = ['NR.AhR','NR.AR','NR.AR.LBD','NR.Aromatase','NR.ER','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_er_lbd = merged_df.drop(columns=columns_to_remove)


# In[327]:


X = df_er_lbd.drop(columns=['NR.ER.LBD'])
y = df_er_lbd['NR.ER.LBD']  # Target variable


# In[328]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[329]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[330]:


predictions = rf_classifier.predict(X_test)


# In[331]:


predictions


# In[332]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting NR.PPAR.gamma

# In[335]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[336]:


X = df_nr_ppar.drop(columns=['NR.PPAR.gamma'])
y = df_nr_ppar['NR.PPAR.gamma']  # Target variable


# In[337]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[338]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[339]:


predictions = rf_classifier.predict(X_test)


# In[340]:


predictions


# In[341]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting SR.ARE

# In[342]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','NR.PPAR.gamma','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[343]:


X = df_nr_ppar.drop(columns=['SR.ARE'])
y = df_nr_ppar['SR.ARE']  # Target variable


# In[344]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[345]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[346]:


predictions = rf_classifier.predict(X_test)


# In[347]:


predictions


# In[348]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting SR.ATAD5

# In[349]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','SR.ARE','NR.PPAR.gamma','SR.HSE','SR.MMP','SR.p53']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[350]:


X = df_nr_ppar.drop(columns=['SR.ATAD5'])
y = df_nr_ppar['SR.ATAD5']  # Target variable


# In[351]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[352]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[353]:


predictions = rf_classifier.predict(X_test)


# In[354]:


predictions


# In[355]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# In[ ]:





# ## Predicting SR.HSE

# In[356]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','SR.ARE','SR.ATAD5','NR.PPAR.gamma','SR.MMP','SR.p53']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[357]:


X = df_nr_ppar.drop(columns=['SR.HSE'])
y = df_nr_ppar['SR.HSE']  # Target variable


# In[358]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[359]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[360]:


predictions = rf_classifier.predict(X_test)


# In[361]:


predictions


# In[362]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# In[ ]:





# ## Predicting SR.MMP

# In[363]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','SR.ARE','SR.ATAD5','SR.HSE','NR.PPAR.gamma','SR.p53']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[364]:


X = df_nr_ppar.drop(columns=['SR.MMP'])
y = df_nr_ppar['SR.MMP']  # Target variable


# In[365]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[366]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[367]:


predictions = rf_classifier.predict(X_test)


# In[368]:


predictions


# In[369]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:





# ## Predicting SR.p53

# In[370]:


columns_to_remove = ['NR.AhR','NR.AR','NR.Aromatase','NR.ER','NR.ER.LBD','NR.AR.LBD','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','NR.PPAR.gamma']
df_nr_ppar = merged_df.drop(columns=columns_to_remove)


# In[371]:


X = df_nr_ppar.drop(columns=['SR.p53'])
y = df_nr_ppar['SR.p53']  # Target variable


# In[372]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[373]:


# Create a RandomForestClassifier instance with desired parameters
# Example: n_estimators is the number of decision trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to your training data
rf_classifier.fit(X_train, Y_train)


# In[374]:


predictions = rf_classifier.predict(X_test)


# In[375]:


predictions


# In[376]:


accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy*100)


# In[ ]:




