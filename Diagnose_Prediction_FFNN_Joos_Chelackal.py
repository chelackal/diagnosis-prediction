############################################################################

#          PREDICTION OF SECONDARY DIAGNOSES USING MULTILABEL FFNN
#          Authors: Alish Chelackal & Matthias Joos 
#          Project Work NN_AL, 25.5.2020

############################################################################

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import feature_column

# read  data 
data = pd.read_csv("C:/Users/joosm/Dropbox/Master ACLS/NN_DL/Project Work/diagnoses.csv",delimiter=',')
data = pd.DataFrame(data)

############################################################################
#########               DATA PREPROCESSING                        ##########
############################################################################

# delete rows where secondary diagnose (label) is identical to main diagnose
# to prevent that label does not contain the same information as input variables
data = data[data['Nebendiagnose'] != data['Hauptdiagnose']] 

# determine how many unique secondary diagnose label exist 
# for FFNN output layer 
outputlayer_size = data.Nebendiagnose.drop_duplicates().count()

# separate categorial variables
categ_variables = data.select_dtypes(include=[object]) 
categ_variables.pop('Nebendiagnose')

# one hot encoding secondary diagnoses (labels)
data['Nebendiagnose'] = pd.Categorical(data['Nebendiagnose'])
onehot_labels = pd.get_dummies(data['Nebendiagnose'], prefix = 'SD')

# separate numerical variables
num_variables = data[['Alter','MDC','PCCL']]

# concatenate preprocessed dataframes 
data = pd.concat([categ_variables, num_variables, onehot_labels], axis=1)

# split into train, validation and test samples
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)

############################################################################
#########      CREATE FEATURE COLUMNS FOR FFNN                    ##########
############################################################################

# Feature columns map the desired format type to each variable, before feeding the data into the FFNN
# with feature columns the transformation of the dataframe itself is not required

# function to build a pipeline on the dataset for the feature columns
def df_to_dataset(df, shuffle=True, batch_size=32):
  data = df.copy()
  labels = data.iloc[:,7:].values
  #labels = labels.astype('float32')
  input_data = data.iloc[:,0:7]
  ds = tf.data.Dataset.from_tensor_slices((dict(input_data), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(input_data))
  ds = ds.batch(batch_size)
  return ds

# define data pipelines for train, validation and test dataset
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# define format for different columns
# normalizing numerical features age, MDC and PCCL
num_colums = ['Alter','MDC','PCCL',]
# one-hot-encoding categorical features sex and partition
one_hot_columns =  ['Geschlecht','Partition']
# embedding encoding main diagnose and DRG
embedding_columns = ['Hauptdiagnose','DRG'] 

# further explanation regarding the medical terms:
# MDC = Major Diagnostic Category
# PCCL = Patient Clinical Complexity Level
# Partition O = surgical case rate 
# Partition M = medical case rate 
# Partition 0 = other case rate 
# DRG = Diagnose Related Group

# function for scaling (normalizing) numerical values
def get_scal(feature):
  def minmax(x):
    mini = train[feature].min()
    maxi = train[feature].max()
    return (x - mini)/(maxi-mini)
  return(minmax)

# create container for feature columns
feature_columns = []

# add numerical 
for header in num_colums:
  scal_input_fn = get_scal(header)
  feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn)) 

# add one-hot encoded 
for feature_name in one_hot_columns:
  vocabulary = data[feature_name].unique()
  cat_column = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  one_hot_columns = feature_column.indicator_column(cat_column)
  feature_columns.append(one_hot_columns)

# add embedding-encoded
for feature_name in embedding_columns:
  vocabulary = data[feature_name].unique()
  cat_column = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  embedding_column = feature_column.embedding_column(cat_column, dimension=8)
  feature_columns.append(embedding_column)
  
# create feature layer 
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

############################################################################
#########      TRAIN FFNN MODEL                                   ##########
############################################################################

# two hidden layers, batch normalisation, dropout layer to minimize the risk of overfitting
# sigmoid activation in the last layer calculates a probability value between 0 and 1 for 
# the occurennce of each secondary diagnose --> Multilabel Classification
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(15, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(32, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout(0.1),
  layers.Dense(outputlayer_size, activation='sigmoid')
])

# compile FFNN model
optimizer = optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

# fit FFFN model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# check accuracy of trained model on test dataset
loss, accuracy = model.evaluate(test_ds)
print('Accuracy:', accuracy)

############################################################################
#########     EXAMPLE PREDICTION                                  ##########
############################################################################

# create input samples for FFNN model prediction
pred_df = test.iloc[0:10,0:7].reset_index()
pred_ds = tf.data.Dataset.from_tensor_slices(dict(pred_df))
pred_ds = pred_ds.batch(1)                                              

# predict top five frequent secondary diagnose for input samples
pred = model.predict(pred_ds)
pred = pd.DataFrame(pred)
column_names = test.columns[7:]
pred.columns = column_names
c = ['1st Prob','2nd Prob','3rd Prob','4th Prob','5th Prob']
topfive_sd_df = (pred.apply(lambda x: pd.Series(x.nlargest(5).index, index=c), axis=1))

# check dataframe below to view top five secondary diagnose predictions 
# (input data --> a slice of the test set) 
predict_df = pred_df.join(topfive_sd_df)

# hint: compare top five predicted secondary diagnose directly with true values from dataset 
# (search by index in Excel file) as number of secondary diagnoses vary from patient to patient

####################################################################################
######    SECONDARY DIAGNOSE PREDICTION WITH USER-DEFINED INPUT VARIABLES    #######
####################################################################################

# with the function below, the user can define the input variables himself and the model
# predicts the respective top five secondary diagnoses
# required input variables: Geschlecht, Partition, Hauptdiagnose, DRG, Alter, MDC, PCCL

def user_input():
    input_sex = input("Enter person's sex [m or w]:  ")
    v1 = str(input_sex)
    input_partition = input("Enter partition [A or O or M or 0]:  ")
    v2 = str(input_partition)
    input_main_diagnose = input("Enter main diagnose [chose own value]:  ")
    v3 = str(input_main_diagnose)
    input_DRG = input("Enter DRG code [chose own value]:  ")
    v4 = str(input_DRG)
    input_age = input("Enter age [chose own value]:  ")
    v5 = int(input_age)
    input_MDC = input("Enter MDC code [chose own value]:  ")
    v6 = int(input_MDC)
    input_PCCL = input("Enter PCCL code [chose from range 0-4]:  ")
    v7 = int(input_PCCL)
    variable_input_list =v1,v2,v3,v4,v5,v6,v7
    variable_input = np.reshape(variable_input_list,(1,7))
    column_names = test.columns[:7]
    variable_input = pd.DataFrame(variable_input,columns = column_names)
    variable_input[['Alter','MDC','PCCL']] = variable_input[['Alter','MDC','PCCL']].apply(pd.to_numeric)
    user_pred_ds = tf.data.Dataset.from_tensor_slices(dict(variable_input))
    user_pred_ds = user_pred_ds.batch(1)                                              
    user_pred = model.predict(user_pred_ds)
    user_pred = pd.DataFrame(user_pred)
    column_names = test.columns[7:]
    user_pred.columns = column_names
    c = ['1st Prob','2nd Prob','3rd Prob','4th Prob','5th Prob']
    topfive_sd_df2 = (user_pred.apply(lambda x: pd.Series(x.nlargest(5).index, index=c), axis=1))
    user_predict_df = pred_df.join(topfive_sd_df2)
    user_predict_df = user_predict_df.iloc[0,:]
    print("")
    statement = print('predicted top five secondary diagnoses from user inputs',"\n", user_predict_df.iloc[8:])
    return statement
    
# uncomment below to test with own user input   
# print(user_input())
 
############################################################################
#########     VISUALIZATION OF FFNN LOSS AND ACCURACY TRAINING HISTORY   ###
############################################################################

# choose same range as used for fitting the model
epochs = range(10)
plt.title('Accuracy')
plt.plot(epochs,  history.history['accuracy'], color='blue', label='Train')
plt.plot(epochs, history.history['val_accuracy'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

_ = plt.figure()
plt.title('Loss')
plt.plot(epochs, history.history['loss'], color='blue', label='Train')
plt.plot(epochs, history.history['val_loss'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()