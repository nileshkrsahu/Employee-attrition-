import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1 = pd.read_csv(r'train.csv')
dataset=dataset1.drop(['Employee_ID','Hometown','VAR1','VAR2','VAR3','VAR4','VAR5','VAR6','VAR7'],axis=1)

gender=dataset.iloc[:,0]
age=dataset.iloc[:,1:3]
candb=dataset.iloc[:,3:6]
numa=dataset.iloc[:,6:9]
numb=dataset.iloc[:,9:12]
comp=dataset.iloc[:,12]
bal=dataset.iloc[:,13].values
lop=dataset.iloc[:,14]

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer = imputer.fit(age)
age = imputer.transform(age)

imputer = imputer.fit(numa)
numa = imputer.transform(numa)

imputer = imputer.fit(numb)
numb = imputer.transform(numb)

bal=bal.reshape(-1,1)
imputer = imputer.fit(bal)
bal = imputer.transform(bal)

gender=pd.get_dummies(gender)
candb=pd.get_dummies(candb)
comp=pd.get_dummies(comp)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
age = sc_X.fit_transform(age)
numa = sc_X.fit_transform(numa)

final_dataset=np.column_stack((gender,age,candb,numa,numb,comp,bal,lop))
X = final_dataset[:, :-1]
y = final_dataset[:, 34]

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[34])])
model.compile(optimizer='adam', loss=['mean_absolute_error'])
history=model.fit(X, y, validation_split=0.33, batch_size=5,epochs=50)
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('mean_absolute_error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper right')
plt.show()

model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print("Saved Model to disk")