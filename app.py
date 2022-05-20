import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('gpascore.csv') 
data = data.dropna()

y_data = data['admit'].values

x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])






model = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),     
])

model.compile(optimizer='adam' , loss='binary_crossentropy', metrics= ['accuracy'])

model.fit(np.array(x_data), np.array(y_data), epochs=100)

prediction = model.predict(np.array([[750, 4, 3], [200, 2.3, 1]]))
print(prediction)
