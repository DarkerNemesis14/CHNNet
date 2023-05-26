# import necessary libraries & modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & do slight pre-processing
dataset = pd.read_csv('abalone.csv')

Gender = dataset.pop('Sex')
dataset['M'] = (Gender == 'M')*1.0
dataset['F'] = (Gender == 'F')*1.0
dataset['I'] = (Gender == 'I')*1.0

dataset = dataset[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight','Shell weight','M','F','I','Rings']]

x=dataset.iloc[:,0:10]
y=dataset.iloc[:,10].values
 
scalar= MinMaxScaler()
x= scalar.fit_transform(x)
y= y.reshape(-1,1)
y=scalar.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# declare hyperparameters
epoch = 20
batchSize = 256
MLP_h1 = 8
CHN_h1 = 8
optimizer = 'adam'
loss = 'mse'


# train MLP model
MLP_model = Sequential([
  Dense(MLP_h1, input_dim = 10, activation="relu"),
  Dense(1, activation="linear")
])

MLP_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['mae'])

MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# train CHN model
CHN_model = Sequential([
  CHNLayer(CHN_h1, input_dim = 10, activation="relu"),
  Dense(1, activation="linear")
])

CHN_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['mae'])

CHN_History = CHN_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# print test results of MLP model
MLP_results = MLP_model.evaluate(x_test, y_test)
print(f'MLP Loss: {MLP_results[0]}')

MLP_model.summary()


# print test results of CHN model
CHN_results = CHN_model.evaluate(x_test, y_test)
print(f'CHN Loss: {CHN_results[0]}')

CHN_model.summary()


# generate loss graph
plt.plot(MLP_History.history['loss'])
plt.plot(CHN_History.history['loss'])
plt.title("Abalone Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["MLP", "CHN"])
plt.show()


# generate accuracy graph
plt.plot(MLP_History.history['mae'])
plt.plot(CHN_History.history['mae'])
plt.title("Abalone MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend(["MLP", "CHN"])
plt.show()
