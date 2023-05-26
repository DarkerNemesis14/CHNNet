# import necessary libraries & modules
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & do slight pre-processing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=113
)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)


# declare hyperparameters
epoch = 50
batchSize = 128
MLP_h1 = 64
MLP_h2 = 64
CHN_h1 = 64
CHN_h2 = 64
optimizer = 'rmsprop'
loss = 'mse'


# train MLP model
MLP_model = Sequential([
  Dense(MLP_h1, input_dim = 13, activation="relu"),
  Dense(MLP_h2, activation="relu"),
  Dense(1, activation="linear")
])

MLP_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['mae'])

MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# train CHN model
CHN_model = Sequential([
  CHNLayer(CHN_h1, input_dim = 13, activation="relu"),
  CHNLayer(CHN_h2, activation="relu"),
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
plt.title("Boston Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["MLP", "CHN"])
plt.show()


# generate accuracy graph
plt.plot(MLP_History.history['mae'])
plt.plot(CHN_History.history['mae'])
plt.title("Boston MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend(["MLP", "CHN"])
plt.show()