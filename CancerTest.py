# import necessary libraries & modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & do slight pre-processing
dataset = load_breast_cancer()
data = pd.DataFrame(data = dataset.data,
                       columns = dataset.feature_names)

x = data.values
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")


# declare hyperparameters
epoch = 1
batchSize = 128
MLP_h1 = 64
MLP_h2 = 64
CHN_h1 = 64
CHN_h2 = 64
optimizer = 'adam'
loss = 'binary_crossentropy'


# train MLP model
MLP_model = Sequential([
    Dense(MLP_h1, activation='relu'),
    Dense(MLP_h2, activation='relu'),
    Dense(1, activation='sigmoid')
])

MLP_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# train CHN model
CHN_model = Sequential([
    CHNLayer(CHN_h1, activation='relu'),
    CHNLayer(CHN_h2, activation='relu'),
    Dense(1, activation='sigmoid')
])

CHN_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

CHN_History = CHN_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# print test results of MLP model
MLP_results = MLP_model.evaluate(x_test, y_test)
print(f'MLP Loss: {MLP_results[0]}\nMLP Accuracy: {MLP_results[1]}')

MLP_model.summary()


# print test results of CHN model
CHN_results = CHN_model.evaluate(x_test, y_test)
print(f'CHN Loss: {CHN_results[0]}\nCHN Accuracy: {CHN_results[1]}')

CHN_model.summary()


# generate loss graph
plt.plot(MLP_History.history['loss'])
plt.plot(CHN_History.history['loss'])
plt.title("Cancer Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["MLP", "CHN"])
plt.show()


# generate accuracy graph
plt.plot(MLP_History.history['accuracy'])
plt.plot(CHN_History.history['accuracy'])
plt.title("Cancer Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["MLP", "CHN"])
plt.show()