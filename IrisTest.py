# import necessary libraries & modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & split the dataset
(x_train, y_train), (x_test, y_test) = tfds.load(
    'iris',
    split=['train', 'train'],
    batch_size=-1,
    as_supervised=True,
)

x_train = x_train[:120]
x_test = x_test[120:]
y_train = to_categorical(y_train[:120])
y_test = to_categorical(y_test[120:])

# declare hyperparameters
epoch = 50
batchSize = None
MLP_h1 = 128
CHN_h1 = 128
optimizer = 'adam'
loss = 'categorical_crossentropy'


# train MLP model
MLP_model = Sequential([
  Dense(MLP_h1, activation='relu'),
  Dense(3, activation="softmax")
])

MLP_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# train CHN model
CHN_model = Sequential([
  CHNLayer(CHN_h1, activation='relu'),
  Dense(3, activation="softmax")
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
plt.title("Iris Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["MLP", "CHN"])
plt.show()


# generate accuracy graph
plt.plot(MLP_History.history['accuracy'])
plt.plot(CHN_History.history['accuracy'])
plt.title("Iris Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["MLP", "CHN"])
plt.show()
