# import necessary libraries & modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot as plt


# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & do slight pre-processing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")


# declare hyperparameters
epoch = 10
batchSize = 32
MLP_h1 = 256
MLP_h2 = 256
MLP_h3 = 256
CHN_h1 = 256
CHN_h2 = 256
CHN_h3 = 256
optimizer = SGD(lr=0.001, momentum=0.9)
loss = SparseCategoricalCrossentropy(from_logits=True)


# train MLP model
MLP_model = Sequential([
    Flatten(input_shape=[28,28]),
    Dense(MLP_h1, activation='relu'),
    Dense(MLP_h2, activation='relu'),
    Dense(MLP_h3, activation='relu'),
    Dense(10, activation='softmax')
])

MLP_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)


# train CHN model
CHN_model = Sequential([
    Flatten(input_shape=[28,28]),
    CHNLayer(CHN_h1, activation='relu'),
    CHNLayer(CHN_h2, activation='relu'),
    CHNLayer(CHN_h3, activation='relu'),
    Dense(10, activation='softmax')
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
plt.title("Fashion MNIST Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["MLP", "CHN"])
plt.show()


# generate accuracy graph
plt.plot(MLP_History.history['accuracy'])
plt.plot(CHN_History.history['accuracy'])
plt.title("Fashion MNIST Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["MLP", "CHN"])
plt.show()