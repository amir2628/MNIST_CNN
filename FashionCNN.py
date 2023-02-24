import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class FashionMNIST:
    
    def __init__(self):
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = fashion_mnist.load_data()
        self.classes = np.unique(self.train_Y)
        self.nClasses = len(self.classes)
        self.train_X, self.test_X = self.preprocess_images(self.train_X, self.test_X)
        self.train_Y_one_hot, self.test_Y_one_hot = self.one_hot_encoding(self.train_Y, self.test_Y)
        self.train_X, self.valid_X, self.train_label, self.valid_label = train_test_split(
            self.train_X, self.train_Y_one_hot, test_size=0.2, shuffle=True)
        
    def preprocess_images(self, train_X, test_X):
        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)
        train_X = train_X.astype('float32') / 255.
        test_X = test_X.astype('float32') / 255.
        return train_X, test_X

    def one_hot_encoding(self, train_Y, test_Y):
        train_Y_one_hot = to_categorical(train_Y)
        test_Y_one_hot = to_categorical(test_Y)
        return train_Y_one_hot, test_Y_one_hot

class FashionModel:
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_X, train_Y, valid_X, valid_Y, batch_size=64, epochs=10):
        history = self.model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_Y))
        return history

    def evaluate(self, test_X, test_Y):
        test_loss, test_acc = self.model.evaluate(test_X, test_Y, verbose=0)
        return test_loss, test_acc

    def predict(self, image):
        return np.argmax(self.model.predict(image))

    def plot_history(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Training and Validation Metrics')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    fashion_mnist = FashionMNIST()
    fashion_model = FashionModel()
    history = fashion_model.train(fashion_mnist.train_X, fashion_mnist.train_label, fashion_mnist.valid_X, fashion_mnist.valid_label, epochs=10)
    fashion_model.plot_history(history)
    test_loss, test_acc = fashion_model.evaluate(fashion_mnist.test_X, fashion_mnist.test_Y_one_hot)
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
    test_Y_pred = np.argmax(fashion_model.model.predict(fashion_mnist.test_X), axis=-1)
    print(classification_report(np.argmax(fashion_mnist.test_Y_one_hot, axis=1), test_Y_pred))

    # Show 10 random images from the testing dataset
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    axes = axes.ravel()
    for i in np.arange(0, 10):
        axes[i].imshow(fashion_mnist.test_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        axes[i].set_title(f"True: {fashion_mnist.classes[fashion_mnist.test_Y[i]]}\nPredict: {fashion_mnist.classes[test_Y_pred[i]]}")
        axes[i].axis('off')
        plt.subplots_adjust(wspace=0.5)
    plt.show()

    # Show 10 random misclassified images from the testing dataset
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    axes = axes.ravel()
    misclassified_idx = np.where(test_Y_pred != fashion_mnist.test_Y)[0]
    for i in np.arange(0, 10):
        idx = misclassified_idx[i]
        axes[i].imshow(fashion_mnist.test_X[idx].reshape(28, 28), cmap=plt.get_cmap('gray'))
        axes[i].set_title(f"True: {fashion_mnist.classes[fashion_mnist.test_Y[idx]]}\nPredict: {fashion_mnist.classes[test_Y_pred[idx]]}")
        axes[i].axis('off')
        plt.subplots_adjust(wspace=0.5)
    plt.show()


