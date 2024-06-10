import os
import argparse

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report

from util import TrainingResult, measure_time
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_GTSRB, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_CIFAR100
from Applications.Poisoning.model import get_RESNET50_CIFAR10, get_RESNET50_GTSRB, get_RESNET50_MNIST, get_RESNET50_FASHION, get_RESNET50_SVHN, get_RESNET50_CIFAR100
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD


def are_weights_loaded(model, weights_path):
    initial_weights = model.get_weights()
    model.load_weights(weights_path)  # Load new weights
    loaded_weights = model.get_weights()  # Get the loaded weights
    
    # Compare initial weights with loaded weights
    for initial, loaded in zip(initial_weights, loaded_weights):
        if not np.array_equal(initial, loaded):
            return True  # Weights have changed
    return False 


# Evaluate the model
def evaluate(dataset, modeltype, poisoned_weights):
    if dataset == "Cifar10":
        data = Cifar10.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_CIFAR10()
        else:
            model = get_VGG16_CIFAR10()
    if dataset == "Mnist":
        data = Mnist.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_MNIST()
        else:
            model = get_VGG16_MNIST()

    if dataset == "FashionMnist":
        data = FashionMnist.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_FASHION()
        else:
            model = get_VGG16_FASHION()

    if dataset == "SVHN":
        data = SVHN.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_SVHN()
        else:
            model = get_VGG16_SVHN()

    if dataset == "GTSRB":
        data = GTSRB.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_GTSRB()
        else:
            model = get_VGG16_GTSRB()

    if dataset == "Cifar100":
        data = Cifar100.load()
        if modeltype == "RESNET50":
            model = get_RESNET50_CIFAR100()
        else:
            model = get_VGG16_CIFAR100()
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = data

    if are_weights_loaded(model, poisoned_weights):
        print("Model weights loaded successfully.")
    else:
        print("Failed to load model weights.")


    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)
    df_report = pd.DataFrame(report).transpose()
    csv_path = os.path.join(poisoned_weights, "report.csv")
    
    return csv_path

    
def main(dataset, modeltype, poisoned_weights):
    evaluate(dataset, modeltype, poisoned_weights)

if __name__ == '__main__':
    main()