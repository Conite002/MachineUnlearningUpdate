import os
import argparse

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report
from pathlib import Path

from util import TrainingResult, measure_time
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_GTSRB, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_CIFAR100, extractfeatures_RESNET50, extractfeatures_VGG16, classifier_RESNET50, classifier_VGG16
from Applications.Poisoning.model import get_RESNET50_CIFAR10, get_RESNET50_GTSRB, get_RESNET50_MNIST, get_RESNET50_FASHION, get_RESNET50_SVHN, get_RESNET50_CIFAR100
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score
import numpy as np



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
def evaluate(model_folder, dataset, modelname, type):
    if dataset == "Cifar10":
        data = Cifar10.load()
        
        if modelname == "VGG16":
            model = get_VGG16_CIFAR10()
        elif modelname == "RESNET50":
            model = get_RESNET50_CIFAR10()
        elif modelname == "extractfeatures_VGG16":
            model = extractfeatures_VGG16()
        elif modelname == "extractfeatures_RESNET50":
            model = extractfeatures_RESNET50()
        elif modelname == "classifier_VGG16":
            model = classifier_VGG16()
        elif modelname == "classifier_RESNET50":
            model = classifier_RESNET50()
        else:
            raise ValueError(f"Unknown modelname: {modelname}")
    elif dataset == "Mnist":
        data = Mnist.load()
        if modelname == "RESNET50":
            model = get_RESNET50_MNIST()
        elif modelname == "VGG16":
            model = get_VGG16_MNIST()
        elif modelname == "extractfeatures_VGG16":
            model = extractfeatures_VGG16()
        elif modelname == "extractfeatures_RESNET50":
            model = extractfeatures_RESNET50()
        elif modelname == "classifier_VGG16":
            model = classifier_VGG16()
        elif modelname == "classifier_RESNET50":
            model = classifier_RESNET50()
        else:
            raise ValueError(f"Unknown modelname: {modelname}")



    elif dataset == "FashionMnist":
        data = FashionMnist.load()
        if modelname == "VGG16":
            model = get_VGG16_FASHION()
        elif modelname == "RESNET50":
            model = get_RESNET50_FASHION()
        elif modelname == "extractfeatures_VGG16":
            model = extractfeatures_VGG16()
        elif modelname == "extractfeatures_RESNET50":
            model = extractfeatures_RESNET50()
        elif modelname == "classifier_VGG16":
            model = classifier_VGG16()
        elif modelname == "classifier_RESNET50":
            model = classifier_RESNET50()
        else:
            raise ValueError(f"Unknown modelname: {modelname}")
        
    elif dataset == "SVHN":
        data = SVHN.load()
        if modelname == "VGG16":
            model = get_VGG16_SVHN()
        elif modelname == "RESNET50":
            model = get_RESNET50_SVHN()
        elif modelname == "extractfeatures_VGG16":
            model = extractfeatures_VGG16()
        elif modelname == "extractfeatures_RESNET50":
            model = extractfeatures_RESNET50()
        elif modelname == "classifier_VGG16":
            model = classifier_VGG16()
        elif modelname == "classifier_RESNET50":
            model = classifier_RESNET50()
        else:
            raise ValueError(f"Unknown modelname: {modelname}")
        
    elif dataset == "Cifar100":
        data = Cifar100.load()
        if modelname == "VGG16":
            model = get_VGG16_CIFAR100()
        elif modelname == "RESNET50":
            model = get_RESNET50_CIFAR100()
        elif modelname == "extractfeatures_VGG16":
            model = extractfeatures_VGG16()
        elif modelname == "extractfeatures_RESNET50":
            model = extractfeatures_RESNET50()
        elif modelname == "classifier_VGG16":
            model = classifier_VGG16()
        elif modelname == "classifier_RESNET50":
            model = classifier_RESNET50()


    else:
        raise ValueError("Invalid dataset name")
        data = None
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = data

    model_folder = Path(model_folder)
    weights_folder = model_folder / f"{dataset}_{modelname}_{type}_model.hdf5"
    if are_weights_loaded(model, weights_folder):

        print(f"Model weights loaded successfully from {weights_folder}.")
    else:
        print("Failed to load model weights.")


    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy {dataset} model: {accuracy}")
    
    return accuracy

    
def main(model_folder, dataset, modelname, type):
    evaluate(model_folder, dataset, modelname, type)


def evaluate_and_save_results(model_folder, dataset, modelname, clean_folder, poisoned_folder, first_update_folder, second_update_folder, csv_dir):
   
    poisoned_folder = Path(poisoned_folder)
    first_update_folder = Path(first_update_folder)
    second_update_folder = Path(second_update_folder)
    clean_folder = Path(clean_folder)

    poisoned_weights = poisoned_folder / f"{dataset}_{modelname}_poisoned_model.hdf5"
    first_update_weights = first_update_folder / f"{dataset}_{modelname}_repaired_model.hdf5"
    second_update_weights = second_update_folder / f"{dataset}_{modelname}_repaired_model.hdf5"
    clean_weights = clean_folder / f"{dataset}_{modelname}_best_model.hdf5"

    clean_accuracy = evaluate(model_folder=clean_folder ,dataset=dataset, modelname=modelname, type="best")
    poisoned_accuracy = evaluate(model_folder=poisoned_folder, dataset=dataset, modelname=modelname, type="poisoned")
    first_update_accuracy = evaluate(model_folder=first_update_folder, dataset=dataset, modelname=modelname, type="repaired")
    second_update_accuracy = evaluate(model_folder=second_update_folder, dataset=dataset, modelname=modelname, type="repaired")


    results = pd.DataFrame({
        "Model": [modelname],
        "Dataset": [dataset],
        "Clean": [clean_accuracy],
        "Poisoned": [poisoned_accuracy],
        "First Update": [first_update_accuracy],
        "Second Update": [second_update_accuracy]
    })

    df = pd.DataFrame(results)
    cvs_file = os.path.join(csv_dir, f"{dataset}_{modelname}_results.csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    else:
        df.to_csv(cvs_file, mode='a', header=False)

    print("Results saved successfully.")
    print(df)

    


if __name__ == '__main__':
    main()