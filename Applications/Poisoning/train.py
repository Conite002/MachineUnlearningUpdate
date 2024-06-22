import os
import argparse

import numpy as np
from sklearn.metrics import classification_report

from util import TrainingResult, measure_time
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_GTSRB, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_CIFAR100
from Applications.Poisoning.model import get_RESNET50_CIFAR10, get_RESNET50_GTSRB, get_RESNET50_MNIST, get_RESNET50_FASHION, get_RESNET50_SVHN, get_RESNET50_CIFAR100
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD


# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 20:
        lr = lr / 10
    return lr


def freeze_layers(model, num_layers_to_freeze):
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze the last num_layers_to_freeze layers
    if num_layers_to_freeze > 0:
        for layer in model.layers[-num_layers_to_freeze:]:
            layer.trainable = True

def train(dataset, modeltype, model_init, model_folder, data, epochs, batch_size, model_filename='best_model.hdf5', classes=10, unfreeze_layers_steps=None, **kwargs):
    os.makedirs(model_folder, exist_ok=True)
    model_save_path = os.path.join(model_folder, model_filename)
    if os.path.exists(model_save_path):
        return model_save_path
    

    csv_save_path = os.path.join(model_folder, 'train_log.csv')
    result = TrainingResult(model_folder, dataset, modeltype)

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = data
    model = model_init()

    metric_for_min = 'loss'
    loss_ckpt = ModelCheckpoint(model_save_path, monitor=metric_for_min, save_best_only=True,
                                save_weights_only=True)
    csv_logger = CSVLogger(csv_save_path)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    callbacks = [loss_ckpt, csv_logger, early_stopping, LearningRateScheduler(lr_schedule, verbose=1)]

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    # if os.path.exists(model_save_path):
    #     #finetuninig
    #     if unfreeze_layers_steps is None:
    #         unfreeze_layers_steps = [len(model.layers)]
            
    #     for num_layers_to_freeze in unfreeze_layers_steps:
    #         freeze_layers(model, num_layers_to_freeze)
    #         opt = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    #         model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
                          

    with measure_time() as t:
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_test, y_test), verbose=1,
                            callbacks=callbacks).history
        training_time = t()
    best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
    best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
    print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
    best_model = model_init()

    # Test if weights are loaded correctly
    if are_weights_loaded(best_model, model_save_path):
        print("Model weights loaded successfully.")
    else:
        print("Failed to load model weights.")
    best_model.load_weights(model_save_path)

    # calculate test metrics on final model
    y_test_hat = np.argmax(best_model.predict(x_test), axis=1)
    test_loss = best_model.evaluate(x_test, y_test, batch_size=1000, verbose=0)[0]
    report = classification_report(np.argmax(y_test, axis=1), y_test_hat, digits=4, output_dict=True)
    report['train_loss'] = best_loss
    report['test_loss'] = test_loss
    report['epochs_for_min'] = int(best_loss_epoch)  # json does not like numpy ints
    report['time'] = training_time
    result.update(report)
    result.save()

    # print evaluation metrics
    # print("Model evaluation metrics: ", model.metrics_names)
    # print error metrics
    # Evaluate with test data
    results = best_model.evaluate(x_test, y_test, batch_size=100)
    print('test loss, test acc:', results)

    return model_save_path

def are_weights_loaded(model, weights_path):
    initial_weights = model.get_weights()  # Save initial weights
    model.load_weights(weights_path)  # Load new weights
    loaded_weights = model.get_weights()  # Get the loaded weights
    
    # Compare initial weights with loaded weights
    for initial, loaded in zip(initial_weights, loaded_weights):
        if not np.array_equal(initial, loaded):
            return True  # Weights have changed
    return False  # Weights have not changed


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str)
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['Cifar10', 'Cifar100', 'Mnist', 'FashionMnist', 'SVHN', 'GTSRB'])
    parser.add_argument('--modeltype', type=str, default='RESNET50', choices=['RESNET50', 'VGG16'])
    parser.add_argument('--classes', type=int, default=10)
    return parser

def main(model_folder, dataset="Cifar10", modeltype="RESNET50", classes=10):
    train_conf = os.path.join(model_folder, 'train_config.json')
    train_kwargs = Config.from_json(train_conf)

    if dataset == "Cifar10":
        data = Cifar10.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_CIFAR10(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_CIFAR10(dense_units=train_kwargs['model_size'])
    if dataset == "Mnist":
        data = Mnist.load()
        classes = 10
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_MNIST(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_MNIST(dense_units=train_kwargs['model_size'])
    
    if dataset == "FashionMnist":
        data = FashionMnist.load()
        classes = 10
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_FASHION(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_FASHION(dense_units=train_kwargs['model_size'])
    
    if dataset == "SVHN":
        data = SVHN.load()
        classes = 10
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_SVHN(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_SVHN(dense_units=train_kwargs['model_size'])

    if dataset == "GTSRB":
        data = GTSRB.load()
        classes = 10
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_GTSRB(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_GTSRB(dense_units=train_kwargs['model_size'])
        
    if dataset == "Cifar100":
        data = Cifar100.load()
        classes = 100
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_CIFAR100(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_CIFAR100(dense_units=train_kwargs['model_size'])
            
    train(dataset, modeltype, model_init, model_folder, data, **train_kwargs, model_filename=dataset  +"_"+modeltype+ '_best_model.hdf5', classes=classes)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
