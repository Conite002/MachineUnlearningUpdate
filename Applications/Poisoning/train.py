import os
import argparse

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report

from util import TrainingResult, measure_time
from Applications.Poisoning.model import get_VGG_CIFAR10, get_VGG_CIFAR10, get_VGG16_SVHN, get_RESNET50_SVHN, get_RESNET50_CIFAR10, get_RESNET50_CIFAR100
from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.dataset import Cifar10, SVHN, Cifar100


def train(model_init, model_folder, data, epochs, batch_size, target_args='', model_filename='best_model.hdf5', **kwargs):
    os.makedirs(model_folder, exist_ok=True)
    
    print(f"model_filename {model_filename}")
    model_save_path = os.path.join(model_folder, model_filename)
    modelname, dataset, target = target_args.split('_', 2) 
    
    if os.path.exists(model_save_path):
        result = TrainingResult(model_folder, dataset, modelname)
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = data
        model = model_init()
        model.load_weights(model_save_path)
        # calculate test metrics on final model
        y_test_hat = np.argmax(model.predict(x_test), axis=1)
        test_loss = model.evaluate(x_test, y_test, batch_size=1000, verbose=0)[0]
        report = classification_report(np.argmax(y_test, axis=1), y_test_hat, digits=4, output_dict=True)
        report['test_loss'] = test_loss
        result.update(report)
        result.save()

        return model_save_path
       
    csv_save_path = os.path.join(model_folder, 'train_log.csv')
    result = TrainingResult(model_folder, dataset, modelname)

    (x_train, y_train), (x_test, y_test), _ = data
    model = model_init()

    metric_for_min = 'loss'
    loss_ckpt = ModelCheckpoint(model_save_path, monitor=metric_for_min, save_best_only=True,
                                save_weights_only=True)
    csv_logger = CSVLogger(csv_save_path)
    callbacks = [loss_ckpt, csv_logger]

    with measure_time() as t:
        hist = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_test, y_test), verbose=1,
                            callbacks=callbacks).history
        training_time = t()
    best_loss = np.min(hist[metric_for_min]) if metric_for_min in hist else np.inf
    best_loss_epoch = np.argmin(hist[metric_for_min]) + 1 if metric_for_min in hist else 0
    print('Best model has test loss {} after {} epochs'.format(best_loss, best_loss_epoch))
    best_model = model_init()
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
    return model_save_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str)
    return parser


def main(model_folder, target_args='', model_filename=''):
    train_conf = os.path.join(model_folder, 'train_config.json')
    train_kwargs = Config.from_json(train_conf)
    modelname, dataset, target = target_args.split('_', 2)

    # Dictionaries for dataset loading and model initialization
    dataset_loaders = {
        'Cifar10': Cifar10.load,
        'Cifar100': Cifar100.load,
        'SVHN': SVHN.load
    }

    model_initializers = {
        'RESNET50': {
            'Cifar10': lambda: get_RESNET50_CIFAR10(dense_units=train_kwargs['model_size']),
            'Cifar100': lambda: get_RESNET50_CIFAR100(dense_units=train_kwargs['model_size']),
            'SVHN': lambda: get_RESNET50_SVHN(dense_units=train_kwargs['model_size'])
        },
        'VGG16': {
            'Cifar10': lambda: get_VGG_CIFAR10(dense_units=train_kwargs['model_size']),
            'Cifar100': lambda: get_VGG_CIFAR100(dense_units=train_kwargs['model_size']),
            'SVHN': lambda: get_VGG16_SVHN(dense_units=train_kwargs['model_size'])
        }
    }
    data = dataset_loaders[dataset]()
    model_init = model_initializers[modelname][dataset]
    train(model_init, model_folder, data, target_args=target_args,model_filename=model_filename,  **train_kwargs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
