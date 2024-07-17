import os
from os.path import dirname as parent
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.train import train
from Applications.Poisoning.model import get_VGG_CIFAR10, get_VGG16_CIFAR100, get_VGG16_SVHN, get_RESNET50_SVHN, get_RESNET50_CIFAR10, get_RESNET50_CIFAR100, get_VGG19_CIFAR10, get_VGG19_CIFAR100, get_VGG19_SVHN
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10, SVHN, Cifar100
from Applications.Sharding.ensemble import train_models


def get_parser():
    parser = argparse.ArgumentParser("poison_models", description="Train poisoned models.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='poison_config.json', help="config file with parameters for this experiment")
    return parser


def train_poisoned(model_folder, poison_kwargs, train_kwargs, target_args=''):

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
            'Cifar100': lambda: get_VGG16_CIFAR100(dense_units=train_kwargs['model_size']),
            'SVHN': lambda: get_VGG16_SVHN(dense_units=train_kwargs['model_size'])
        },
         'VGG19': {
            'Cifar10': lambda: get_VGG19_CIFAR10(dense_units=train_kwargs['model_size']),
            'Cifar100': lambda: get_VGG19_CIFAR100(dense_units=train_kwargs['model_size']),
            'SVHN': lambda: get_VGG19_SVHN(dense_units=train_kwargs['model_size'])
        }
    }
    data = dataset_loaders[dataset]()
    model_init = model_initializers[modelname][dataset]
    
    
    (x_train, y_train), _, _ = data

    # inject label flips
    if 'sharding' in str(model_folder):
        injector_path = os.path.join(parent(model_folder), 'injector.pkl')
    else:
        injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle(injector_path)
    else:
        print(poison_kwargs)
        injector = LabelflipInjector(model_folder, **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    injector.save(injector_path)
    data = ((x_train, y_train), data[1], data[2])

    
    
    
    if 'sharding' in str(model_folder):
        n_shards = Config.from_json(os.path.join(model_folder, 'unlearn_config.json'))['n_shards']
        train_models(model_init, model_folder, data, n_shards, model_filename='poisoned_model.hdf5', **train_kwargs)
    else:
        train(model_init, model_folder, data, model_filename=dataset+"_"+modelname+'_poisoned_model.hdf5', target_args=target_args, **train_kwargs)


def main(model_folder, config_file, target_args=''):
    if 'sharding' in str(model_folder):
        poison_kwargs = Config.from_json(os.path.join(parent(model_folder), config_file))
        train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    else:
        poison_kwargs = Config.from_json(os.path.join(model_folder, config_file))
        train_kwargs = Config.from_json(os.path.join(model_folder, 'train_config.json'))
    train_poisoned(model_folder, poison_kwargs, train_kwargs, target_args=target_args)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
