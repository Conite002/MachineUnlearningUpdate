import os
from os.path import dirname as parent
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.train import train
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_GTSRB, get_VGG16_CIFAR100
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from Applications.Sharding.ensemble import train_models


def get_parser():
    parser = argparse.ArgumentParser("poison_models", description="Train poisoned models.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='poison_config.json', help="config file with parameters for this experiment")
    return parser


def train_poisoned(model_folder, poison_kwargs, train_kwargs, dataset='cifar10', modeltype='VGG16', classes=10):
    
    if dataset == 'Cifar10':
        data = Cifar10.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG_CIFAR10(classes)
        else:
            model_init = lambda: get_RESNET50_CIFAR10(classes)
    if dataset == 'Mnist':
        data = Mnist.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG16_MNIST(classes)
        else:
            model_init = lambda: get_RESNET50_MNIST(classes)

    if dataset == 'FashionMnist':
        data = FashionMnist.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG16_FASHION(classes)
        else:
            model_init = lambda: get_RESNET50_FASHION(classes)
    
    if dataset == 'SVHN':
        data = SVHN.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG16_SVHN(classes)
        else:
            model_init = lambda: get_RESNET50_SVHN(classes)

    if dataset == 'GTSRB':
        data = GTSRB.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG16_GTSRB(classes)
        else:
            model_init = lambda: get_RESNET50_GTSRB(classes)

    if dataset == 'Cifar100':
        data = Cifar100.load()
        if modeltype == 'VGG16':
            model_init = lambda: get_VGG16_CIFAR100(classes)
        else:
            model_init = lambda: get_RESNET50_CIFAR100(classes)

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

    model_init = lambda: get_VGG_CIFAR10(dense_units=train_kwargs['model_size'])
    if 'sharding' in str(model_folder):
        n_shards = Config.from_json(os.path.join(model_folder, 'unlearn_config.json'))['n_shards']
        train_models(model_init, model_folder, data, n_shards, model_filename=dataset+"_"+modeltype+'_poisoned_model.hdf5', **train_kwargs)
    else:
        train(model_init, model_folder, data, model_filename=dataset+"_"+modeltype+'_poisoned_model.hdf5', **train_kwargs)


def main(model_folder, config_file, dataset='cifar10', modeltype='VGG16', classes=10):
    if 'sharding' in str(model_folder):
        poison_kwargs = Config.from_json(os.path.join(parent(model_folder), config_file))
        train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    else:
        poison_kwargs = Config.from_json(os.path.join(model_folder, config_file))
        train_kwargs = Config.from_json(os.path.join(model_folder, 'train_config.json'))
    train_poisoned(model_folder, poison_kwargs, train_kwargs, dataset, modeltype, classes=classes)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
