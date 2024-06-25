import os
from os.path import dirname as parent
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.train import train
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_GTSRB, get_VGG16_CIFAR100, extractfeatures_VGG16, extractfeatures_RESNET50, classifier_VGG16, classifier_RESNET50 
from Applications.Poisoning.model import get_RESNET50_CIFAR10, get_RESNET50_MNIST, get_RESNET50_FASHION, get_RESNET50_SVHN, get_RESNET50_GTSRB, get_RESNET50_CIFAR100
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from Applications.Sharding.ensemble import train_models


def get_parser():
    parser = argparse.ArgumentParser("poison_models", description="Train poisoned models.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='poison_config.json', help="config file with parameters for this experiment")
    return parser


def train_poisoned(model_folder, poison_kwargs, train_kwargs, dataset='cifar10', modelname='VGG16', classes=10):
    data = None
    model_init = None



    if dataset == 'Cifar10':
        data = Cifar10.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_CIFAR10(classes)
        else:
            model_init = lambda: get_RESNET50_CIFAR10(num_classes=10, dense_units=512, lr_init=0.001, sgd=False)
    elif dataset == 'Mnist':
        data = Mnist.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_MNIST(classes)
            print(f"Classes: {classes}, Modelname: {modelname}")
        else:
            model_init = lambda: get_RESNET50_MNIST(classes)

    elif dataset == 'FashionMnist':
        data = FashionMnist.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_FASHION(classes)
        else:
            model_init = lambda: get_RESNET50_FASHION(classes)
    
    elif dataset == 'SVHN':
        data = SVHN.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_SVHN(classes)
        else:
            model_init = lambda: get_RESNET50_SVHN(classes)

    elif dataset == 'GTSRB':
        data = GTSRB.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_GTSRB(classes)
        else:
            model_init = lambda: get_RESNET50_GTSRB(classes)

    elif dataset == 'Cifar100':
        data = Cifar100.load()
        if modelname == 'VGG16':
            model_init = lambda: get_VGG16_CIFAR100(classes)
        else:
            model_init = lambda: get_RESNET50_CIFAR100(classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    if data is None or model_init is None:
        raise ValueError(f"Data or model initialization function not properly set for dataset: {dataset} and modelname: {modelname}")

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
        train_models(model_init, model_folder, data, n_shards, model_filename=dataset+"_"+modelname+'_poisoned_model.hdf5', **train_kwargs)
    else:
        train(dataset, modelname,model_init, model_folder, data, model_filename=dataset+"_"+modelname+'_poisoned_model.hdf5', **train_kwargs)


def main(model_folder, config_file, dataset='cifar10', modelname='VGG16', classes=10):
    if 'sharding' in str(model_folder):
        poison_kwargs = Config.from_json(os.path.join(parent(model_folder), config_file))
        train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    else:
        poison_kwargs = Config.from_json(os.path.join(model_folder, config_file))
        train_kwargs = Config.from_json(os.path.join(model_folder, 'train_config.json'))
    train_poisoned(model_folder, poison_kwargs, train_kwargs, dataset, modelname, classes=classes)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
