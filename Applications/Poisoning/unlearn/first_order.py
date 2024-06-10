import os
from os.path import dirname as parent
import json
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_GTSRB, get_VGG16_CIFAR100
from Applications.Poisoning.model import get_RESNET50_CIFAR10, get_RESNET50_MNIST, get_RESNET50_FASHION, get_RESNET50_SVHN, get_RESNET50_GTSRB, get_RESNET50_CIFAR100
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from Applications.Poisoning.unlearn.common import evaluate_unlearning
from util import UnlearningResult, reduce_dataset


def get_parser():
    parser = argparse.ArgumentParser("first_order", description="Unlearn with first-order method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    return parser


def run_experiment(dataset, modeltype, model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, reduction=1.0, verbose=False):
    if dataset == "Cifar10":
        data = Cifar10.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_CIFAR10(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_CIFAR10(dense_units=train_kwargs['model_size'])

    if dataset == "Mnist":
        data = Mnist.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_MNIST(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_MNIST(dense_units=train_kwargs['model_size'])

    if dataset == "FashionMnist":
        data = FashionMnist.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_FASHION(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_FASHION(dense_units=train_kwargs['model_size'])

    if dataset == "SVHN":
        data = SVHN.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_SVHN(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_SVHN(dense_units=train_kwargs['model_size'])
    
    if dataset == "GTSRB":
        data = GTSRB.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_GTSRB(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_GTSRB(dense_units=train_kwargs['model_size'])
    if dataset == "Cifar100":
        data = Cifar100.load()
        if modeltype == "RESNET50":
            model_init = lambda: get_RESNET50_CIFAR100(dense_units=train_kwargs['model_size'])
        else:
            model_init = lambda: get_VGG16_CIFAR100(dense_units=train_kwargs['model_size'])

        
    (x_train, y_train), _, _ = data
    y_train_orig = y_train.copy()

    # inject label flips
    injector_path = os.path.join(model_folder, 'injector.pkl')
    if os.path.exists(injector_path):
        injector = LabelflipInjector.from_pickle(injector_path)
    else:
        injector = LabelflipInjector(parent(model_folder), **poison_kwargs)
    x_train, y_train = injector.inject(x_train, y_train)
    data = ((x_train, y_train), data[1], data[2])

    # prepare unlearning data
    (x_train,  y_train), _, _ = data
    x_train, y_train, idx_reduced, delta_idx = reduce_dataset(
        x_train, y_train, reduction=reduction, delta_idx=injector.injected_idx)
    if verbose:
        print(f">> reduction={reduction}, new train size: {x_train.shape[0]}")
    y_train_orig = y_train_orig[idx_reduced]
    data = ((x_train, y_train), data[1], data[2])

    poisoned_filename = dataset+"_"+modeltype+'_poisoned_model.hdf5'
    repaired_filename = dataset+"_"+modeltype+'_repaired_model.hdf5'
    first_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data,
                           y_train_orig, injector.injected_idx, unlearn_kwargs, verbose=verbose)


def first_order_unlearning(dataset, modeltype, model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx,
                            unlearn_kwargs, order=1, verbose=False):
    unlearning_result = UnlearningResult(model_folder, dataset, modeltype)
    poisoned_weights = os.path.join(parent(model_folder), poisoned_filename)
    log_dir = model_folder

    # start unlearning hyperparameter search for the poisoned model
    with open(model_folder.parents[2]/'clean'/'train_results.json', 'r') as f:
        clean_acc = json.load(f)['accuracy']
    repaired_filepath = os.path.join(model_folder, repaired_filename)
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    unlearn_kwargs['order'] = order
    acc_before, acc_after, diverged, logs, unlearning_duration_s, params = evaluate_unlearning(model_init, poisoned_weights, data, delta_idx, y_train_orig, unlearn_kwargs, clean_acc=clean_acc,
                                                                                       repaired_filepath=repaired_filepath, verbose=verbose, cm_dir=cm_dir, log_dir=log_dir)
    acc_perc_restored = (acc_after - acc_before) / (clean_acc - acc_before)

    unlearning_result.update({
        'acc_clean': clean_acc,
        'acc_before_fix': acc_before,
        'acc_after_fix': acc_after,
        'acc_perc_restored': acc_perc_restored,
        'diverged': diverged,
        'n_gradients': sum(logs),
        'unlearning_duration_s': unlearning_duration_s,
        'num_params': params
    })
    unlearning_result.save()


def main(model_folder, config_file, verbose, dataset='Cifar10', modeltype="VGG16"):
    config_file = os.path.join(model_folder, config_file)
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    unlearn_kwargs = Config.from_json(config_file)
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    run_experiment(dataset, modeltype, model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, verbose=verbose)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
