import os
from os.path import dirname as parent
import json
import argparse

from Applications.Poisoning.configs.config import Config
from Applications.Poisoning.model import get_VGG_CIFAR10, get_VGG_CIFAR10, get_VGG16_SVHN, get_RESNET50_SVHN, get_RESNET50_CIFAR10, get_RESNET50_CIFAR100
from Applications.Poisoning.poison.injector import LabelflipInjector
from Applications.Poisoning.dataset import Cifar10, SVHN, Cifar100
from Applications.Poisoning.unlearn.common import evaluate_unlearning
from util import UnlearningResult, reduce_dataset

from sklearn.metrics import accuracy_score 
import numpy as np

def evaluate(model, data, weights_path):
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = data
    model.load_weights(weights_path)
    y_pred = model.predict(x=x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc


def get_parser():
    parser = argparse.ArgumentParser("first_order", description="Unlearn with first-order method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    return parser


def run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, target_args='', reduction=1.0, verbose=False, modelweights=None):           
    modelname, dataset, target= target_args.split('_', 2)
    target, prefix, num_layers = target.split('-')
    
#     print(f'{modelname} {dataset} {target}')
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
    data_copy = data.copy()
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

    # Initialize model
    model_init = model_initializers[modelname][dataset]
    
#     poisoned_filename = 'poisoned_model.hdf5'
#     repaired_filename = 'repaired_model.hdf5'
    poisoned_filename = f'{dataset}_{modelname}_poisoned_model.hdf5'
    if modelweights is not None:
        poisoned_filename = modelweights
        acc = evaluate(model_init,data=data_copy, weights_path=poisoned_filename)
        print(f"Accuracy init model :  {acc}")
        
    repaired_filename = f'{modelname}_{dataset}_{target}_{prefix}_repaired_model.hdf5'
    
    first_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data,
                           y_train_orig, injector.injected_idx, unlearn_kwargs, verbose=verbose, target_args=target_args)


def first_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx,
                            unlearn_kwargs, order=1, verbose=False, target_args=''):
    
    modelname, dataset, target= target_args.split('_', 2)
    target, prefix, num_layers = target.split('-')
#     modelname, dataset, target = target_args.split('_')
    print(f" target : {target}")
    print(f"prefix ==: {prefix}")
    name = f'{dataset}_{modelname}_{target}_{prefix}'
    unlearning_result = UnlearningResult(model_folder, dataset, name)
    poisoned_weights = os.path.join(parent(model_folder), poisoned_filename)
        
    log_dir = model_folder
    
    # check if unlearning has already been performed
    if unlearning_result.exists:
        print(f"Unlearning results already exist for {modelname} {dataset}")
        return
    
    
    # start unlearning hyperparameter search for the poisoned model
    
    train_results = f"{modelname}_{dataset}_{prefix}_train_results.json"
    with open(model_folder.parents[2]/'clean'/train_results, 'r') as f:
        clean_acc = json.load(f)['accuracy']
    repaired_filepath = os.path.join(model_folder, repaired_filename)
    cm_dir = os.path.join(model_folder, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    unlearn_kwargs['order'] = order
    acc_before, acc_after, diverged, logs, unlearning_duration_s, params = evaluate_unlearning(model_init, poisoned_weights, data, delta_idx, y_train_orig, unlearn_kwargs, clean_acc=clean_acc,
                                                                                       repaired_filepath=repaired_filepath, verbose=verbose, cm_dir=cm_dir, log_dir=log_dir, target_args=target_args)
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


def main(model_folder, config_file, verbose, target_args='', modelweights=None):
    config_file = os.path.join(model_folder, config_file)
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    unlearn_kwargs = Config.from_json(config_file)
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, verbose=verbose, target_args=target_args, modelweights=modelweights)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
