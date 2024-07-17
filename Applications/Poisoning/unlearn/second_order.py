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


def get_parser():
    parser = argparse.ArgumentParser("second_order", description="Unlearn with second-order method.")
    parser.add_argument("model_folder", type=str, help="base directory to save models and results in")
    parser.add_argument("--config_file", type=str, default='unlearn_config.json',
                        help="config file with parameters for this experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable additional outputs")
    return parser


def run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, target_args='', reduction=1.0, verbose=False):
#     modeltype, dataset, target = target_args.split('_')
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

    model_init = model_initializers[modelname][dataset]
    
    
#     poisoned_filename = 'poisoned_model.hdf5'
#     repaired_filename = 'repaired_model.hdf5'
    poisoned_filename = f'{dataset}_{modelname}_poisoned_model.hdf5'
    repaired_filename = f'{modelname}_{dataset}_{target}_{prefix}_repaired_model.hdf5'
    
    second_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig,
                            injector.injected_idx, unlearn_kwargs=unlearn_kwargs, verbose=verbose, target_args=target_args)


def second_order_unlearning(model_folder, poisoned_filename, repaired_filename, model_init, data, y_train_orig, delta_idx,
                            unlearn_kwargs, order=2, verbose=False, target_args=''):
    
        
    modelname, dataset, target= target_args.split('_', 2)
    target, prefix, num_layers = target.split('-')
    
    name = f'{dataset}_{modelname}_{target}_{prefix}'
    
    unlearning_result = UnlearningResult(model_folder, dataset, name)
    poisoned_weights = os.path.join(parent(model_folder), poisoned_filename)
    log_dir = model_folder
    
        # check if unlearning has already been performed
    if unlearning_result.exists:
        print(f"Unlearning results already exist for {modelname} {dataset}")
        return
    
    
    train_results = f"{dataset}_{modelname}_train_results.json"
    # start unlearning hyperparameter search for the poisoned model
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


def main(model_folder, config_file, verbose, target_args=''):
    config_file = os.path.join(model_folder, config_file)
    train_kwargs = Config.from_json(os.path.join(parent(model_folder), 'train_config.json'))
    unlearn_kwargs = Config.from_json(config_file)
    poison_kwargs = Config.from_json(os.path.join(parent(model_folder), 'poison_config.json'))
    run_experiment(model_folder, train_kwargs, poison_kwargs, unlearn_kwargs, verbose=verbose, target_args=target_args)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
