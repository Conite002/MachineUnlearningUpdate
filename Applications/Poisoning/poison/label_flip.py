import os
import argparse

import numpy as np
from tensorflow.keras.losses import categorical_crossentropy

from Applications.Poisoning.dataset import Cifar10, Mnist, FashionMnist, SVHN, GTSRB, Cifar100
from Applications.Poisoning.model import get_VGG16_CIFAR10, get_VGG16_MNIST, get_VGG16_FASHION, get_VGG16_SVHN, get_VGG16_GTSRB, get_VGG16_CIFAR100, extractfeatures_RESNET50, get_RESNET50_CIFAR10
from Applications.Poisoning.model import get_RESNET50_CIFAR100, get_RESNET50_SVHN, get_RESNET50_MNIST, get_RESNET50_FASHION, classifier_VGG16, classifier_RESNET50, extractfeatures_VGG16, classifier_VGG16_CIFAR100, classifier_RESNET50_CIFAR100, extractfeatures_VGG16_CIFAR100, extractfeatures_RESNET50_CIFAR100
from Applications.Poisoning.train import train
from util import UnlearningResult, MixedResult, GradientLoggingContext, LabelFlipResult, save_train_results


def search_flip_samples(model, data, budget):
    (x_train, y_train), _, _ = data
    # 1. for each sample, evaluate loss towards all classes
    preds = model.predict(x_train)
    n_classes = y_train.shape[1]
    y_all = np.eye(n_classes)
    losses = np.zeros(len(x_train))
    targets = np.zeros((len(x_train), n_classes))
    for i in range(len(preds)):
        pred = np.tile(preds[i], n_classes).reshape(n_classes, -1)
        loss = categorical_crossentropy(pred, y_all)
        maxloss_idx = np.argmax(loss)
        losses[i] = loss[maxloss_idx]
        targets[i, maxloss_idx] = 1

    # 2. greedy search over largest loss values until `budget` samples are selected
    sort_idx = np.argsort(losses)[::-1]
    return sort_idx[:budget], targets[sort_idx[:budget]]


def flip_max_loss(model, data, budget=200):
    flip_idx, targets = search_flip_samples(model, data, budget)
    (x_train, y_train), _, _ = data
    y_train[flip_idx] = targets
    return ((x_train, y_train), data[1], data[2])


def create_rand_offset(Y, seed):
    n_classes = Y.shape[1]
    np.random.seed(seed)
    rand_offset = np.random.randint(0, n_classes, size=n_classes)
    return rand_offset


def _flip_labels(Y, rand_offset):
    n_classes = Y.shape[1]
    y_flip = np.argmax(Y, axis=1)
    rand_offset = rand_offset[y_flip]
    y_flip = (y_flip + rand_offset) % n_classes
    y_onehot = np.zeros((len(y_flip), n_classes))
    y_onehot[range(len(y_onehot)), y_flip] = 1
    return y_onehot


def flip_labels(Y, budget=200, seed=42, target=-1, verbose=False):
    np.random.seed(seed)
    idx = np.random.permutation(len(Y))
    if target >= 0:
        idx = idx[np.argwhere(np.argmax(Y[idx], axis=1) == target)[:, 0]]

    sources = list(range(10))
    targets = sources[::-1]
    idx_list = []
    Y_orig = Y.copy()
    budget //= 10  # 10th of the budget for each pair
    for s, t in zip(sources, targets):
        _idx = idx[np.argwhere(np.argmax(Y_orig[idx], axis=1) == s)[:, 0]][:budget]
        label = np.eye(10)[t].reshape(1, 10)
        if verbose:
            print(f">> flipping {len(_idx)} labels from {s} to {t}")
        Y[_idx] = label
        idx_list.append(_idx)
    idx = np.concatenate(idx_list, axis=0)
    if verbose:
        print(f">>> injected {len(idx)} flips into {len(Y)} labels")
    return Y, idx


def get_parser():
    parser = argparse.ArgumentParser("label_flip", description="Poison models using label flipping and measure backdoor success.")
    parser.add_argument("model_folder", type=str, help="Where to save models.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.", default=64)
    parser.add_argument("--lr_init", type=float, help="Initial learning rate.", default=1e-4)
    parser.add_argument("--epochs", type=int, help="No epochs to train.", default=100)
    parser.add_argument('--budget', type=int,
                        help='Number of training data with injected backdoor', default=200)
    parser.add_argument('--base_seed', type=int, help='Base seed', default=42)
    parser.add_argument('--n_repititions', type=int, help='Number of random source/target pairs to generate.',
                        default=1)
    return parser


def main(model_folder, budget, batch_size=64, lr_init=1e-4,
         epochs=100, base_seed=42, n_repititions=1, dataset="cifar10", modelname="VGG16"):
    train_kwargs = dict(batch_size=batch_size, lr_init=lr_init, epochs=epochs)

    # train and evaluate clean model as reference
    # TODO
    # train_clean(os.path.join(model_folder, 'clean-model'), skip_existing=True, **train_kwargs)

    # train models on poisoned data
    for i in range(n_repititions):
        seed = base_seed + i
        model_folder = os.path.join(model_folder, f'budget-{budget}', f'seed{seed}')
        if dataset == "Cifar10":
            data = Cifar10.load()
            if modelname == "VGG16":
                eval_flipped_model(dataset, modelname, data, get_VGG16_CIFAR10, model_folder, budget, seed, **train_kwargs)
            elif modelname == "RESNET50":
                eval_flipped_model(dataset, modelname, data, get_RESNET50_CIFAR10, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_VGG16":
                eval_flipped_model(dataset, modelname, data, classifier_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_RESNET50":
                eval_flipped_model(dataset, modelname, data, classifier_RESNET50, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_VGG16":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_RESNET50":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)

            else:
                raise ValueError("Invalid model name")
            
        elif dataset == "Mnist":
            data = Mnist.load()
            if modelname == "VGG16":
                eval_flipped_model(dataset,modelname,  data, get_VGG16_MNIST, model_folder, budget, seed, **train_kwargs)
            elif modelname == "RESNET50":
                eval_flipped_model(dataset, modelname, data, get_RESNET50_MNIST, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_VGG16":
                eval_flipped_model(dataset, modelname, data, classifier_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_RESNET50":
                eval_flipped_model(dataset, modelname, data, classifier_RESNET50, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_VGG16":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_RESNET50":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            else:
                raise ValueError("Invalid model name")
            
        elif dataset == "FashionMnist":
            data = FashionMnist.load()
            if modelname == "VGG16":
                eval_flipped_model(dataset, modelname, data, get_VGG16_FASHION, model_folder, budget, seed, **train_kwargs)
            elif modelname == "RESNET50":
                eval_flipped_model(dataset, modelname, data, get_RESNET50_FASHION, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_VGG16":
                eval_flipped_model(dataset, modelname, data, classifier_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_RESNET50":
                eval_flipped_model(dataset, modelname, data, classifier_RESNET50, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_VGG16":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_RESNET50":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            else:
                raise ValueError("Invalid model name")
        
        elif dataset == "SVHN":
            data = SVHN.load()
            if modelname == "VGG16":
                eval_flipped_model(dataset, modelname, data, get_VGG16_SVHN, model_folder, budget, seed, **train_kwargs)
            elif modelname == "RESNET50":
                eval_flipped_model(dataset, modelname, data, get_RESNET50_SVHN, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_VGG16":
                eval_flipped_model(dataset, modelname, data, classifier_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_RESNET50":
                eval_flipped_model(dataset, modelname, data, classifier_RESNET50, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_VGG16":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_RESNET50":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16, model_folder, budget, seed, **train_kwargs)
            else:
                raise ValueError("Invalid model name")
            
        elif dataset == "Cifar100":
            data = Cifar100.load()
            if modelname == "VGG16":
                eval_flipped_model(dataset, modelname, data, get_VGG16_CIFAR100, model_folder, budget, seed, **train_kwargs)
            elif modelname == "RESNET50":
                eval_flipped_model(dataset, modelname, data, get_RESNET50_CIFAR100, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_VGG16":
                eval_flipped_model(dataset, modelname, data, classifier_VGG16_CIFAR100, model_folder, budget, seed, **train_kwargs)
            elif modelname == "classifier_RESNET50":
                eval_flipped_model(dataset, modelname, data, classifier_RESNET50_CIFAR100, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_VGG16":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16_CIFAR100, model_folder, budget, seed, **train_kwargs)
            elif modelname == "extractfeatures_RESNET50":
                eval_flipped_model(dataset, modelname, data, extractfeatures_VGG16_CIFAR100, model_folder, budget, seed, **train_kwargs)
            else:
                raise ValueError("Invalid model name")
        else:
            raise ValueError("Invalid dataset name")            

def eval_flipped_model(dataset, modelname, data, model_init, model_folder, budget, seed=42, **train_kwargs):
    os.makedirs(model_folder, exist_ok=True)
    result = LabelFlipResult(model_folder)

    if result.exists:
        return
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = data
    y_train_orig = y_train.copy()
    y_train, _ = flip_labels(y_train, create_rand_offset(y_train, seed), budget, seed)

    weight_path = os.path.join(model_folder, dataset +"_"+modelname+"_"+'_best_model.hdf5')
    if not os.path.exists(weight_path):
        train(model_init, model_folder, data, **train_kwargs)
        save_train_results(model_folder)
    flipped_model = model_init(weight_path=weight_path)

    retrain_folder = os.path.join(model_folder, 'retraining')
    retrain_weights = os.path.join(retrain_folder,  dataset+"_"+modelname+'best_model.hdf5')
    os.makedirs(retrain_folder, exist_ok=True)
    y_train = y_train_orig
    train(model_init, retrain_folder, data, **train_kwargs)
    save_train_results(retrain_folder)
    print(f'Retraining model finished')
    retrained_model = model_init(weight_path=retrain_weights)

    # flipped model on clean validation data vs. retrained model
    flipped_acc = flipped_model.evaluate(x_valid, y_valid, verbose=0)[1]
    retrained_acc = retrained_model.evaluate(x_valid, y_valid, verbose=0)[1]
    print(f'Results: retrained_acc={retrained_acc}, flipped_acc={flipped_acc}')

    result.update({
        'retrained_acc': retrained_acc,
        'flipped_acc': flipped_acc
    })
    result.save()


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(**vars(args))
