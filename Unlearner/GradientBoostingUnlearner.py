
import numpy as np
import json
import pickle as pkl
import os
import hashlib
# import Suppport Vectors Machine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import scipy.sparse as sp
from scipy.special import expit
from scipy.linalg import inv
from Unlearner.DNNUnlearner import DNNUnlearner


class GradientBoostingUnlearner(DNNUnlearner):
    def __init__(self, train_data, test_data, voc, model_param_str, n_estimators=None, learning_rate=None, max_depth=None, random_state=None):
        self.set_train_test_data(train_data, test_data)
        self.voc = voc
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        self.model_param_str = model_param_str
        self.theta = None

    def train_model(self, model_folder, **kwargs):
        self.model.fit(self.x_train, self.y_train)
        model_name = 'rf_model_{}.pkl'.format(self.model_param_str)
        report_name = 'rf_performance_{}.json'.format(self.model_param_str)
        report = self.get_performance(self.x_test, self.y_test, model=self.model)
        print('Training results ({}):'.format(self.model_param_str))
        print(json.dumps(report, indent=4))
        json.dump(report, open(os.path.join(model_folder, report_name), 'w'), indent=4)
        pkl.dump(self.model, open(os.path.join(model_folder, model_name), 'wb'))
        self.set_model(self.model)


    def set_train_test_data(self, train_data, test_data):
        self.x_train = train_data[0]
        self.x_test = test_data[0]
        self.y_train = train_data[1]
        self.y_test = test_data[1]
        self.n = self.x_train.shape[0]
        self.dim = self.x_train.shape[1]

    def set_x_train(self, x_train):
        self.x_train = x_train.toarray() if sp.issparse(x_train) else x_train
        self.n = x_train.shape[0]
        self.dim = x_train.shape[1]

    def set_y_train(self, y_train):
        self.y_train = y_train
        self.n = y_train.shape[0]

    def set_model(self, model):
        self.model = model
        self.theta = np.squeeze(model.coef_.T)

    def retrain_model( self, indice_to_delete, save_folder, retrain_labels=False, **kwargs):
        new_model = GradientBoostingClassifier(n_estimators=kwargs['n_estimators'], learning_rate=kwargs['learning_rate'], max_depth=kwargs['max_depth'], random_state=kwargs['random_state'])
        if retrain_labels:
            x_train_delta = self.x_train
            x_test_delta = self.x_test
            y_train_delta = self.get_data_copy_y('train', indices_to_delete=indice_to_delete)
            y_test_delta = self.y_test
        else:
            x_train_delta = self.get_data_copy('train', indices_to_delete=indice_to_delete)
            x_test_delta = self.get_data_copy('test', indices_to_delete=indice_to_delete)
            y_train_delta = self.y_train
            y_test_delta = self.y_test
        new_model.fit(x_train_delta, y_train_delta)
        report = self.get_performance(x_test_delta, y_test_delta, model=new_model)
        no_features_to_delete = len(indice_to_delete)
        combination_string = hashlib.sha256('-'.join([str(i) for i in indice_to_delete]).encode()).hexdigest()
        model_folder = os.path.join(save_folder, str(no_features_to_delete), combination_string)
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        report_name = 'retraining_performance_{}.json'.format(self.model_param_str)
        with open(os.path.join(model_folder, report_name), 'w') as f:
            json.dump(report, f, indent=4)
        return new_model
    
    def get_performance(self, x, y, **kwargs):
        assert ('theta' in kwargs or 'model' in kwargs)
        assert x.shape[0] == y.shape[0], '{} != {}'.format(x.shape[0], y.shape[0])
        if 'model' in kwargs:
            model = kwargs['model']
            logits = model.predict_proba(x)[:,1]
            y_pred = model.predict(x)
            theta = np.squeeze(model.coef_.T)
        else:
            theta = kwargs['theta']
            logits = expit(np.dot(x, theta))
            y_pred = np.array([1 if l >= 0.5 else -1 for l in logits])
        accuracy = len(np.where(y_pred == y)[0])/x.shape[0]
        fpr, tpr, _ = roc_curve(y, logits)
        prec, rec, _ = precision_recall_curve(y, logits)
        auc_roc = auc(fpr, tpr)
        auc_pr = auc(rec, prec)
        report = classification_report(y, y_pred, digits=4, output_dict=True)
        n_data = x.shape[0]
        loss = 1./(n_data * self.lambda_) * self.get_loss(theta, x, y) + 0.5 * np.dot(theta, theta.T)
        grad = 1./(n_data * self.lambda_) * self.get_gradient(theta, x, y) + theta
        report['test_loss'] = loss
        report['gradient_norm'] = np.sum(grad**2)
        report['train_loss'] = self.get_train_set_loss(theta)
        report['gradient_norm_train'] = np.sum(self.get_train_set_gradient(theta)**2)
        report['accuracy'] = accuracy
        report['test_roc_auc'] = auc_roc
        report['test_pr_auc'] = auc_pr
        return report
    
    def get_loss(self, theta, x, y):
        dot_prod = np.dot(x, theta) * y
        data_loss = np.log(1 + np.exp(-dot_prod))
        total_loss = np.sum(data_loss, axis=0)
        return total_loss

    # return total loss L on train set.
    def get_train_set_loss(self, theta):
        summed_loss = self.get_loss(theta, self.x_train, self.y_train)
        total_loss = self.C * summed_loss + 0.5 * np.dot(theta, theta.T)**2
        return total_loss

    # returns loss L on test set. Notice that the C is not correct in this loss since N_train != N_test
    def get_test_set_loss(self, theta):
        n_test = self.x_test.shape[0]
        summed_loss = self.get_loss(theta, self.x_test, self.y_test)
        total_loss = 1./(self.lambda_ * n_test) * summed_loss + 0.5 * np.dot(theta, theta.T)**2
        return total_loss

    # get gradient w.r.t. parameters (-y*x*sigma(-y*Theta^Tx)) for y in {-1,1}
    # this gradient is only the gradient of l, not of L!
    def get_gradient(self, theta, x, y):
        assert x.shape[0] == y.shape[0]
        dot_prod = np.dot(x, theta) * y
        factor = -expit(-dot_prod) * y
        grad = np.sum(np.expand_dims(factor,1) * x, axis=0)
        return grad

    # this is the gradient of L on the train set. This should be close to zero after fitting.
    def get_train_set_gradient(self, theta):
        grad = self.get_gradient(theta, self.x_train, self.y_train)
        return self.C*grad + self.theta

    # get gradient w.r.t. input (-y*Theta*sigma(-y*Theta^Tx))
    def get_gradient_x(self, x, y, theta):
        assert x.shape[0] == y.shape[0]
        dot_prod = np.dot(x, theta) * y
        factor = -expit(-dot_prod) * y
        return np.expand_dims(factor, 1) * theta

    # computes inverse hessian for data x. As we only need the inverse hessian on the entire dataset we return the
    # Hessian on the full L loss.
    def get_inverse_hessian(self, x):
        dot = np.dot(x, self.theta)
        probs = expit(dot)
        weighted_x = np.reshape(probs * (1 - probs), (-1, 1)) * x  # sigma(-t) = (1-sigma(t))
        cov = self.C * np.dot(x.T, weighted_x)
        cov += np.eye(self.dim)  # hessian of regularization
        cov_inv = inv(cov)
        return cov_inv

    def get_relevant_indices(self, indices_to_delete):
        # get the rows (samples) where the features appear
        relevant_indices = np.where(self.x_train[:, indices_to_delete] != 0)[0]
        # to avoid having samples more than once
        relevant_indices = np.unique(relevant_indices)
        return relevant_indices

    # indices is a list of dimensions referring to the dimensions of training set
    def approx_retraining(self, indices_to_delete, retrain_y=False, **kwargs):
        if retrain_y:
            assert np.min(indices_to_delete) >= 0 and np.max(indices_to_delete) < self.n
        else:
            assert np.min(indices_to_delete) >= 0 and np.max(indices_to_delete) <= self.dim
        method = kwargs['method'] if 'method' in kwargs else 'influence'
        if method == 'lazy':
            theta_approx = self.theta
        else:
            # often H^-1 fits into memory and has to be computed only once
            if 'H_inv' in kwargs:
                H_inv = kwargs['H_inv']
            else:
                if method == 'newton':
                    x_train_deleted = self.get_data_copy('train', indices_to_delete)
                    H_inv = self.get_inverse_hessian(x_train_deleted)
                else:
                    H_inv = self.get_inverse_hessian(self.x_train)
            if retrain_y:
                z_x = self.x_train[indices_to_delete]
                z_y = self.y_train[indices_to_delete]
                z_x_delta = z_x
                z_y_delta = self.get_data_copy_y('train', indices_to_delete=indices_to_delete)[indices_to_delete]
            else:
                relevant_indices = self.get_relevant_indices(indices_to_delete)
                z_x = self.x_train[relevant_indices]
                z_y = self.y_train[relevant_indices]
                z_x_delta = self.get_data_copy('train', indices_to_delete=indices_to_delete)[relevant_indices]
                z_y_delta = z_y
            grad_x = self.get_gradient(self.theta, z_x, z_y)
            grad_x_delta = self.get_gradient(self.theta, z_x_delta, z_y_delta)
            # compute parameter update. Note that here we have to choose epsilon=C because of the loss function used
            # in sklearn
            delta_theta = -self.C * H_inv.dot(grad_x_delta - grad_x)
            theta_approx = self.theta + delta_theta
        return theta_approx
    