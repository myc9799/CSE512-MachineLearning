import numpy as np
import pandas as pd
from random import randrange
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--mode')
    parser.add_argument('--dataset')
    return parser

def decision_stump(data, d, T):
    prediction = np.ones((np.shape(data)[0], 1))
    prediction[data[:, d] > T] = -1.0
    return prediction

def build_stump(data, label, d):
    data = np.mat(data)
    label = np.mat(label).T
    m, n = np.shape(data)
    steps = 10.0
    best_stump = {}
    best_result = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min = data[:, i].min()
        range_max = data[:, i].max()
        step_size = (range_max - range_min) / steps
        for j in range(-1, int(steps)+1):
            T = (range_min + float(j) * step_size)
            prediction = decision_stump(data, i, T)
            error = np.mat(np.ones((m, 1)))
            error[prediction == label] = 0
            weighted_error = d.T * error
            if weighted_error < min_error:
                min_error = weighted_error
                best_result = prediction.copy()
                best_stump['dim'] = i
                best_stump['thresh'] = T
    return best_stump, min_error, best_result

def adaptive_boosting(data, label, num):
    weak_classifier = []
    loss_list = []
    m = np.shape(data)[0]
    d = np.mat(np.ones((m, 1))/m)
    combined_classifier = np.mat(np.zeros((m, 1)))
    for i in range(num):
        best_stump, error, result = build_stump(data, label, d)
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_classifier.append(best_stump)
        exp = np.multiply(-1 * alpha * np.mat(label).T, result)
        d = np.multiply(d, np.exp(exp))
        d = d/d.sum()
        combined_classifier = combined_classifier + alpha * result
        combined_error = np.multiply(np.sign(combined_classifier)!= np.mat(label).T,np.ones((m,1)))
        loss = combined_error.sum()/m
        loss_list.append(loss)
        print("\rTraining Episodes: {0}/{1}   Loss: {2:.3f}".format(i + 1, num, loss))
        if(loss == 0.0):
            break
    return weak_classifier,loss_list

def compute_loss(data, label, weight):
    predicted = np.zeros((np.shape(data)[0],1))
    for i in range(len(weight)):
        predicted = predicted + decision_stump(data,weight[i]['dim'],weight[i]['thresh'])
    predicted = np.array(predicted)
    predicted = np.sign(predicted)
    error = 0
    for i in range(len(label)):
        if label[i] != predicted[i]:
            error += 1
    loss = round(error / float(len(label)), 3)
    return loss

def _10_fold(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def cross_model(dataset,n_folds,num):
    folds = _10_fold(dataset, n_folds)
    flag = 0
    weight_list = []
    loss_list = []
    for fold in folds:
        flag = flag + 1
        print("-----------------------fold {}-------------------".format(flag))
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        train_label = np.array(train_set)[:, -1]
        train_set = np.array(train_set)[:, :-1]
        testdata = list()
        for row in fold:
            row_copy = list(row)
            testdata.append(row_copy)
            row_copy[-1] = None
        actual = [row[-1] for row in fold]
        actual = np.array(actual)
        testdata = np.array(testdata)[:, :-1]
        weights,_ = adaptive_boosting(train_set,train_label, num)
        weight_list.append(weights)
        loss = compute_loss(testdata,actual,weights)
        loss_list.append(loss)
    return loss_list,weight_list

def ERM():
    print("========================ERM Training=======================")
    print("===========================================================")
    label = raw_data.to_numpy()[:,-1]
    data = raw_data.to_numpy()[:,:-1]
    weight_list,loss= adaptive_boosting(data, label, num=10)
    weight_list = pd.DataFrame.from_dict(weight_list)
    print("========================REPORT==============================")
    print("The information of the weight assigned to each h(x) is:\n",weight_list)

def _10_folds():
    print("======================10 folds Training====================")
    print("===========================================================")
    dataset = raw_data.to_numpy().tolist()
    _10_fold_loss,_10_fold_weight_list = cross_model(dataset,10, num=25)
    print("========================REPORT==============================")
    print("The average loss of 10-fold-validation method is:",round(np.array(_10_fold_loss).mean(),3))
    print("The information of the weight list assigned to each h(x) in 10 folds are:\n"
          ,pd.DataFrame.from_dict(_10_fold_weight_list[len(_10_fold_weight_list)-1]))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    data_path = str(args.dataset)    
    raw_data = pd.read_csv(data_path)
    raw_data.loc[raw_data['diagnosis']==0,'diagnosis'] = -1
    if(mode == 'erm'):
        ERM()
    elif(mode == 'cross'):
        _10_folds()
 