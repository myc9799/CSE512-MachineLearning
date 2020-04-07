from random import randrange
from statistics import mean
import pandas as pd
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--mode')
    parser.add_argument('--dataset')
    return parser
n_folds = 10

def predict(each, weight):
    activate = weight[0]
    for i in range(len(each) - 1):
        activate += weight[i + 1] * each[i]
    if activate >= 0.0:
        return 1.0
    else:
        return 0.0

def compute_loss(actual, predicted):
	error = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i]:
			error += 1
	return round(error / float(len(actual)),3)

def model(data, testdata, episodes):
    weight = [0.0 for i in range(len(data[0]))]
    test_predicted = []
    loss_list = []
    for episode in range(episodes):
        num_error = 0.0
        for each in data:
            prediction = predict(each, weight)
            error = each[-1] - prediction
            num_error += error ** 2
            weight[0] = weight[0] + error
            for i in range(len(data[0]) - 1):
                weight[i + 1] = weight[i + 1] + error * each[i]
            loss = num_error/len(data)
        loss_list.append(loss)
        print("\rTraining Episodes: {0}/{1}   Loss: {2:.3f}".format(episode + 1, episodes, loss))

    for each in testdata:
        test_prediction = predict(each, weight)
        test_predicted.append(test_prediction)
    return test_predicted,loss_list,weight

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

def cross_model(dataset,n_folds, episodes):
    folds = _10_fold(dataset, n_folds)
    loss_list = list()
    flag = 0
    weight_list = []
    for fold in folds:
        flag = flag + 1
        # print("-----------------------fold {}-------------------".format(flag))
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        testdata = list()
        for row in fold:
            row_copy = list(row)
            testdata.append(row_copy)
            row_copy[-1] = None
        actual = [row[-1] for row in fold]
        predicted ,_,weight= model(train_set, testdata, episodes)
        weight_list.append(weight)
        loss = compute_loss(actual, predicted)
        loss_list.append(loss)
    return loss_list,weight_list

def ERM(dataset,episode):
    print("========================ERM Training=======================")
    print("===========================================================")
    _, loss, weight = model(dataset, dataset, episode)
    print("========================REPORT==============================")
    print("The final weight parameters are:", weight)

def _10_folds(dataset,n_folds, episode):
    print("======================10 folds Training====================")
    print("===========================================================")
    _10_fold_loss, weight_list = cross_model(dataset, n_folds, episode)
    print("========================REPORT==============================")
    print('The loss list for all the folds is :', _10_fold_loss)
    print('The average loss for all the folds is :', round(mean(_10_fold_loss), 3))
    print("The final list of weight parameters are:", weight_list)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    data_path = str(args.dataset)    
    data = pd.read_csv(data_path)
    data = data.to_numpy().tolist()

    if(len(data[0])==3):
        episode = 15
    else:
        episode = 2000

    if(mode == 'erm'):
        ERM(data, episode)
    elif(mode == 'cross'):
        _10_folds(data, n_folds, episode)
