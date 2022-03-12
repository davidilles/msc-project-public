import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import pickle
import datetime

data_dir = "/home/idomino/OU/t847/data/samples_new/"
model_dir = "/home/idomino/OU/t847/data/models_new/"
results_dir = "/home/idomino/OU/t847/data/results_new/"

def get_dataset(n_malware, ratio_benign, dataset_type, features, seed):
    
    filename = f'{dataset_type}_{n_malware}_malware_x{ratio_benign}_benign_{features}_s{seed}.pkl'
    
    df = pd.read_pickle(data_dir + filename, compression='zip')
    print('Read sample:', filename)
    
    feature_columns = [x for x in df.columns if x not in ['appeared','label','avclass']]
    
    X = df[feature_columns]
    y = df['label']
    
    return (X, y, df)

def get_classifier(clf, seed):
    if clf == 'DT':
        return DecisionTreeClassifier(random_state=seed)
    elif clf == 'RF':
        return RandomForestClassifier(random_state=seed)
    elif clf == 'SVM':
        return make_pipeline(StandardScaler(), LinearSVC(random_state=seed))
    elif clf == 'LGBM':
        return lgb.LGBMClassifier(random_state=seed)
    else:
        return None

def train_and_save_model(n_malware, ratio_benign, features, clf_type, seed):
    X, y, df = get_dataset(n_malware, ratio_benign, 'train', features, seed)
    print(f'[{datetime.datetime.now()}]', 'Loaded X', X.shape, 'and y', y.shape)
    clf = get_classifier(clf_type, seed)
    print(f'[{datetime.datetime.now()}]','Starting fit:')
    clf.fit(X,y)
    print(f'[{datetime.datetime.now()}]','Fitted',clf)

    filename = f'{clf_type}_{n_malware}_malware_x{ratio_benign}_benign_{features}_s{seed}.pkl'
    pickle.dump(clf, open( model_dir + filename, "wb" ))
    print(f'[{datetime.datetime.now()}]','Saved model:', filename)
    print()
    # Clean up
    del X
    del y
    del df


for n_malware in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]:
    for clf_type in ['DT', 'RF', 'LGBM','SVM']:
        for features in ['parsed', 'format_agnostic', 'combined']:
            for seed in [1337, 1338, 1339]:
                train_and_save_model(n_malware, 1, features, clf_type, seed)

def create_observation(question, train_n_malware, test_n_malware, test_ratio_benign,
                       features, clf_type, metric, seed):
    
    model_pkl = f'{clf_type}_{train_n_malware}_malware_x1_benign_{features}_s{seed}.pkl'
    model = pickle.load(open(model_dir + model_pkl,"rb"))
    print('Loaded model', model, 'from:', model_pkl)
    
    X, y, df = get_dataset(test_n_malware, test_ratio_benign, 'test', features, seed)
    y_pred = None
    y_score = None
    retval = {
        'question': question, 'algorithm': clf_type, 'feature_set': features,
        'train_set_size': train_n_malware * 2,
        'test_set_size': test_n_malware + test_n_malware * test_ratio_benign,
        'test_set_ratio': f'1:{test_ratio_benign}',
        'perf_measure': metric,
        'seed': seed
    }
    
    if metric == 'accuracy':
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        retval['performance'] = acc
        retval['other_info'] = None
    elif metric == 'AUC' and clf_type != 'SVM':
        y_score = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_score)
        retval['performance'] = auc
        retval['other_info'] = None
    elif metric == 'AUC' and clf_type == 'SVM':
        retval['performance'] = None
        retval['other_info'] = None
    elif metric == 'real-life' and clf_type != 'SVM':
        y_score = model.predict_proba(X)[:,1]
        fpr,tpr,thresholds = roc_curve(y, y_score, drop_intermediate=False)
        i = np.argmax(fpr>=0.01)
        retval['performance'] = tpr[i]
        retval['other_info'] = fpr[i]
    elif metric == 'real-life' and clf_type == 'SVM':
        y_pred = model.predict(X)
        FP = np.logical_and(y == 0, y_pred == 1).sum()
        fpr = FP/len(y)
        tpr = recall_score(y, y_pred)
        retval['performance'] = tpr
        retval['other_info'] = fpr
    else:
        raise Exception('Invalid metric:', metric)
    
    
    # Clean up
    del X
    del y
    del df
    del model
    if y_pred is not None:
        del y_pred
    if y_score is not None:
        del y_score
        
    return retval

observations = []

# Question 1:
for n_malware in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]:
    for clf_type in ['DT', 'RF', 'LGBM','SVM']:
        for features in ['parsed', 'format_agnostic', 'combined']:
            for seed in [1337, 1338, 1339]:
                obs = create_observation(1, n_malware, n_malware, 1, features, clf_type, 'accuracy', seed)
                observations.append(obs)

# Question 2:
for n_malware in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]:
    for clf_type in ['DT', 'RF', 'LGBM','SVM']:
        for features in ['parsed', 'format_agnostic', 'combined']:
            for seed in [1337, 1338, 1339]:
                obs = create_observation(2, n_malware, 1250, 128, features, clf_type, 'real-life', seed)
                observations.append(obs)
            
# Question 3
for benign_ratio in [1, 2, 4, 8, 16, 32, 64, 128]:
    for clf_type in ['DT', 'RF', 'LGBM', 'SVM']:
        for features in ['parsed', 'format_agnostic', 'combined']:
            for seed in [1337, 1338, 1339]:
                obs = create_observation(3, 102400, 1250, benign_ratio, features, clf_type, 'real-life', seed)
                observations.append(obs)

# Create and save dataframe
observations_df = pd.DataFrame(observations, columns=['question', 'algorithm', 'feature_set',
                                   'train_set_size', 'test_set_size', 'test_set_ratio',
                                   'perf_measure', 'performance', 'other_info', 'seed'])

observations_df.to_csv(results_dir + 'results.csv', index=False)

