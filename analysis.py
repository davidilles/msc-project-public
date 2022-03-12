import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

sns.set_theme(style="ticks", palette="pastel")

data = pd.read_csv('results.csv')
print(data.shape)
data.head()


def get_question_df(question):
    df = data[data.question == question]
    print(f'Returned dataframe for question {question} with shape {df.shape}')
    return df

def get_corr_coeff(df, col1, col2, algorithm=None):
    data = df[df.algorithm==algorithm]
    return {
        'algorithm': algorithm,
        'corr_coeff': data[col1].corr(data[col2])
    }


def get_sensitivity_data_point(df, algorithm, feature_set, from_train_set_size, input_mult, seed):
    tmp = df[df.algorithm==algorithm]
    tmp = tmp[tmp.feature_set==feature_set]
    tmp = tmp[tmp.seed==seed]
    from_perf = tmp[tmp.train_set_size==from_train_set_size]['performance'].iloc[0]
    to_perf = tmp[tmp.train_set_size==from_train_set_size*input_mult]['performance'].iloc[0]
    return {
        'algorithm': algorithm,
        'feature_set': feature_set,
        'from_train_set_size': from_train_set_size,
        'input_mult': input_mult,
        'sensitivity': to_perf-from_perf,
        'seed': seed
    }

def get_sensitivity_df(df, train_set_sizes, input_mult):
    sens_dict = []
    for index, row in df[['algorithm','feature_set', 'seed']].drop_duplicates().iterrows():
        for i in train_set_sizes:
            p = get_sensitivity_data_point(df, row['algorithm'], row['feature_set'],
                                           i, input_mult, row['seed'])
            sens_dict.append(p)
    sens_df = pd.DataFrame(sens_dict)
    return sens_df

def get_algorithm_name(algo):
    if algo == 'DT':
        return 'Decision Tree'
    elif algo == 'RF':
        return 'Random Forest'
    elif algo == 'LGBM':
        return 'Gradient Boosted Trees'
    elif algo == 'SVM':
        return 'Support Vector Machines'
    else:
        return None


q1 = get_question_df(1)

q1[q1.algorithm=='LGBM'].head()

def plot_q1_accuracy(algorithm, ax):
    plotdata = q1[q1.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.lineplot(ax=ax, data=plotdata,
                      x="train_set_size", y="performance",
                      hue="feature_set",
                      style='seed',
                      marker="o")
    ax.set_xscale('log')
    ax.set_title(get_algorithm_name(algorithm))
    ax.set_xlabel('Training Set Size [log scale]')
    ax.set_ylabel('Accuracy')
    ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
    ax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
    
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
plot_q1_accuracy('DT', axes[0,0])
plot_q1_accuracy('RF', axes[0,1])
plot_q1_accuracy('LGBM', axes[1,0])
plot_q1_accuracy('SVM', axes[1,1])

pd.DataFrame([get_corr_coeff(q1, 'train_set_size', 'performance', x)
              for x in list(q1.algorithm.unique())])

accuracy_sens_df = get_sensitivity_df(q1, [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400], 2)
accuracy_sens_df.head(30).head()

def plot_q1_sensitivity(algorithm, ax):
    plotdata = accuracy_sens_df[accuracy_sens_df.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.lineplot(ax=ax, data=plotdata,
                      x="from_train_set_size", y="sensitivity",
                      hue="feature_set",
                      style='seed',
                      marker="o")
    ax.set_xscale('log')
    ax.set_title(get_algorithm_name(algorithm))
    ax.set_xlabel('Training Set Size [log scale]')
    ax.set_ylabel('Absolute Accuracy Change When Increasing Dataset Size x2')
    
    ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])

fig, axes = plt.subplots(2, 2, figsize=(18, 15))
plot_q1_sensitivity('DT', axes[0,0])
plot_q1_sensitivity('RF', axes[0,1])
plot_q1_sensitivity('LGBM', axes[1,0])
plot_q1_sensitivity('SVM', axes[1,1])    


def regplot_q1_sensitivity(algorithm, ax):
    plotdata = accuracy_sens_df[accuracy_sens_df.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.regplot(ax=ax, data=plotdata,
                      x="from_train_set_size", y="sensitivity")
    ax.set_xscale('log')
    ax.set_title(get_algorithm_name(algorithm))
    ax.set_xlabel('Training Set Size [log scale]')
    ax.set_ylabel('Absolute Accuracy Change When Increasing Dataset Size x2')
    
    ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.axhline(y=0.0, color='black', linestyle='--')
    
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
regplot_q1_sensitivity('DT', axes[0,0])
regplot_q1_sensitivity('RF', axes[0,1])
regplot_q1_sensitivity('LGBM', axes[1,0])
regplot_q1_sensitivity('SVM', axes[1,1])

q1[q1.performance > 0.93]

2 = get_question_df(2)

q2[q2.algorithm=='SVM'].head()


def plot_q2_performance(algorithm, lineax, boxax):
    plotdata = q2[q2.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.lineplot(ax=lineax, data=plotdata,
                      x="train_set_size", y="performance",
                      hue="feature_set",
                      style='seed',
                      marker="o")
    lineax.set_xscale('log')
    lineax.set_title(get_algorithm_name(algorithm))
    lineax.set_ylabel('Recall')
    lineax.set_xlabel('Training Set Size [log scale]')
    lineax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
    lineax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
    
    sns.boxplot(ax=boxax, x="feature_set", y="other_info",
            data=plotdata)
    boxax.set_title(get_algorithm_name(algorithm))
    boxax.set_ylabel('False Positive Rate')
    boxax.set_xlabel('Feature Set')
    
fig, axes = plt.subplots(4, 2, figsize=(18, 30))
plot_q2_performance('DT', axes[0,0], axes[0,1])
plot_q2_performance('RF', axes[1,0], axes[1,1])
plot_q2_performance('LGBM', axes[2,0], axes[2,1])
plot_q2_performance('SVM', axes[3,0], axes[3,1])


pd.DataFrame([get_corr_coeff(q2, 'train_set_size', 'performance', x)
              for x in list(q2.algorithm.unique())])


real_perf_sens_df = get_sensitivity_df(q2, [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400], 2)
real_perf_sens_df.head(30).head()


def plot_q2_sensitivity(algorithm, ax):
    plotdata = real_perf_sens_df[real_perf_sens_df.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.lineplot(ax=ax, data=plotdata,
                      x="from_train_set_size", y="sensitivity",
                      hue="feature_set",
                      style='seed',
                      marker="o")
    ax.set_xscale('log')    
    ax.set_title(get_algorithm_name(algorithm))
    ax.set_xlabel('Training Set Size [log scale]')
    ax.set_ylabel('Absolute Recall Change When Increasing Dataset Size x2')
    
    
    ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
plot_q2_sensitivity('DT', axes[0,0])
plot_q2_sensitivity('RF', axes[0,1])
plot_q2_sensitivity('LGBM', axes[1,0])
plot_q2_sensitivity('SVM', axes[1,1])


def regplot_q2_sensitivity(algorithm, ax):
    plotdata = real_perf_sens_df[real_perf_sens_df.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.regplot(ax=ax, data=plotdata,
                      x="from_train_set_size", y="sensitivity")
    ax.set_xscale('log')
    ax.set_title(get_algorithm_name(algorithm))
    ax.set_xlabel('Training Set Size [log scale]')
    ax.set_ylabel('Absolute Recall Change When Increasing Dataset Size x2')
    
    ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.set_xticklabels(labels=[200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
    ax.axhline(y=0.0, color='black', linestyle='--')
    
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
regplot_q2_sensitivity('DT', axes[0,0])
regplot_q2_sensitivity('RF', axes[0,1])
regplot_q2_sensitivity('LGBM', axes[1,0])
regplot_q2_sensitivity('SVM', axes[1,1])


q3 = get_question_df(3).copy()
q3['ratio_benign'] = q3.test_set_ratio.apply(lambda x: int(x.split(':')[1]))


def plot_q3_performance(algorithm, lineax, boxax):
    plotdata = q3[q3.algorithm==algorithm]
    sns.set(rc={'figure.figsize':(10,8)})
    sns.lineplot(ax=lineax, data=plotdata,
                      x="test_set_ratio", y="performance",
                      hue="feature_set",
                      style='seed',
                      marker="o")
    lineax.set_title(get_algorithm_name(algorithm))
    lineax.set_xlabel('Testing Set Malware Ratio [malware:benignware]')
    lineax.set_ylabel('Recall')
    
    sns.boxplot(ax=boxax, x="feature_set", y="other_info",
            data=plotdata)
    boxax.set_title(get_algorithm_name(algorithm))
    boxax.set_ylabel('False Positive Rate')
    boxax.set_xlabel('Feature Set')

fig, axes = plt.subplots(4, 2, figsize=(18, 30))
plot_q3_performance('DT', axes[0,0], axes[0,1])
plot_q3_performance('RF', axes[1,0], axes[1,1])
plot_q3_performance('LGBM', axes[2,0], axes[2,1])
plot_q3_performance('SVM', axes[3,0], axes[3,1])

