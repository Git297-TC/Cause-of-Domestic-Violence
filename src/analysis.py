# Analysis script
#def analyze():
    #print('Data analysis')
import utils

def eda(dataframe):
    dataframe.describe()

    dataframe.mean()

    dataframe.median()

    dataframe.mode()

    dataframe_age_round = dataframe.groupby(utils.pandas.cut(dataframe['Age'], [0,15,30,45,60])).sum()
    dataframe_age_round

    # plot using saborn
    utils.matplotlib.pyplot.figure(figsize=(5,5))
    utils.seaborn.histplot(dataframe['Age'], bins=7, kde=True, color="skyblue")
    # add vertical lines for mean, median and mode
    utils.matplotlib.pyplot.axvline(dataframe['Age'].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
    utils.matplotlib.pyplot.axvline(dataframe['Age'].median(), color="blue", linestyle="-", linewidth=2, label="Median")
    utils.matplotlib.pyplot.axvline(dataframe['Age'].mode()[0], color="green", linestyle="-", linewidth=2, label="Mode")
    utils.matplotlib.pyplot.legend()
    utils.matplotlib.pyplot.title("Mean, Median & Mode")
    utils.matplotlib.pyplot.show()

    dataframe['Age'].plot.kde()

    dataframe['Education'].plot.kde()

    dataframe['Employment'].plot.kde()

    dataframe['Marital status'].plot.kde()

    dataframe['Violence'].plot.kde()

    dataframe_violence = dataframe[dataframe['Violence'] == 1]
    dataframe_violence

    dataframe_violence['Education'].value_counts()

    dataframe_violence['Employment'].value_counts()

    dataframe_violence['Income'].value_counts()

    dataframe_violence['Marital status'].value_counts()

    dataframe_corr = dataframe[['Age',	'Education',	'Employment',	'Income',	'Marital status',	'Violence']]
    dataframe_corr.corr()

    utils.seaborn.heatmap(dataframe_corr.corr(), vmin=-1, vmax=1, annot=True)

    return dataframe

model_algorithm_list = {
    'randomForest': utils.sklearn.ensemble.RandomForestClassifier(random_state=50),
    'logisticRegression': utils.sklearn.linear_model.LogisticRegression(max_iter=200),
    'decistionTree': utils.sklearn.tree.DecisionTreeClassifier(random_state=50),
    'gradientBoosting': utils.sklearn.ensemble.GradientBoostingClassifier(random_state=50),
    'supportVectorMachine_sigmoid': utils.sklearn.svm.SVC(kernel='sigmoid', random_state=50),
    'supportVectorMachine_rbf': utils.sklearn.svm.SVC(kernel='rbf', random_state=50),
    'supportVectorMachine_poly': utils.sklearn.svm.SVC(kernel='poly', random_state=50),
    'supportVectorMachine_linear': utils.sklearn.svm.SVC(kernel='linear', random_state=50)
    
}

sampling_method_list = {
    'no_sampling': None,
    'smote': utils.imblearn.over_sampling.SMOTE(random_state=50)
}

def sampling(method, features, target):
    features_resampled, target_resampled = method.fit_resample(features, target) if method else (features, target)
    return features_resampled, target_resampled

def model_result(model, features_test, target_test):
    #accuracy = model.score(features_test, target_test)
    target_predicted   = model.predict(features_test)
    #precision = utils.sklearn.metrics.precision_score(target_test, target_predicted)
    #precision_r, recall, thresholds = utils.sklearn.metrics.precision_recall_curve(target_test, target_predicted)
    #recall = recall_score(target_test, target_predicted)
    report = utils.sklearn.metrics.classification_report(target_test, target_predicted)
    return target_predicted, report

def processing(sampling_method_key, sampling_method, model_algorithm_key, model_algorithm, features_train, features_test, target_train, target_test):
    features_train_sampled, target_train_sampled = sampling(sampling_method, features_train, target_train) if sampling_method else (features_train, target_train)
    model = model_algorithm.fit(features_train_sampled, target_train_sampled)
    target_predicted, report = model_result(model, features_test, target_test)
    return sampling_method_key, model_algorithm_key, model, target_predicted, report

def result_show(model, report, sampling_method, model_algorithm, features_test, target_test ):
    print(report[sampling_method][model_algorithm])
    utils.sklearn.metrics.ConfusionMatrixDisplay.from_estimator(model[sampling_method][model_algorithm], features_test, target_test)
    utils.sklearn.metrics.RocCurveDisplay.from_estimator(model[sampling_method][model_algorithm], features_test, target_test)
    utils.sklearn.metrics.PrecisionRecallDisplay.from_estimator(model[sampling_method][model_algorithm], features_test, target_test)

def analysis(dataframe):
    
    dataframe = eda(dataframe)

    features = dataframe[['Age', 'Education', 'Employment', 'Income', 'Marital status']].values
    target = dataframe['Violence'].values

    features_train, features_test, target_train, target_test = utils.sklearn.model_selection.train_test_split(features, target, test_size=0.2, random_state=50)

    results = utils.joblib.Parallel(n_jobs=-1)(
        utils.joblib.delayed(processing)(sampling_method_key, sampling_method, model_algorithm_key, model_algorithm, features_train, features_test, target_train, target_test)
        for sampling_method_key, sampling_method in sampling_method_list.items()
        for model_algorithm_key, model_algorithm in model_algorithm_list.items()
    )

    model = {}
    target_predicted = {}
    report = {}
    for sampling_method_key, model_algorithm_key, results_model, results_target_predicted, results_report in results:
        if sampling_method_key not in model:
            model[sampling_method_key] = {}
            target_predicted[sampling_method_key] = {}
            report[sampling_method_key] = {}
        model[sampling_method_key][model_algorithm_key] = results_model
        target_predicted[sampling_method_key][model_algorithm_key] = results_target_predicted
        report[sampling_method_key][model_algorithm_key] = results_report

    

    return dataframe