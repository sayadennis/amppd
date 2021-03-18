import os
import sys
import numpy as np
import pandas as pd

# classifiers 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier

# tools for cross-validation 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# dimensionality reduction 
from sklearn.decomposition import NMF

# visualization
import matplotlib.pyplot as plt


# Function that will convert inputs into compatible format 
def confirm_numpy(X, y):
    if type(X) != np.ndarray:
        X = X.to_numpy()
    if type(y) != np.ndarray:
        y = y.to_numpy()
    y = y.ravel() # convert to 1D array 
    return X, y


# Logistic regression cross-validation 
def lrm_cv(X_train, y_train, lrm_Cs=[1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3], seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    lrCV = LogisticRegressionCV(Cs=lrm_Cs, class_weight="balanced", n_jobs=8, max_iter=1000,
        scoring="roc_auc", cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True), refit=True
    )
    lrCV.fit(X_train, y_train)
    opt_C = lrCV.C_[0]
    opt_mean_score = np.mean(lrCV.scores_[1][:,np.where(lrm_Cs == opt_C)])
    return opt_C, opt_mean_score, lrCV


# LASSO cross-validation
def lasso_cv(X_train, y_train, eps=1e-2, n_alphas=20, seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    lasscv = LassoCV(
        eps=eps, n_alphas=n_alphas, max_iter=1000, n_jobs=8,
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True), random_state=seed
    ).fit(X_train, y_train)
    opt_alpha = lasscv.alpha_
    # get the cross-validation ROC when trained with this optimal alpha
    lasso = Lasso(alpha=opt_alpha, max_iter=1000, random_state=seed)
    lasso.fit(X_train, y_train)
    scores = cross_val_score(
        lasso, X_train, y_train, scoring="roc_auc", 
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    )
    opt_mean_score = np.mean(scores)
    return opt_alpha, opt_mean_score, lasso


# Elastic Net cross-validation 
def elasticnet_cv(X_train, y_train, l1_ratio=np.arange(0.1, 1., step=0.1), eps=1e-2, n_aplhas=20, seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    elastcv = ElasticNetCV(
        l1_ratio=l1_ratio, eps=1e-3, n_alphas=n_aplhas, max_iter=1000, n_jobs=8,
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True), random_state=seed
    ).fit(X_train, y_train)
    opt_l1 = elastcv.l1_ratio_
    opt_alpha = elastcv.alpha_
    elast = ElasticNet(alpha=opt_alpha, l1_ratio=opt_l1, random_state=seed)
    elast.fit(X_train, y_train)
    scores = cross_val_score(
        elast, X_train, y_train, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    )
    opt_mean_score = np.mean(scores)
    return {"L1" : opt_l1, "alpha" : opt_alpha}, opt_mean_score, elast


# Support vector machine cross-validation 
svm_params = {
    "kernel" : ["linear", "rbf"], 
    "C" : [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],  # 1e-5, 1e-4, 
    "gamma" : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3] # 1e-5, 1e-4, 
}

def svm_cv(X_train, y_train, svm_params=svm_params, seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    gsCV = GridSearchCV(
        SVC(class_weight="balanced", max_iter=1000, probability=True, random_state=seed),  #, decision_function_shape="ovr"
        param_grid=svm_params, n_jobs=8, scoring="roc_auc", refit=True,
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    )
    gsCV.fit(X_train, y_train)
    opt_params = gsCV.best_params_
    opt_mean_score = np.mean(
        gsCV.cv_results_["mean_test_score"][
            (gsCV.cv_results_["param_kernel"] == opt_params["kernel"]) & 
            (gsCV.cv_results_["param_C"] == opt_params["C"]) & 
            (gsCV.cv_results_["param_gamma"] == opt_params["gamma"])
        ]
    )
    return opt_params, opt_mean_score, gsCV


# Random Forest cross-validation
rf_params = {
    # "criterion" : ["gini", "entropy"],
    "max_depth" : [5, 10, 25, 50, 75], # or could set min_samples_split 
    "min_samples_leaf" : [2, 4, 6, 8, 10, 15, 20]
}

def rf_cv(X_train, y_train, rf_params=rf_params, seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    gsRF = GridSearchCV(
        RandomForestClassifier(n_estimators=100, criterion="gini", max_features="auto", class_weight="balanced", n_jobs=8, random_state=seed),
        param_grid=rf_params, n_jobs=8, scoring="balanced_accuracy", refit=True,
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    )
    gsRF.fit(X_train, y_train)
    opt_params = gsRF.best_params_
    opt_mean_score = np.mean(
        gsRF.cv_results_["mean_test_score"][
            (gsRF.cv_results_["param_max_depth"] == opt_params["max_depth"]) &
            (gsRF.cv_results_["param_min_samples_leaf"] == opt_params["min_samples_leaf"])
        ]
    )
    return opt_params, opt_mean_score, gsRF


# Gradient Boosting cross-validation
gb_params = {
    "loss" : ["deviance", "exponential"],
    "min_samples_split" : [2, 6, 10, 15, 20],
    "max_depth" : [5, 10, 25, 50, 75]
}

def gb_cv(X_train, y_train, gb_params=gb_params, seed=0):
    X_train, y_train = confirm_numpy(X_train, y_train)
    gsGB = GridSearchCV(
        GradientBoostingClassifier(subsample=0.8, random_state=seed),
        param_grid=gb_params, n_jobs=8, scoring="balanced_accuracy", refit=True,
        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    )
    gsGB.fit(X_train, y_train)
    opt_params = gsGB.best_params_
    opt_mean_score = np.mean(
        gsGB.cv_results_["mean_test_score"][
            (gsGB.cv_results_["param_loss"] == opt_params["loss"]) &
            (gsGB.cv_results_["param_min_samples_split"] == opt_params["min_samples_split"]) &
            (gsGB.cv_results_["param_max_depth"] == opt_params["max_depth"])
        ]
    )
    return opt_params, opt_mean_score, gsGB

## Function that will take classifier and evaluate on test set
def evaluate_model(clf, X_test, y_test):
    pf_dict = {} # performance dictionary
    if ((type(clf) == Lasso) | (type(clf) == ElasticNet)): # these classifiers output continuous values with .predict() method 
        y_pred = np.array(clf.predict(X_test).round(), dtype=int)
    else:
        y_pred = clf.predict(X_test)
    pf_dict["balanced_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)
    pf_dict["precision"] = metrics.precision_score(y_test, y_pred)
    pf_dict["recall"] = metrics.recall_score(y_test, y_pred)
    pf_dict["f1"] = metrics.f1_score(y_test, y_pred)
    return pf_dict


## Function that will take X_train, y_train, run all the hyperparameter tuning, and record cross-validation performance 
def record_tuning(X_train, y_train, X_test, y_test, outfn):
    tuner_dict = {
        "LRM" : lrm_cv,
        "SVM" : svm_cv,
        "LASSO" : lasso_cv,
        "ElasticNet" : elasticnet_cv,
        "RF" : rf_cv,
        "GB" : gb_cv
    }
    cv_record = pd.DataFrame(
        None, index=tuner_dict.keys(), 
        columns=["opt_params", "crossval roc/acc", "balanced_acc", "precision", "recall", "f1"]
    )
    for model_key in tuner_dict.keys():
        opt_params, opt_score, clf = tuner_dict[model_key](X_train, y_train)
        cv_record.loc[model_key]["opt_params"] = str(opt_params)
        cv_record.loc[model_key]["crossval roc/acc"] = opt_score
        pf_dict = evaluate_model(clf, X_test, y_test)
        for pf_key in pf_dict:
            cv_record.loc[model_key][pf_key] = pf_dict[pf_key]
    cv_record.to_csv(outfn, header=True, index=True)
    return


## Below is for NMF ## 

def get_F(k, X_train, y_train, X_test, y_test):
    train_size = X_train.shape[0] # rows are patients for X
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    A = X.T
    nmf = NMF(n_components=k, init="nndsvd", random_state=24)
    W = nmf.fit_transform(A)
    F = np.dot(A.T, W)
    F_train = F[:train_size, :]
    F_test = F[train_size:, :]

    return F_train, F_test, W


def coef_genes(W, genes, thres=0.01): # length of genes and rows of W should match
    if W.shape[0] != len(genes):
        return -1
    else:
        group_list = []
        for i in range(W.shape[1]): # iterate through columns of W i.e. weight vectors of each factor groups 
            coef_genes = genes[np.where(W[:,i] > thres)[0]]
            genes_final = []
            for j in range(len(coef_genes)):
                split_list = coef_genes[j].split(";") # element in list can contain multiple gene names (overlapping)
                for gene in split_list:
                    if gene in genes_final:
                        continue
                    else:
                        genes_final.append(gene)
            group_list.append([genes_final])
    return genes_final


def record_tuning_NMF(X_train, y_train, X_test, y_test, outfn, k_list=None):
    if k_list is None:
        k_list = [20, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    
    tuner_dict = {
        "LRM" : lrm_cv,
        "SVM" : svm_cv,
        "LASSO" : lasso_cv,
        "ElasticNet" : elasticnet_cv,
        "RF" : rf_cv,
        "GB" : gb_cv
    }

    cv_record = pd.DataFrame(
        None, index=np.arange(len(list(tuner_dict))*len(k_list)),
        columns=["model", "NMF_k", "opt_params", "crossval roc/acc", "balanced_acc", "precision", "recall", "f1"]
    )

    # fill in models and NMF_k by all possible combinations
    cv_record["model"] = np.repeat(list(tuner_dict.keys()), repeats=len(k_list))
    cv_record["NMF_k"] = np.repeat(np.array(k_list).reshape((-1,1)), repeats=len(list(tuner_dict.keys())), axis=1).T.ravel()

    for k in k_list:
        F_train, F_test, _ = get_F(k, X_train, y_train, X_test, y_test)
        for model_key in tuner_dict.keys():
            opt_params, opt_score, clf = tuner_dict[model_key](F_train, y_train)
            bool_loc = np.array(
                [x == model_key for x in cv_record["model"]]
            ) & np.array(
                [x == k for x in cv_record["NMF_k"]]
            )
            cv_record.iloc[bool_loc,[x == "opt_params" for x in cv_record.columns]] = str(opt_params)
            cv_record.iloc[bool_loc,[x == "crossval roc/acc" for x in cv_record.columns]] = opt_score
            pf_dict = evaluate_model(clf, F_test, y_test)
            for pf_key in pf_dict:
                cv_record.iloc[bool_loc,[x == pf_key for x in cv_record.columns]] = pf_dict[pf_key]
    cv_record.to_csv(outfn, header=True, index=True)
    return
