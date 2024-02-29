import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit, train_test_split, cross_val_score, cross_val_predict
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.stats import uniform, randint
import shap
import warnings
warnings.filterwarnings("ignore")

np.int=int
df_file = pd.read_excel(r"C:\Users\Госпожа Виктория\Desktop\ДАта фор МО.xlsx")
df_file.to_csv(r"C:\Users\Госпожа Виктория\Desktop\ДАта фор МО.csv", index=False)

def get_column_reference(row):
    return pd.Series([row['Fingerprints']])

references = df_file.apply(get_column_reference, axis=1)
df = pd.DataFrame(references)
df_references = df.values.tolist(columns=['Fingerprints'])

y = df_file.loc[:, ['Citrus']]
x = df_file.loc[:, 'FP1':'FP2048']
x = x.astype(int)

           
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=315, test_size=450-315,
                                                    random_state=10)
#Пустые значения Citrus медианными значениями
imputer = SimpleImputer(strategy='median') #или убрать 
y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1)).flatten()
#print(y_train_imputed)
#print(y_test_imputed)
def calculate_accuracy(x_test, y_test_imputed):
    regr_plot = LogisticRegression(max_iter=1000)
    regr_plot.fit(x_test, y_test_imputed)
    
    y_pred = regr_plot.predict(x_test)
    
    threshold = 0.5
    y_pred_binary = (y_pred>threshold).astype(int)
    accuracy = accuracy_score(y_test_imputed, y_pred_binary)
    skplt.metrics.plot_confusion_matrix(y_test_imputed, y_pred_binary, normalize=True)
    plt.show()

    return accuracy

result_accuracy1 = calculate_accuracy(x_test, y_test_imputed)
result_accuracy2 = calculate_accuracy(x_train, y_train_imputed)
print(f'Accuracy_test: {result_accuracy1:.4f}')
print(f'Accuracy_train: {result_accuracy2:.4f}')

def calculate_f1_score(x_test, y_test_imputed):
    regr_plot = LogisticRegression(max_iter=1000)
    regr_plot.fit(x_test, y_test_imputed)
    
    y_pred = regr_plot.predict(x_test)
    f1 = f1_score(y_test_imputed, y_pred)
    
    return f1

result_f1_score_test = calculate_f1_score(x_test, y_test_imputed)
result_f1_score_train = calculate_f1_score(x_train, y_train_imputed)
print(f'F1 Score_test: {result_f1_score_test:.4f}')
print(f'F1 Score_train: {result_f1_score_train:.4f}')

def calculate_roc_auc_score(x_test, y_test_imputed):
    regr_plot = LogisticRegression(max_iter=1000)
    regr_plot.fit(x_test, y_test_imputed)
    
    y_prob = regr_plot.predict_proba(x_test)[:, 1].ravel()
    roc_auc = roc_auc_score(y_test_imputed, y_prob)
    
    fpr, tpr, _ = roc_curve(y_test_imputed, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return roc_auc

result_roc_auc_score_test = calculate_roc_auc_score(x_test, y_test_imputed)
result_roc_auc_score_train = calculate_roc_auc_score(x_train, y_train_imputed)
print(f'ROC AUC Score_test: {result_roc_auc_score_test:.4f}')
print(f'ROC AUC Score_train: {result_roc_auc_score_train:.4f}')

'''def shap_interpret(model, data):

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(data)

    return shap_values'''


#DecisionTree
clf = DecisionTreeClassifier()
param_tree = {
    'criterion': Categorical(['gini', 'entropy']),
    'splitter': Categorical(['best', 'random']),
    'max_depth': (1, 30),
    'min_samples_split': Real(0.01, 0.6),
    'min_samples_leaf': Real(0.01, 0.4),
    'max_features': Categorical(['sqrt', 'log2', None])
}

tree_search = BayesSearchCV(clf, param_tree, cv=5, n_iter=100, random_state=10)
best_tree = tree_search.fit(x_train, y_train_imputed)  # find best hyperparameters  
best_params_tree = best_tree.best_params_ # remember the best hyperparameters
best_clf = DecisionTreeClassifier(**best_params_tree)
best_clf.fit(x_train, y_train_imputed)\
                      
print('Best tree params:', best_params_tree)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_tree = cross_val_score(best_clf, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_tree = cross_val_score(best_clf, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_tree_test = cross_val_predict(best_clf, x_test, y_test_imputed, cv=cv)
y_pred_tree_train = cross_val_predict(best_clf, x_train, y_train_imputed, cv=cv)

#best_tree_shap_test = shap_interpret(best_clf, x_test)
#best_tree_shap_train = shap_interpret(best_clf, x_train)
#shap.summary_plot(best_tree_shap_test, x_test)
#shap.summary_plot(best_tree_shap_train, x_train)

print("f1_tree_test(CV):", crocc_val_f1_test_tree.mean())
print("f1_tree_train(CV):", crocc_val_f1_train_tree.mean())
print("f1_tree_test:", metrics.f1_score(y_test_imputed, y_pred_tree_test))
print("f1_tree_train:", metrics.f1_score(y_train_imputed, y_pred_tree_train))

#RandomForest
rf = RandomForestClassifier()
param_forest = {
    'n_estimators': (20, 500),
    'criterion':Categorical(['gini', 'entropy', 'log_loss']),
    'max_depth':(1, 12),
    'min_samples_leaf': Integer(1, 16),
    'oob_score': Categorical([True])
    }

forest_search = BayesSearchCV(rf, param_forest, cv=5, random_state=10)
best_forest = forest_search.fit(x_train, y_train_imputed)
best_params_forest = best_forest.best_params_
best_rf = RandomForestClassifier(**best_params_forest)
best_rf.fit(x_train, y_train_imputed)

print('Best forest params:', best_params_forest)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_forest = cross_val_score(best_rf, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_forest = cross_val_score(best_rf, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_forest_test = cross_val_predict(best_rf, x_test, y_test_imputed, cv=cv)
y_pred_forest_train = cross_val_predict(best_rf, x_train, y_train_imputed, cv=cv)

#best_forest_shap_test = shap_interpret(best_rf, x_test)
#best_forest_shap_train = shap_interpret(best_rf, x_train)
#shap.summary_plot(best_forest_shap_test, x_test)
#shap.summary_plot(best_forest_shap_train, x_train)

print("f1_forest_test(CV):", crocc_val_f1_test_forest.mean())
print("f1_forest_train(CV):", crocc_val_f1_train_forest.mean())
print("F1_forest_test:", metrics.f1_score(y_test_imputed, y_pred_forest_test))
print("F1_forest_train:", metrics.f1_score(y_train_imputed, y_pred_forest_train))


#XGBoost
xgb_class = XGBClassifier(verbosity=0)
params_xgb = {
    'colsample_bytree': (0.3, 0.7),
    'gamma': (0, 0.5),
    'learning_rate': (0.03, 0.3), 
    'max_depth': (2, 6),
    'n_estimators': (100, 150), 
    'subsample': (0.6, 1.0)
}

xgb_search = BayesSearchCV(xgb_class, params_xgb, cv=5, random_state=10)
best_xgb = xgb_search.fit(x_train, y_train_imputed)
best_params_xgb = best_xgb.best_params_
best_xgb_class = XGBClassifier(**best_params_xgb, verbosity=0)
best_xgb_class.fit(x_train, y_train_imputed)

print('Best XGB params:', best_params_xgb)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_XGB = cross_val_score(best_xgb_class, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_XGB = cross_val_score(best_xgb_class, x_train, y_train_imputed,  cv=cv, scoring='f1')

xgb_predict_test = cross_val_predict(best_xgb_class, x_test, y_test_imputed, cv=cv)
xgb_predict_train = cross_val_predict(best_xgb_class, x_train, y_train_imputed, cv=cv)

#best_XGB_shap_test = shap_interpret(best_xgb_class, x_test)
#best_XGB_shap_train = shap_interpret(best_xgb_class, x_train)
#shap.summary_plot(best_XGB_shap_test, x_test)
#shap.summary_plot(best_XGB_shap_train, x_train)

print("f1_XGB_test(CV):", crocc_val_f1_test_XGB.mean())
print("f1_XGB_train(CV):", crocc_val_f1_train_XGB.mean())
print("F1_XGB_test:", metrics.f1_score(y_test_imputed, xgb_predict_test))
print("F1_XGB_train:", metrics.f1_score(y_train_imputed, xgb_predict_train))


#CatBoost
cat_model=CatBoostClassifier(verbose=False)
params_cat = {
    'iterations': (10, 500),
    'depth': (1, 10),
    'learning_rate': (0.001, 0.5),
    'random_strength': (1e-9, 10)
    }
cat_search = BayesSearchCV(cat_model, params_cat, cv=5, random_state=10)
best_cat = cat_search.fit(x_train, y_train_imputed)
best_params_cat = best_cat.best_params_
best_cat_model = CatBoostClassifier(**best_params_cat, verbose=False)
best_cat_model.fit(x_train, y_train_imputed)

print('Best CAT params:', best_params_cat)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_cat = cross_val_score(best_cat_model, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_cat = cross_val_score(best_cat_model, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_cat_test = cross_val_predict(best_cat_model, x_test, y_test_imputed, cv=cv)
y_pred_cat_train = cross_val_predict(best_cat_model, x_train, y_train_imputed, cv=cv)

#best_cat_shap_test = shap_interpret(best_cat_model, x_test)
#best_cat_shap_train = shap_interpret(best_cat_model, x_train)
#shap.summary_plot(best_cat_shap_test, x_test)
#shap.summary_plot(best_cat_shap_train, x_train)

print("f1_cat_test(CV):", crocc_val_f1_test_cat.mean())
print("f1_cat_train(CV):", crocc_val_f1_train_cat.mean())
print("F1_cat_test:", metrics.f1_score(y_test_imputed, y_pred_cat_test))
print("F1_cat_train:", metrics.f1_score(y_train_imputed, y_pred_cat_train))


#GradientBoosting
GBM_model = GradientBoostingClassifier()
params_GBM = {
    'n_estimators': (50, 500),
    'max_depth': (3, 12),
    'criterion': ['friedman_mse'],
    'min_samples_leaf': Integer(1, 16),
    'min_samples_split': Integer(2, 16),
    'learning_rate': Real(0.01, 0.5),
    'subsample': Real(0.5, 1.0)
}
gbm_search = BayesSearchCV(GBM_model, params_GBM, cv=5, random_state=10)
GBM_best = gbm_search.fit(x_train, y_train_imputed)
best_params_gbm = GBM_best.best_params_

GBM_best_model = GradientBoostingClassifier(**best_params_gbm)
GBM_best_model.fit(x_train, y_train_imputed)

print('Best GBM params:', GBM_best_model)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_gbm = cross_val_score(GBM_best_model, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_gbm = cross_val_score(GBM_best_model, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_GBM_test = cross_val_predict(GBM_best_model, x_test, y_test_imputed, cv=cv)
y_pred_GBM_train = cross_val_predict(GBM_best_model, x_train, y_train_imputed, cv=cv)

#best_GBM_shap_test = shap_interpret(GBM_best_model, x_test)
#best_GBM_shap_train = shap_interpret(GBM_best_model, x_train)
#shap.summary_plot(best_GBM_shap_test, x_test)
#shap.summary_plot(best_GBM_shap_train, x_train)

print("f1_GBM_test(CV):", crocc_val_f1_test_gbm.mean())
print("f1_GBM_train(CV):", crocc_val_f1_train_gbm.mean())
print("F1_GBM_test:", metrics.f1_score(y_test_imputed, y_pred_GBM_test))
print("F1_GBM_train:", metrics.f1_score(y_train_imputed, y_pred_GBM_train))


#K-NN
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn_params = {
    'n_neighbors' : list(range(2,21)) , 
    'algorithm' : ['auto','ball_tree','kd_tree','brute'],
    'p': [1, 2],
    'weights': ['uniform', 'distance'],
    'leaf_size': list(range(10, 51, 10))
}
knn_search = BayesSearchCV(knn, knn_params, n_iter=30, random_state=10)
knn_best = knn_search.fit(x_train, y_train_imputed)
best_params_knn = knn_best.best_params_
KNN_best_model = KNeighborsClassifier(**best_params_knn)
KNN_best_model.fit(x_train, y_train_imputed)

print('Best KNN params:', KNN_best_model)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_knn = cross_val_score(KNN_best_model, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_knn = cross_val_score(KNN_best_model, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_knn_test = cross_val_predict(KNN_best_model, x_test, y_test_imputed, cv=cv)
y_pred_knn_train = cross_val_predict(KNN_best_model, x_train, y_train_imputed, cv=cv)

#best_knn_shap_test = shap_interpret(KNN_best_model, x_test)
#best_knn_shap_train = shap_interpret(KNN_best_model, x_train)
#shap.summary_plot(best_knn_shap_test, x_test)
#shap.summary_plot(best_knn_shap_train, x_train)

print("f1_KNN_test(CV):", crocc_val_f1_test_knn.mean())
print("f1_KNN_train(CV):", crocc_val_f1_train_knn.mean())
print("F1_knn_test:", metrics.f1_score(y_test_imputed, y_pred_knn_test))
print("F1_knn_train:", metrics.f1_score(y_train_imputed, y_pred_knn_train))


#SVM
svm_model = SVC()
params_SVC = {
    'C': Real(0.1, 10),  
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'gamma': Real(0.001, 1, prior='log-uniform'),
    'shrinking': Categorical([True, False]), 
    'class_weight': [None, 'balanced'], 
    'probability': [True, False], 
    'tol': Real(1e-5, 1e-1, prior='log-uniform')
}
svm_search = BayesSearchCV(svm_model, params_SVC, n_iter=30, random_state=10)
svm_best = svm_search.fit(x_train, y_train_imputed)
best_params_svm = svm_best.best_params_
svm_best_model = SVC(**best_params_svm)
svm_best_model.fit(x_train, y_train_imputed)

print('Best SVM params:', svm_best_model)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_svm = cross_val_score(svm_best_model, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_svm = cross_val_score(svm_best_model, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_svm_test = cross_val_predict(svm_best_model, x_test, y_test_imputed, cv=cv)
y_pred_svm_train = cross_val_predict(svm_best_model, x_train, y_train_imputed, cv=cv)

#best_svm_shap_test = shap_interpret(svm_best_model, x_test)
#best_svm_shap_train = shap_interpret(svm_best_model, x_train)
#shap.summary_plot(best_svm_shap_test, x_test)
#shap.summary_plot(best_svm_shap_train, x_train)

print("f1_SVM_test(CV):", crocc_val_f1_test_svm.mean())
print("f1_SVM_train(CV):", crocc_val_f1_train_svm.mean())
print("F1_svm_test:", metrics.f1_score(y_test_imputed, y_pred_svm_test))
print("F1_svm_train:", metrics.f1_score(y_train_imputed, y_pred_svm_train))


#MLP
clf_mlp = MLPClassifier()
params_mlp = {
    'hidden_layer_sizes': Integer(12, 150),  
    'activation': Categorical(['logistic', 'tanh', 'relu']),
    'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
    'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
    'max_iter': Integer(50, 500)
}

mlp_search = BayesSearchCV(clf_mlp, params_mlp, n_iter=100, random_state=10)
mlp_best = mlp_search.fit(x_train, y_train_imputed)
best_params_mlp = mlp_best.best_params_
mlp_best_model = MLPClassifier(**best_params_mlp)
mlp_best_model.fit(x_train, y_train_imputed)

print('Best MLP params:', mlp_best_model)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
crocc_val_f1_test_mlp = cross_val_score(mlp_best_model, x_test, y_test_imputed,  cv=cv, scoring='f1')
crocc_val_f1_train_mlp = cross_val_score(mlp_best_model, x_train, y_train_imputed,  cv=cv, scoring='f1')

y_pred_mlp_test = cross_val_predict(mlp_best_model, x_test, y_test_imputed, cv=cv)
y_pred_mlp_train = cross_val_predict(mlp_best_model, x_train, y_train_imputed, cv=cv)

#best_mlp_shap_test = shap_interpret(mlp_best_model, x_test)
#best_mlp_shap_train = shap_interpret(mlp_best_model, x_train)
#shap.summary_plot(best_mlp_shap_test, x_test)
#shap.summary_plot(best_mlp_shap_train, x_train)

print("f1_MLP_test(CV):", crocc_val_f1_test_mlp.mean())
print("f1_MLP_train(CV):", crocc_val_f1_train_mlp.mean())
print("F1_mlp_test:", metrics.f1_score(y_test_imputed, y_pred_mlp_test))
print("F1_mlp_train:", metrics.f1_score(y_train_imputed, y_pred_mlp_train))

'''metrics_data = {
    'Metric': ['Accuracy_test','Accuracy_train', 'F1 Score_test','F1 Score_train', 'ROC AUC Score_test', 'ROC AUC Score_train',
               "Accuracy_tree:", "f1_tree:", "ROC AUC Score_tree:",
               "Accuracy_forest:", "F1_forest:", "ROC AUC Score_forest:",
               "Accuracy_XGB:", "F1_XGB:", "ROC AUC Score_XGB:"],
    'Value': [result_accuracy1, result_accuracy2, result_f1_score_test, result_f1_score_train, result_roc_auc_score_test, result_roc_auc_score_train,
              metrics.accuracy_score(y_test_imputed, y_pred_tree), metrics.f1_score(y_test_imputed, y_pred_tree), metrics.roc_auc_score(y_test_imputed, y_pred_tree),
              metrics.accuracy_score(y_test_imputed, y_pred_forest), metrics.f1_score(y_test_imputed, y_pred_forest), metrics.roc_auc_score(y_test_imputed, y_pred_forest),
              metrics.accuracy_score(y_test_imputed, predictions), metrics.f1_score(y_test_imputed, predictions), metrics.roc_auc_score(y_test_imputed, predictions)]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('Results.csv', index=False)'''

#CROSS-VALIDATION VISUALISATION
cross_values = [
    crocc_val_f1_train_tree.mean(),
    crocc_val_f1_train_forest.mean(),
    crocc_val_f1_train_XGB.mean(),
    crocc_val_f1_train_cat.mean(),
    crocc_val_f1_train_gbm.mean(),
    crocc_val_f1_train_knn.mean(),
    crocc_val_f1_train_svm.mean()
]

errors_cross = np.std(cross_values)
cross_labels = [
    'crocc_val_tree',
    'crocc_val_forest',
    'crocc_val_XGB',
    'crocc_val_cat',
    'crocc_val_GBM',
    'crocc_val_KNN',
    'crocc_val_SVM'
]
cmap = cm.get_cmap('viridis')
fig, ax = plt.subplots()
ax.bar(cross_labels, cross_values, yerr=errors_cross, capsize=5, color=cmap(np.arange(len(cross_values))), alpha=0.7)
ax.set_ylabel('Среднее значение F1-меры')
ax.set_title('Сравнение Эффективности Моделей')

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Индекс', rotation=270, labelpad=15)


plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
    
    
#OUTPUT VISUALISATION
predict_values = [
    metrics.f1_score(y_train_imputed, y_pred_tree_train),
    metrics.f1_score(y_train_imputed, y_pred_forest_train),
    metrics.f1_score(y_train_imputed, xgb_predict_train),
    metrics.f1_score(y_train_imputed, y_pred_cat_train),
    metrics.f1_score(y_train_imputed, y_pred_GBM_train),
    metrics.f1_score(y_train_imputed, y_pred_knn_train),
    metrics.f1_score(y_train_imputed, y_pred_svm_train)
]

errors_predict = np.std(predict_values)
predict_labels = [
    'train_tree',
    'train_forest',
    'train_XGB',
    'train_cat',
    'train_GBM',
    'train_KNN',
    'train_SVM'
]
cmap = cm.get_cmap('tab10')
fig, ax = plt.subplots()
ax.bar(predict_labels, predict_values, yerr=errors_predict, capsize=5, color=cmap(np.arange(len(predict_values))), alpha=0.7)
ax.set_ylabel('Среднее значение F1-меры')
ax.set_title('Сравнение Производительности Моделей')

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Индекс', rotation=270, labelpad=15)


plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

