from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import time
import pandas as pd
from xgboost import XGBClassifier
import optuna
import os
from itertools import cycle


class NewsClassificationModel:
    OUTPUT_FOLDER = '/home/sotatek/PycharmProjects/disertation_nlp/output'
    CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_FOLDER, 'confusion_matrix')
    ROC_PATH = os.path.join(OUTPUT_FOLDER, 'roc')
    CLASSIFICATION_PATH = os.path.join(OUTPUT_FOLDER, 'classification')
    MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'model')
    MODEL_BEST_PARAMS_PATH = os.path.join(OUTPUT_FOLDER, 'best_params')

    def __init__(self, data=None):
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def pre_unbalanced_classification(self, max_numbers_sample=5000):

        print(self.data['topic'].value_counts())
        label_encoder = LabelEncoder()
        self.data['topic'] = label_encoder.fit_transform(self.data['topic'])
        self.label_encoder = label_encoder

        tfidf_vectorizer = TfidfVectorizer()
        self.X = tfidf_vectorizer.fit_transform(self.data['content'])
        self.y = self.data['topic']

        oversampled_data = []
        undersampled_data = []
        numbers_activities = self.data['topic'].value_counts()

        for label, number in numbers_activities.items():
            if number < max_numbers_sample:
                oversampled_sample = np.random.choice(self.data.index[self.y == label], size=max_numbers_sample,
                                                      replace=True)

                oversampled_data.append(self.data.loc[oversampled_sample])
            elif number > max_numbers_sample:
                undersampled_sample = np.random.choice(self.data.index[self.y == label], size=max_numbers_sample,
                                                       replace=False)

                undersampled_data.append(self.data.loc[undersampled_sample])
            else:
                oversampled_data.append(self.data[self.y == label])
                undersampled_data.append(self.data[self.y == label])
        oversampled_df = pd.concat(oversampled_data)
        undersampled_df = pd.concat(undersampled_data)
        df = pd.concat([oversampled_df, undersampled_df])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = df
        numbers_activities = self.data['topic'].value_counts()
        print(numbers_activities)

    def prepare_train_test_set(self):
        test_size = 0.3
        label_encoder = LabelEncoder()
        self.data['topic'] = label_encoder.fit_transform(self.data['topic'])
        text = self.data['content']
        label = self.data['topic']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(text, label, test_size=test_size,
                                                                                random_state=42)

    def evaluation_model(self, name_model, classifier, train_time, fine_turning=False):

        # Predict on test set
        y_pred = classifier.predict(self.X_test)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        class_names = list(self.label_encoder.classes_)
        report = metrics.classification_report(self.y_test, y_pred, target_names=class_names)
        if fine_turning is False:
            report_filename = os.path.join(NewsClassificationModel.CLASSIFICATION_PATH,
                                           f"{name_model}_classification_report.txt")
        else:
            report_filename = os.path.join(NewsClassificationModel.CLASSIFICATION_PATH,
                                           f"FT_{name_model}_classification_report.txt")
        if fine_turning is False:
            with open(report_filename, 'w') as file:
                file.write(f"{name_model} Training Time: {train_time} seconds\n\n")
                file.write(f'{name_model}, Accuracy ={accuracy}\n')
                file.write("Classification Report:\n")
                file.write(report)
        else:
            with open(report_filename, 'w') as file:
                file.write(f"Best {name_model} Training Time: {train_time} seconds\n\n")
                file.write(f'Best {name_model}, Accuracy ={accuracy}\n')
                file.write("Classification Report:\n")
                file.write(report)

        # Create a Confusion Matrix of model and save it
        labels = list(self.label_encoder.classes_)
        cm = metrics.confusion_matrix(self.y_test, y_pred)
        fig_size = max(8, len(labels) / 3)
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                    yticklabels=labels)
        plt.title(f'Confusion Matrix for {name_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save confusion matrix
        if fine_turning is False:
            cm_image_filename = os.path.join(NewsClassificationModel.CONFUSION_MATRIX_PATH,
                                             f"{name_model}_confusion_matrix.png")
        else:
            cm_image_filename = os.path.join(NewsClassificationModel.CONFUSION_MATRIX_PATH,
                                             f"FT_{name_model}_confusion_matrix.png")
        plt.savefig(cm_image_filename, bbox_inches='tight')
        plt.close()

        """ Create ROC Curve"""
        # Binarize the labels for multi-class ROC curve
        y_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        if name_model == 'SVM':
            y_score = classifier.decision_function(self.X_test)
        else:
            y_score = classifier.predict_proba(self.X_test)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= n_classes

        # Plot all ROC curves
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
                  'gray']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC of {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-Class ROC for {name_model}')
        plt.legend(loc="lower right")
        plt.tight_layout()

        # Save the ROC curve
        if fine_turning is False:
            roc_curve_filename = os.path.join(NewsClassificationModel.ROC_PATH,
                                              f"{name_model}_multi_class_roc_curve.png")
        else:
            roc_curve_filename = os.path.join(NewsClassificationModel.ROC_PATH,
                                              f"FT_{name_model}_multi_class_roc_curve.png")
        plt.savefig(roc_curve_filename)
        plt.close()
        print(f'Done with {name_model}')

    def core_models(self, name_model, clf):
        start_time = time.time()
        text_clf = clf.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        model_filename = os.path.join(NewsClassificationModel.MODEL_PATH, f'{name_model}.joblib')
        dump(text_clf, model_filename)
        print(f"Model saved as {model_filename}")
        self.evaluation_model(name_model=name_model, classifier=text_clf, train_time=train_time)

    def core_models_hyperparams(self, name_model, clf, param_grid):
        start_time = time.time()
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Save the best model
        model_filename = os.path.join(NewsClassificationModel.MODEL_PATH, f"{name_model}_best_model.joblib")
        dump(grid_search.best_estimator_, model_filename)
        print(f"Best model saved as {model_filename}")

        # Optionally, save the best parameters in a separate file
        params_filename = os.path.join(NewsClassificationModel.MODEL_BEST_PARAMS_PATH, f"{name_model}_best_params.txt")
        with open(params_filename, 'w') as file:
            file.write(str(grid_search.best_params_))
        print(f"Best parameters saved as {params_filename}")
        best_clf = grid_search.best_estimator_

        self.evaluation_model(name_model=name_model, classifier=best_clf, train_time=train_time, fine_turning=True)

    def models_naive_bayes(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        self.core_models('Naive Bayes', text_clf)

    def models_naive_bayes_hyperparams(self):
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('clf', MultinomialNB())
                             ])

        param_grid = {
            'clf__alpha': [0, 1e-2, 1e-3, 1e-1, 1.0],
            'vect__ngram_range': [(1, 1)],
            'vect__max_df': [0.7, 0.8, 0.9],
            'vect__max_features': [None, 5000, 10000],
            'vect__token_pattern': [r'\w{1,}']
        }

        self.core_models_hyperparams('Naive Bayes', text_clf, param_grid)

    def model_logistic_classification(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', LogisticRegression(solver='lbfgs',
                                                        multi_class='auto',
                                                        max_iter=10000))
                             ])
        self.core_models('Logistic Classification', text_clf)

    def model_logistic_classification_hyperparams(self):
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.7)),
            ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100))
        ])
        param_grid = {
            'vect__max_features': [None, 5000],
            'vect__token_pattern': [r'\w{1,}'],
            'clf__C': [0.1, 1, 10, 100],
            'clf__penalty': ['l2']
        }
        self.core_models_hyperparams('Logistic Classification', text_clf, param_grid)

    def model_svm(self):

        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SVC(gamma='scale', probability=True))
                             ])
        self.core_models('SVM', text_clf)

    def model_svm_hyperparams(self):

        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SVC(probability=True))
                             ])
        param_grid = {
            'clf__gamma': ['auto', 'scale']
        }
        self.core_models_hyperparams('SVM', text_clf, param_grid)

    def model_decisiontree(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', DecisionTreeClassifier(criterion='gini',
                                                            splitter='best',
                                                            max_depth=None,
                                                            min_samples_leaf=1,
                                                            min_samples_split=2))
                             ])
        self.core_models('ROC Decision Tree', text_clf)

    def model_decisiontree_hyperparams(self):
        # Define the pipeline
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)),
            ('clf', DecisionTreeClassifier())
        ])

        # Define the parameter grid
        param_grid = {
            'clf__criterion': ['gini'],
            'clf__max_depth': [None, 20, 30, 40],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        }
        self.core_models_hyperparams('Decision Tree', text_clf, param_grid)

    def model_randomforest(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', RandomForestClassifier(max_depth=100, random_state=0))
                             ])
        self.core_models('Random Forest', text_clf)

    def model_randomforest_hyperparams(self):

        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)),
            ('clf', RandomForestClassifier(random_state=0))
        ])

        # Define the parameter grid
        param_grid = {
            'clf__max_depth': [None, 50, 100, 150]
        }
        self.core_models_hyperparams('Random Forest', text_clf, param_grid)

    def model_xgboost_hyperparams(self):
        # Define the pipeline
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)),
            ('clf', XGBClassifier(use_label_encoder=False))
        ])

        # Define the parameter grid
        param_grid = {
            'clf__eval_metric': ['mlogloss', 'error']
        }
        self.core_models_hyperparams('XGBoost', text_clf, param_grid)

    def model_xgboost(self):
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.8, max_features=None)),
            ('classifier', XGBClassifier())
        ])
        self.core_models('XGBoost', text_clf)
