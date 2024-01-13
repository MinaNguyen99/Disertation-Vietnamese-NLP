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

from sklearn.svm import SVC
import numpy as np
import time
import pandas as pd
from xgboost import XGBClassifier
import optuna
import os
from itertools import cycle


class NewsClassificationModel:

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

    def balanced_dataset(self, max_numbers_sample):
        label_encoder = LabelEncoder()
        self.data['topic'] = label_encoder.fit_transform(self.data['topic'])
        self.label_encoder = label_encoder
        self.data = self.data.dropna()
        tfidf_vectorizer = TfidfVectorizer()
        self.X = tfidf_vectorizer.fit_transform(self.data['content'])
        self.y = self.data['topic']
        print(self.y.value_counts())
        # Apply SMOTE
        smote = SVMSMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X, self.y)
        # Get the remaining classes after SMOTE
        remaining_classes = set(label_encoder.inverse_transform(y_resampled))

        classes_to_undersample = [label for label in remaining_classes if
                                  sum(y_resampled == label) > max_numbers_sample]
        rus = RandomUnderSampler(sampling_strategy={label: max_numbers_sample for label in classes_to_undersample})
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

        self.data['content'] = X_resampled
        self.data['topic'] = y_resampled

        # Convert back to DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=tfidf_vectorizer.get_feature_names_out())
        df_resampled['topic'] = y_resampled

        # Save resampled data to CSV
        df_resampled.to_csv('balanced_dataset.csv', index=False)

    def balance_dataset_by_rsampling_smote(self):

        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.y_train)
        self.label_encoder = label_encoder

        self.tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)

        # Step 5: Balance the dataset using SMOTE
        smote = SMOTE(random_state=42)
        rus = RandomUnderSampler(random_state=42)

        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, self.y_train)
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

        # Now, X_resampled and y_resampled contain the balanced dataset

        self.X_train = X_resampled
        self.y_train = y_resampled
        self.X_test = self.tfidf_vectorizer.transform(self.X_test)
        self.y_test = self.label_encoder.transform(self.y_test)

        # Step 7: (Optional) Convert X_resampled back to a DataFrame if needed
        df_resampled = pd.DataFrame({'content': X_resampled, 'topic': y_resampled})

        df_resampled.to_csv('balanced_dataset.csv', index=False)

    def prepare_train_test_set(self):
        test_size = 0.3
        label_encoder = LabelEncoder()
        self.data['topic'] = label_encoder.fit_transform(self.data['topic'])
        text = self.data['content']
        label = self.data['topic']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(text, label, test_size=test_size,
                                                                                random_state=42)

    def evaluation_model(self, name_model, classifier, train_time, label_encoder, fine_turning=False):
        output_folder = '/home/sotatek/PycharmProjects/disertation_nlp/output'
        # Predict on test set
        y_pred = classifier.predict(self.X_test)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        class_names = list(self.label_encoder.classes_)
        report = metrics.classification_report(self.y_test, y_pred, target_names=class_names)
        if fine_turning is False:
            report_filename = os.path.join(output_folder, f"{name_model}_classification_report.txt")
        else:
            report_filename = os.path.join(output_folder, f"FT_{name_model}_classification_report.txt")
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
            cm_image_filename = os.path.join(output_folder, f"{name_model}_confusion_matrix.png")
        else:
            cm_image_filename = os.path.join(output_folder, f"FT_{name_model}_confusion_matrix.png")
        plt.savefig(cm_image_filename, bbox_inches='tight')
        plt.close()

        """ Create ROC Curve"""
        # Binarize the labels for multi-class ROC curve
        y_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_bin.shape[1]

        # Compute ROC curve and ROC area for each class
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
            roc_curve_filename = os.path.join(output_folder, f"{name_model}_multi_class_roc_curve.png")
        else:
            roc_curve_filename = os.path.join(output_folder, f"FT_{name_model}_multi_class_roc_curve.png")
        plt.savefig(roc_curve_filename)
        plt.close()
        print(f'Done with {name_model}')

    def core_models(self, name_model, clf):
        start_time = time.time()
        text_clf = clf.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        y_pred = text_clf.predict(self.X_test)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        class_names = list(self.label_encoder.classes_)
        report = metrics.classification_report(self.y_test, y_pred, target_names=class_names)
        output_folder = '/home/sotatek/PycharmProjects/disertation_nlp/output'
        report_filename = os.path.join(output_folder, f"{name_model}_classification_report.txt")
        with open(report_filename, 'w') as file:
            file.write(f"{name_model} Training Time: {train_time} seconds\n\n")
            file.write(f'{name_model}, Accuracy ={accuracy}\n')
            file.write("Classification Report:\n")
            file.write(report)

        # create a confusion matrix of model and save it
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
        cm_image_filename = os.path.join(output_folder, f"{name_model}_confusion_matrix.png")
        plt.savefig(cm_image_filename, bbox_inches='tight')
        plt.close()

        """ Create ROC Curve"""
        # Binarize the labels for multi-class ROC curve
        y_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        y_score = clf.predict_proba(self.X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
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
        roc_curve_filename = os.path.join(output_folder, f"{name_model}_multi_class_roc_curve.png")
        plt.savefig(roc_curve_filename)
        plt.close()
        print(f'Done with {name_model}')

    def core_models_hyperparams(self, name_model, clf, param_grid,
                                output_folder='/home/sotatek/PycharmProjects/disertation_nlp/output'):
        start_time = time.time()
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        print(f'Done training {name_model} in', train_time, 'seconds.')
        print("Best parameters: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("Test set accuracy: {:.2f}".format(grid_search.score(self.X_test, self.y_test)))

        # Save the best model
        model_filename = os.path.join(output_folder, f"{name_model}_best_model.joblib")
        dump(grid_search.best_estimator_, model_filename)
        print(f"Best model saved as {model_filename}")

        # Optionally, save the best parameters in a separate file
        params_filename = f"{name_model}_best_params.txt"
        with open(params_filename, 'w') as file:
            file.write(str(grid_search.best_params_))
        print(f"Best parameters saved as {params_filename}")

    def models_naive_bayes(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        self.core_models('Naive Bayes', text_clf)

    def model_naive_bayes_hyperparameters(self):
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
        param_grid = {
            'vect__ngram_range': [(1, 1)],
            'vect__max_df': [0.7, 0.8, 0.9],
            'tfidf__use_idf': [True, False],
            'clf__alpha': [1e-2, 1e-3, 1e-1, 1.0],
            'clf__fit_prior': [True, False],
        }
        self.core_models_hyperparams('Naive Bayes', text_clf, param_grid)

    def models_naive_bayes_hyperparams(self):

        alpha = np.linspace(0, 6, 5)
        param_grid = {
            'clf__alpha': alpha,
            'vect__ngram_range': [(1, 1)],
            'vect__max_df': [0.7, 0.8, 0.9],
            'vect__max_features': [None, 5000, 10000],
            'vect__token_pattern': [r'\w{1,}']
        }
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('clf', MultinomialNB())
                             ])
        self.core_models_hyperparams('Naive Bayes', text_clf, param_grid)

    def model_logistic_regression(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', LogisticRegression(solver='lbfgs',
                                                        multi_class='auto',
                                                        max_iter=10000))
                             ])
        self.core_models('Logistic Classification', text_clf)

    def model_svm(self):

        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SVC(gamma='scale', probability=True))
                             ])
        self.core_models('SVM', text_clf)

    def model_decisiontree(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=13))
                             ])
        self.core_models('Decision Tree', text_clf)

    def model_randomforest(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('clf', RandomForestClassifier(max_depth=100, random_state=0))
                             ])
        self.core_models('Random Forest', text_clf)

    def model_xgboost(self):
        text_clf = Pipeline([
            ('classifier', XGBClassifier())
        ])
        self.core_models('XGBoost', text_clf)

    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'objective': ['reg:linear', 'binary:logistic'],
            'eval_metric': ['rmse', 'logloss'],
        }

        model = xgb.train(params, dtrain=xgb.DMatrix(self.X_train, self.y_train), evals=[(self.X_test, self.y_test)],
                          early_stopping_rounds=10)
        y_pred = model.predict(self.X_test)
        rmse = np.sqrt(np.mean((y_pred - self.y_test) ** 2))

        return rmse

    def model_xgboost_hyperparameters(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_model = xgb.train(best_params, dtrain=xgb.DMatrix(self.X_train, self.y_train))
        y_pred = best_model.predict(self.X_test)
        rmse = np.sqrt(np.mean((y_pred[:, 0] - self.y_test[:, 0]) ** 2))
        logloss = xgb.logloss(self.y_test[:, 1], y_pred[:, 1])
        print('Test RMSE for linear objective:', rmse)
        print('Test logloss for Gini objective:', logloss)
