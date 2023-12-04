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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.svm import SVC
import numpy as np
import time
import pandas as pd
from xgboost import XGBClassifier
import optuna


class NewsClassification:

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
        # svd_vectorizer = TruncatedSVD(n_components=1, random_state=42)
        # svd_vectorizer.fit(self.X)
        # self.X = svd_vectorizer.transform(self.X)

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

    def core_models(self, name_model, clf):
        start_time = time.time()
        text_clf = clf.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        print(f'Done training {name_model} in', train_time, 'seconds.')
        y_pred = text_clf.predict(self.X_test)
        print(f'{name_model}, Accuracy =', metrics.accuracy_score(self.y_test, y_pred))
        print(metrics.classification_report(self.y_test, y_pred, target_names=list(self.label_encoder.classes_)))

    def core_models_hyperparams(self, name_model, clf, param_grid):
        start_time = time.time()
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        print(f'Done training {name_model} in', train_time, 'seconds.')
        print("Best parameters: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("Test set accuracy: {:.2f}".format(grid_search.score(self.X_test, self.y_test)))

    def models_naive_bayes(self):
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                      max_df=0.8,
                                                      max_features=None)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        self.core_models('Naive Bayes', text_clf)

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
                             ('clf', SVC(gamma='scale'))
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
        self.core_models('Decision Tree', text_clf)

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
