from core_models import NewsClassificationModel
from core.preprocess_nlp import PreprocessingNLP
from core.standardization import VietnameseConverter
from underthesea import word_tokenize
import time
import pandas as pd
import os
import joblib


class VietnameseNewsClassification:

    def __init__(self):
        self.data = None

    def load_and_group_data(self, file_path='Fixed_news_dataset.csv'):
        # Load the data and drop rows with missing 'content'
        data = pd.read_csv(file_path)
        data.dropna(subset=['content'], inplace=True)

        # Filter out necessary columns
        filtered_data = data[['content', 'topic']]

        # Define a mapping for topic renaming
        topic_mapping = {
            'Pháp luật': 'Law', 'Pháp luật': 'Law',
            'Giáo dục': 'Education',
            'Thế giới': 'World',
            'Văn hóa - Giải trí': 'Culture - Entertainment',
            'Bất động sản': 'Real Estate',
            'Công nghệ': 'Technology',
            'Kinh tế': 'Economy - Finance',
            'Kinh doanh - Tài chính': 'Economy - Finance',
            'Chính trị': 'Politics',
            'Quốc phòng': 'National Defense',
            'Xe': 'Automobiles',
            'Thể thao': 'Sport',
            'Sức khỏe - Đời sống': 'Health - Life',
            'Thời sự': "Current Affairs",
            'Xã hội': "Society",
            'Bạn đọc': "Readership"

        }
        filtered_data['topic'] = filtered_data['topic'].map(topic_mapping).fillna(filtered_data['topic'])
        # topics_to_exclude = ['Sức khỏe - Đời sống', 'Thời sự', 'Xã hội', 'Bạn đọc']
        # filtered_data = filtered_data[~filtered_data['topic'].isin(topics_to_exclude)]

        self.data = filtered_data

    def retrieve_data(self):
        if self.data is None:
            output_folder = '/home/sotatek/PycharmProjects/disertation_nlp/output'
            cleaned_filename = os.path.join(output_folder, f"cleaned_data.csv")
            self.data = pd.read_csv(cleaned_filename)
            self.data.dropna(subset=['content'], inplace=True)

    def drop_topics(self):
        self.retrieve_data()
        topics_to_exclude = ['Health - Life', 'Current Affairs', 'Society', 'Readership']
        self.data = self.data[~self.data['topic'].isin(topics_to_exclude)]

    def run_preprocesing(self):
        filtered_data = self.data
        stopword = PreprocessingNLP().get_stopwords()
        text_converter = VietnameseConverter()
        start_time = time.time()

        for index, row in filtered_data.iterrows():
            text = row['content']
            sentence = PreprocessingNLP(sentences=text, stopword=stopword)
            sentence.standard_unicode()
            sentence.remove_html()
            sentence.sentences = text_converter.standardize_vietnamese_tones_in_sentence(sentence.sentences)
            sentence.sentences = word_tokenize(sentence.sentences, format="text")
            sentence.standardisation_case_type()
            sentence.remove_unnecessary_space()
            sentence.remove_stopword()
            filtered_data.loc[index]['content'] = sentence.sentences

        run_time = time.time() - start_time
        print('Processing step runtime: ', run_time)
        output_folder = '/home/sotatek/PycharmProjects/disertation_nlp/output'
        cleaned_filename = os.path.join(output_folder, f"cleaned_data.csv")
        filtered_data.to_csv(cleaned_filename, sep=',')
        self.data = filtered_data

    def prepare_model(self, max_numbers_sample=5000):
        self.retrieve_data()
        model = NewsClassificationModel(data=self.data)
        model.pre_unbalanced_classification(max_numbers_sample)
        model.prepare_train_test_set()
        return model

    def run_model(self, model):
        model.model_svm()
        model.model_logistic_regression()
        model.model_randomforest()
        model.model_decisiontree()
        model.model_xgboost()
        model.model_svm()

    def run_model_hyperparameters(self, model):
        model.model_naive_bayes_hyperparams()
        model.model_svm_hyperparams()
        model.model_logistic_classification_hyperparams()
        model.model_randomforest_hyperparams()
        model.model_decisiontree_hyperparams()
        model.model_xgboost_hyperparams()

"""
Usage:
run_all = VietnameseNewsClassification()
run_all.load_and_group_data() # this is for loading and grouping
run_all.run_preprocesing() # this is for preprocessing 
run_all.drop_topics()  # this is for drop records 
model = run_all.prepare_model()  # this is for balancing, test train split  
run_all.run_model(model) # This is for run example models
run_all.run_model_hyperparameters() this for run and fine-turning params
"""

