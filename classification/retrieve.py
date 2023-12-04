import pandas as pd
import json
from core.preprocess_nlp import PreprocessingNLP
from core.chuan_hoa import chuan_hoa_dau_tu_tieng_viet
from underthesea import word_tokenize
import time

"""
This part is for change and group label name
"""

data = pd.read_csv('Fixed_news_dataset.csv')
data.dropna(subset=['content'], inplace=True)
filtered_data = data[['content', 'topic']]
filtered_data['topic'] = filtered_data['topic'].replace(['Pháp luật', 'Pháp luật'], 'Law')
filtered_data['topic'] = filtered_data['topic'].replace(['Sức khỏe - Đời sống'], 'Health - Life')
filtered_data['topic'] = filtered_data['topic'].replace(['Giáo dục'], 'Education')
filtered_data['topic'] = filtered_data['topic'].replace(['Thế giới'], 'World')
filtered_data['topic'] = filtered_data['topic'].replace(['Văn hóa - Giải trí'], 'Culture - Entertainment')
filtered_data['topic'] = filtered_data['topic'].replace(['Thời sự'], 'Current Affairs')
filtered_data['topic'] = filtered_data['topic'].replace(['Thể thao'], 'Sports')
filtered_data['topic'] = filtered_data['topic'].replace(['Xã hội'], 'Society')
filtered_data['topic'] = filtered_data['topic'].replace(['Bất động sản'], 'Real Estate')
filtered_data['topic'] = filtered_data['topic'].replace(['Công nghệ'], 'Technology')
filtered_data['topic'] = filtered_data['topic'].replace(['Kinh tế'], 'Economy - Finance')
filtered_data['topic'] = filtered_data['topic'].replace(['Kinh doanh - Tài chính'], 'Economy - Finance')
filtered_data['topic'] = filtered_data['topic'].replace(['Chính trị'], 'Politics - National Defense')
filtered_data['topic'] = filtered_data['topic'].replace(['Quốc phòng'], 'Politics - National Defense')
filtered_data['topic'] = filtered_data['topic'].replace(['Xe'], 'Automobiles')
filtered_data = filtered_data[filtered_data['topic'] != 'Bạn đọc']

# for i in range(0, len(filtered_data)):
#     temp.append({'content': filtered_data[i]['content'], 'topic': [filtered_data[i]['topic']]})
# filename = open('news_goods.json', 'w')
# json.dump(temp, filename, indent=6, ensure_ascii=False)
# filename.close()

"""
This is for preprocessing
"""
stopword = PreprocessingNLP().get_stopwords()
i = 0
start_time = time.time()
for index, row in filtered_data.iterrows():
    text = row['content']
    sentence = PreprocessingNLP(sentences=text, stopword=stopword)
    sentence.standard_unicode()
    sentence.sentences = chuan_hoa_dau_tu_tieng_viet(sentence.sentences)
    sentence.sentences = word_tokenize(sentence.sentences, format="text")
    sentence.standardisation_case_type()
    sentence.remove_unnecessary_space()
    sentence.remove_stopword()
    filtered_data.loc[index]['content'] = sentence.sentences

run_time = time.time() - start_time
print('runtime', run_time)
filtered_data.to_csv('good_data.csv', sep=',')
