# Disertation-Vietnamese-NLP
The increasing volume of digital information poses a challenge for efficient information retrieval and categorization, particularly in languages with unique linguistic characteristics such as Vietnamese. This dissertation explores the development of an advanced model for preprocessing and classifying Vietnamese news articles utilising Natural Language Processing (NLP) techniques. The Vietnamese language, with its rich tonal system and distinct diacritics, necessitates a nuanced approach to text processing. The research focuses on developing the preprocessing phase, including lower-case documents, tokenization, removing punctuation and stop-words, standardisation accent, to optimise the representation of textual data. In the preprocessing phase, employing techniques such as lower-casing documents to ensure consistency across the entire dataset. The tokenization process, a key focus, methodically dissects the Vietnamese text into discernible units, navigating the intricacies of its vocabulary for a more comprehensive understanding. Additionally, the research places a crucial emphasis on accent standardisation, a nuanced step aimed at promoting uniformity and accuracy in the representation of accented characters within the textual data. 

The core of the research lies in the development of robust classification models, a task that is pivotal for effective information organisation. Leveraging cutting-edge machine learning algorithms such as Logistic Regression, Multinomial Naive Bayes, SVM, Decision Tree,  Random Forest, and XGBoost, the research aims to accurately categorise news articles while optimising the hyperparameters for these models. Additionally, the study conducts a comparative analysis to evaluate the effectiveness of these models, offering valuable insights into their respective performances.
The SVM model, tuned with a 'scale' gamma, and the XGBoost model, configured with 'mlogloss' as the evaluation metric, exhibited superior performance. They attained a high precision of 91.18% and 90.45%, and recall rates of 91.19% and 90.55%, along with F1-scores of 91.18% and 90.05%, respectively. These results underscore the proficiency of the models in the nuanced task of classifying Vietnamese news articles.
## Classification Report
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/c85c45aa-4a15-4ff6-9d7c-8daf107eb4ad)
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/0284f3e1-2058-4b09-b477-d689b5edb5f5)

## Confusion Matrix
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/ee32864e-8fb5-442a-afba-90383308b54f)
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/ad904f57-6e1e-4d4d-8926-479bc6362a65)

## ROC curve - AUC
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/91cff52d-66d2-40d6-8960-22ab9bf6fa3c)

## Comparison to State-of-the-Art Models
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/5b35e8e4-b193-4ff7-ab1c-e178eba719fa)
![image](https://github.com/MinaNguyen99/Disertation-Vietnamese-NLP/assets/56227298/0b9fa0c9-d79b-48d5-83e9-2abfdf6bd4e4)




