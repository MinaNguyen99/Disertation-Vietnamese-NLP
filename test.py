import unittest
from core.preprocess_nlp import PreprocessingNLP


class TestPreprocessingNLP(unittest.TestCase):
    def test_unicode_erroe(self):
        text1 = 'hiếu'
        text2 = 'hiếu'
        sentence = PreprocessingNLP(text1)
        sentence.standard_unicode()
        self.assertEqual(sentence.sentences, text2)

    def test_remove_special_character(self):
        text = "Xin nhắc lại một chút về bối cảnh lúc đó.**%$$#"
        sentence = PreprocessingNLP(text)
        self.assertEqual(sentence.remove_special_character(), "Xin nhắc lại một chút về bối cảnh lúc đó",
                         "Couldn't remove special character")

    def test_remove_unnecessary_space(self):
        text = "   Xin      nhắc        lại         một     chút    về      bối     cảnh     lúc     đó.  \n"
        sentence = PreprocessingNLP(text)
        self.assertEqual(sentence.remove_unnecessary_space(), "Xin nhắc lại một chút về bối cảnh lúc đó.",
                         "Couldn't remove unnecessary space")

    def test_get_stopwords(self):
        p = PreprocessingNLP()
        self.assertEqual(len(p.stopword), 1942, "TRUE")

    def test_standardisation_case_type(self):
        text = "Xin nhắc lại một chút về bối cảnh lúc đó"
        sentence = PreprocessingNLP(text)
        self.assertEqual(sentence.standardisation_case_type(), "xin nhắc lại một chút về bối cảnh lúc đó",
                         "Couldn't lower case")


if __name__ == '__main__':
    unittest.main()
