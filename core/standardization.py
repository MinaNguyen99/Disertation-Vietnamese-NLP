import regex as re

class VietnameseConverter:
    """
    Class to convert Vietnamese text to Telex typing style and standardize tone marks.
    """
    VOWEL_MATRIX = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                      ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                      ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                      ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                      ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                      ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                      ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                      ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                      ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                      ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                      ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                      ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
    TONE_MARKS = ['', 'f', 's', 'r', 'x', 'j']



    def __init__(self):
        
        self.vowel_to_matrix_indices = self._map_vowels_to_indices()

    @staticmethod
    def _map_vowels_to_indices():
        vowel_map = {}
        for vowel_group_index, vowels in enumerate(VietnameseConverter.VOWEL_MATRIX):
            for tone_index, vowel in enumerate(vowels[:-1]):
                vowel_map[vowel] = (vowel_group_index, tone_index)
        return vowel_map

    def standardize_vietnamese_tones_in_word(self, word):
        """
        Standardize tone marks in a Vietnamese word.
        """
        if not self.is_valid_vietnamese_word(word):
            return word

        characters = list(word)
        tone_mark = 0
        vowel_indices = []
        qu_or_gi_flag = False

        for index, char in enumerate(characters):
            vowel_group_index, tone_index = self.vowel_to_matrix_indices.get(char, (-1, -1))
            if vowel_group_index == -1:
                continue

            if vowel_group_index == 9 and index != 0 and characters[index - 1] == 'q':  # Check for 'qu'
                characters[index] = 'u'
                qu_or_gi_flag = True
            elif vowel_group_index == 5 and index != 0 and characters[index - 1] == 'g':  # Check for 'gi'
                characters[index] = 'i'
                qu_or_gi_flag = True

            if tone_index != 0:
                tone_mark = tone_index
                characters[index] = self.VOWEL_MATRIX[vowel_group_index][0]

            if not qu_or_gi_flag or index != 1:
                vowel_indices.append(index)
        if len(vowel_indices) < 2:
            if qu_or_gi_flag:
                if len(characters) == 2:
                    vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[1], (-1, -1))
                    characters[1] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]
                else:
                    vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[2], (-1, -1))
                    if vowel_group_index != -1:
                        characters[2] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]
                    else:
                        if characters[1] == 'i':
                            characters[1] = self.VOWEL_MATRIX[5][tone_mark]
                        elif characters[1] == 'u':
                            characters[1] = self.VOWEL_MATRIX[9][tone_mark]
                return ''.join(characters)
            return word

        for index in vowel_indices:
            vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[index], (-1, -1))
            if vowel_group_index in [4, 8]:  # ê, ơ
                characters[index] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]
                return ''.join(characters)

        if len(vowel_indices) == 2:
            if vowel_indices[-1] == len(characters) - 1:
                vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[vowel_indices[0]], (-1, -1))
                characters[vowel_indices[0]] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]
            else:
                vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[vowel_indices[1]], (-1, -1))
                characters[vowel_indices[1]] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]
        else:
            vowel_group_index, _ = self.vowel_to_matrix_indices.get(characters[vowel_indices[1]], (-1, -1))
            characters[vowel_indices[1]] = self.VOWEL_MATRIX[vowel_group_index][tone_mark]

        return ''.join(characters)


    def is_valid_vietnamese_word(self, word):
        """
        Check if a word is a valid Vietnamese word.
        """
        characters = list(word)
        last_vowel_index = -1
        for index, char in enumerate(characters):
            vowel_group_index, _ = self.vowel_to_matrix_indices.get(char, (-1, -1))
            if vowel_group_index != -1:
                if last_vowel_index == -1:
                    last_vowel_index = index
                else:
                    if index - last_vowel_index != 1:
                        return False
                    last_vowel_index = index
        return True

    def standardize_vietnamese_tones_in_sentence(self, sentence):
        """
        Standardize tone marks in a Vietnamese sentence.
        """
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            split_word = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            if len(split_word) == 3:
                split_word[1] = self.standardize_vietnamese_tones_in_word(split_word[1])
            words[index] = ''.join(split_word)
        return ' '.join(words)
    
"""
Usage:
if __name__ == '__main__':
    text_converter = VietnameseConverter()
    example_sentence = "Xin chaò bạn có khỏe không?"
    print(text_converter.standardize_vietnamese_tones_in_sentence(example_sentence))
Output: "xin chào bạn có khỏe không?"

"""
