import re
import gluonnlp as nlp

class JamoTokenizer:
    """JamoTokenizer class"""
    def __init__(self) -> None:
        """Instantiating JamoTokenizer class"""
        # 유니코드 한글 시작 : 44032, 끝 : 55199
        self.__base_code = 44032
        self.__chosung = 588
        self.__jungsung = 28
        # 초성 리스트. 00 ~ 18
        self.__chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                               'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                               'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        # 중성 리스트. 00 ~ 20
        self.__jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                                'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                                'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        # 종성 리스트. 00 ~ 27 + 1(1개 없음)
        self.__jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                                'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                                'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        self.token2idx = sorted(
            list(set(self.__chosung_list + self.__jungsung_list + self.__jongsung_list)))
        self.token2idx = ['<pad>', '<eng>', '<num>', '<unk>'] + self.token2idx
        self.token2idx = {token: idx for idx, token in enumerate(self.token2idx)}

    def tokenize(self, string: str) -> list:
        """Tokenizing string to sequences of indices

        Args:
            string (str): characters

        Returns:
            sequence of tokens (list): list of characters
        """
        split_string = list(string)

        sequence_of_tokens = []
        for char in split_string:
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
                if ord(char) < self.__base_code:
                    sequence_of_tokens.append(char)
                    continue

                char_code = ord(char) - self.__base_code
                alphabet1 = int(char_code / self.__chosung)
                sequence_of_tokens.append(self.__chosung_list[alphabet1])
                alphabet2 = int(
                    (char_code - (self.__chosung * alphabet1)) / self.__jungsung)
                sequence_of_tokens.append(self.__jungsung_list[alphabet2])
                alphabet3 = int(
                    (char_code - (self.__chosung * alphabet1) - (self.__jungsung * alphabet2)))
                if alphabet3 != 0:
                    sequence_of_tokens.append(self.__jongsung_list[alphabet3])
            else:
                sequence_of_tokens.append(char)

        return sequence_of_tokens

    def transform(self, sequence_of_tokens: list) -> list:
        """Transforming sequences of tokens to sequences of indices

        Args:
            sequence_of_tokens (list): list of characters

        Returns:
            sequence_of_indices (list): list of integers
        """
        sequence_of_indices = []
        for token in sequence_of_tokens:
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', token) is not None:
                sequence_of_indices.append(self.token2idx.get(token))
            else:
                if re.match('[0-9]', token) is not None:
                    sequence_of_indices.append(self.token2idx.get('<num>'))
                elif re.match('[A-z]', token) is not None:
                    sequence_of_indices.append(self.token2idx.get('<eng>'))
                else:
                    sequence_of_indices.append(self.token2idx.get('<unk>'))

        return sequence_of_indices

    def tokenize_and_transform(self, string: str) -> list:
        """Tokenizing and transforming string to sequence

        Args:
            string (str): characters

        Returns:
            sequence_of_indices (list): list of integers
        """
        sequence_of_indices = self.transform(self.tokenize(string))

        return sequence_of_indices


class Padder:
    def __init__(self, length):
        self.padder = nlp.data.PadSequence(length=length)

