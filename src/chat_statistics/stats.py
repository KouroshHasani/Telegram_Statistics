import json
from pathlib import Path
from typing import Union

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from hazm import Normalizer, word_tokenize
from loguru import logger
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    """
    Generates statistics from a telegram chat json file

    """
    def __init__(self, chat_json: Union[str, Path], stopword_path=DATA_DIR/'persian_stop_words.txt'):
        """
        :param chat_json: path of telegram chat json file
        :param stopword_path: path of stopwords document
        """

        # Load data
        logger.info(f"Loding data...")
        with open(chat_json) as fp:
            self.chat_data = json.load(fp)
        
        #Load stopwords
        logger.info("Loding stopwords...")
        with open(stopword_path) as fp:
            self.stop_words = fp.readlines()
            self.stop_words = list(map(str.strip, self.stop_words))
            self.stop_words = list(map(self.normalizer, self.stop_words))

        #Extract chat messages from loaded data
        logger.info("Extracting chat messages from loaded data...")
        self.chat_text = []
        for msg in self.chat_data['messages']:
            if isinstance(msg['text'], str):
                msg_normal = self.normalizer(msg['text'])
                self.chat_text.append(self.remove_stopwords(msg_normal))
            elif isinstance(msg['text'][0], str):
                msg_normal = self.normalizer(msg['text'][0])
                self.chat_text.append(self.remove_stopwords(msg_normal))
    

    def normalizer(self, text):
        """
        :param text: text you want to normalize 
        """
        normalizer = Normalizer()
        return text
        # return normalizer.normalize(text)


    def remove_stopwords(self, text):
        """
        :param text: text you want to delete stopwords from dat
        """
        tokens = word_tokenize(text)
        tokens = filter(lambda word: word not in self.stop_words, tokens)
        return " ".join(tokens)

    
    def generate_word_cloud(self, output_dir,
                            width=600, height=400, background_color='white',
                            font=str(DATA_DIR/'Iransans.ttf'),
                            max_font_size=200
                            ):
        """
        param output_dir: path of directory that you want to save image
        param width, height and background_color: image size and color
        param font and max_font_size: font and maximum size of the font
        """
        
        logger.info("Generating word Cloud...")
        text = arabic_reshaper.reshape(" ".join(self.chat_text))
        wc = WordCloud(
            width=width, height=height, background_color=background_color, 
            font_path=font, max_font_size=max_font_size
        ).generate(text)
        
        wc.to_file(str(Path(output_dir) / "wordcloud.png"))



if __name__ == "__main__":
    chatstat = ChatStatistics(chat_json=DATA_DIR / 'cs_stack.json')
    chatstat.generate_word_cloud(output_dir='/mnt/g/Courses/Data_Science/02_Python/My_project/02_telegram_statistics/Telegram_Statistics/src/chat_statistics')
    # print(chatstat.stop_words)
