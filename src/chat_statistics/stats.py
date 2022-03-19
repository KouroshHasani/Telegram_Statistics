import json
import pickle
from pathlib import Path
from typing import Union

import arabic_reshaper
from bidi.algorithm import get_display
from loguru import logger
from sklearn.linear_model import LogisticRegression
from src.data import DATA_DIR
from src.utils.io import read_file, read_tel_messages
from wordcloud import WordCloud


class ChatStatistics:
    """
    Generates statistics from a telegram chat json file

    """
    def __init__(self, chat_json: Union[str, Path],
                 stopword_path=DATA_DIR/'persian_stop_words.txt',
                 normalize=False
                ):
        """
        :param chat_json: path of telegram chat json file
        :param stopword_path: path of stopwords document
        """
        #Extract chat messages from loaded data
        logger.info("Extracting chat messages from loaded data...")
        self.chat_text = read_tel_messages(file_path=chat_json, normalize=normalize, stopwords_path=stopword_path)


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
        
        logger.info("Generating word cloud...")
        text = arabic_reshaper.reshape(" ".join(self.chat_text))
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color, 
            font_path=font,
            max_font_size=max_font_size
        ).generate(text)
        
        wc.to_file(str(Path(output_dir) / "wordcloud.png"))


class Predict_text_type:
    """
    Predict that a text is question or not
    """
    def __init__(self):
        self.model = pickle.load(open(DATA_DIR / 'finalized_model.sav', 'rb'))
        self.feature_list = read_file(file_path=DATA_DIR / 'feature.txt').split('\n')


    def prdicet(self, input_: str):
        """
        :param input: A text that you want to predict its type
        """
        list_ = [0] * len(self.feature_list)
        for ind, item in enumerate(self.feature_list):
            if item in input_:
                list_[ind] = 1

        if self.model.predict([list_])[0] == 1:
            return "Text is question"
        
        else:
            return "Text is not question"



if __name__ == "__main__":
    # chatstat = ChatStatistics(chat_json=DATA_DIR / 'cs_stack.json', normalize=False)
    # chatstat.generate_word_cloud(output_dir='/mnt/g/Courses/Data_Science/02_Python/My_project/02_telegram_statistics/Telegram_Statistics/src/chat_statistics')
    # print(chatstat.stop_words)

    model = Predict_text_type()
    print(model.prdicet("سلام ببخشید چطوری میتوتم پایتون رو نصب کنم؟؟"))
