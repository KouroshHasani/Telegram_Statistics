import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union

import arabic_reshaper
import pandas as pd
from bidi.algorithm import get_display
from loguru import logger
from sklearn.linear_model import LogisticRegression
from src.data import DATA_DIR
from src.utils.io import (get_text_from_tel_msg, read_file, read_json,
                          read_tel_messages, remove_stopwords, search_in_lines)
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
        logger.info("Loding data from json...")
        self.data = read_json(chat_json)
        self.stopwords = read_file(stopword_path).split('\n')
        self.chat_text = read_tel_messages(tele_data=self.data, normalize=normalize, stopword=self.stopwords)
        self.message_ids = {}
        for ind, msg in enumerate(self.data['messages']):
            self.message_ids[msg['id']] = ind


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


    def questions_list(self):
        """
        Return all questions exist in telegram json file
        """
        logger.info("Loding questions...")
        questions = []
        for msg in self.data['messages']:
            msg_ = get_text_from_tel_msg(msg)
            msg_ = remove_stopwords(msg_, self.stopwords)
            if sum([i in msg_ for i in ['؟', '?']]) > 0:
                questions.extend(search_in_lines(text=msg_, search_val=['؟', '?']))

        return questions


    def responder_users(self, number: int=5):
        """
        :param number: numer of active users
        """
        logger.info("Loding active responders...")
        users = defaultdict(dict)
        for msg in self.data['messages']:
            if not 'reply_to_message_id' in msg:
                continue
            
            if not msg['reply_to_message_id'] in self.message_ids:
                continue
            
            replied_msg_id = self.message_ids[msg['reply_to_message_id']]
            
            msg_ = get_text_from_tel_msg(self.data['messages'][replied_msg_id])
            if sum([i in msg_ for i in ['؟', '?']]) == 0:
                continue

            users[msg['from_id']]['Name'] = f"{msg['from']}"
            
            if 'Replies' in users[msg['from_id']]:
                users[msg['from_id']]['Replies'] += [replied_msg_id]

            else:
                users[msg['from_id']]['Replies'] = [replied_msg_id]


            responders = pd.DataFrame((len(users[i]['Replies']) for i in users),
            index=(users[i]['Name'] for i in users), columns=['num_replies'])\
            .sort_values('num_replies', ascending=False)
            

        return responders.head(number)



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

        if self.model.predict([list_])[0] == 0:
            return "Text is question"
        
        else:
            return "Text is not question"



if __name__ == "__main__":
    # chatstat = ChatStatistics(chat_json=DATA_DIR / 'cs_stack.json', normalize=False)
    # chatstat.generate_word_cloud(output_dir='/mnt/g/Courses/Data_Science/02_Python/My_project/02_telegram_statistics/Telegram_Statistics/src/chat_statistics')

    # model = Predict_text_type()
    # print(model.prdicet("ببخشید چطوری میتونم پایتون رو نصب کنم؟؟"))

    chatstat = ChatStatistics(chat_json=DATA_DIR / 'cs_stack.json', normalize=False)
    print(chatstat.responder_users(5))
