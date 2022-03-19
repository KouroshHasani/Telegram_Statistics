import json

from hazm import Normalizer, word_tokenize
from loguru import logger


def read_json(file_path: str) -> dict:
    """
    Reads a json file and return the dict
    """
    # Load data
    logger.info("Loding data from json...")
    with open(file_path) as fp:
        return json.load(fp)


def read_file(file_path: str) -> str:
    """
    Reads a file and return content
    """
    # Load data
    logger.info("Loding data from text...")
    with open(file_path) as fp:
        return fp.read()


def remove_stopwords(text: str, stopwords: list):
    """
    :param text: text you want to delete stopwords from dat
    :param stopwords: list of stopwords
    """
    tokens = word_tokenize(text)
    tokens = filter(lambda word: word not in stopwords, tokens)
    return " ".join(tokens)


def read_tel_messages(file_path: str, normalize: bool=False, stopwords_path: str=None) -> list:
    """
    Reads a json file and return the messages as a list
    :param file_path: Path of Text file
    :param normalize: Normalize text or not
    :param stopwords_path: Path of stopwords file
    """
    data = read_json(file_path)

    #Load stopwords
    if not stopwords_path is None:
        stopwords = read_file(stopwords_path).split('\n')

    normalizer = Normalizer()

    #Extract chat messages from loaded data
    logger.info("Extracting chat messages from loaded data...")
    data_text = []
    for msg in data['messages']:
        if isinstance(msg['text'], str):
            msg_ = msg['text']
        
        elif isinstance(msg['text'][0], str):
            msg_ = msg['text'][0]
        
        else:
            pass

        if normalize == True:
            msg_ = normalizer.normalize(msg_)
        
        if not stopwords_path is None: 
            msg_ = remove_stopwords(msg_, stopwords)

        data_text.append(msg_)
            
    return data_text
