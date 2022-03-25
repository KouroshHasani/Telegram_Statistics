import json

from hazm import Normalizer, sent_tokenize, word_tokenize


def read_json(file_path: str) -> dict:
    """
    Reads a json file and return the dict
    """
    # Load data
    with open(file_path) as fp:
        return json.load(fp)


def read_file(file_path: str) -> str:
    """
    Reads a file and return content
    """
    # Load data
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


def read_tel_messages(tele_data: dict, normalize: bool=False, stopword: list=None) -> list:
    """
    Reads a json file and return the messages as a list
    :param tel_data: A dictionary of telegram data
    :param normalize: Normalize text or not
    :param stopword: List of stopwords
    """

    normalizer = Normalizer()
    #Extract chat messages from loaded data
    data_text = []
    for msg in tele_data['messages']:
        msg_ = get_text_from_tel_msg(msg)
        if normalize == True:
            msg_ = normalizer.normalize(msg_)
        
        if not stopword is None: 
            msg_ = remove_stopwords(msg_, stopword)

        data_text.append(msg_)
            
    return data_text


def get_text_from_tel_msg(tel_msg):
    """
    Gets a telegram message and returns all text as string
    :param tel_msg: Telegram message 
    """
    if isinstance(tel_msg['text'], str):
        msg_ = tel_msg['text']
        
    else:
        msg_ = ""
        for sub_msg in tel_msg['text']:
            if isinstance(sub_msg, str):
                msg_ += " " + sub_msg
                
            elif 'text' in sub_msg:
                msg_ += " " + sub_msg['text']
    
    return msg_


def search_in_lines(text: str=None, text_path: str=None,search_val: list=None) -> list:
    """
    Returns all sentences that have at least one of the search values 
    :param text: A text containig multi lines
    :param text_path: Path of a text containig multi lines
    :param search_val: If one of these values exist in setence, that sentece will be returned
    Output: A list of string objects
    """
    sentences = []
    if text is not None:
        text_ = text
    
    elif text_path is not None:
        with open(text_path) as fp:
            text_ = fp.read()
            
    else:
        raise InputError

    textlines = sent_tokenize(text_)
    for sen in textlines:
        if (sum([i in sen for i in search_val]) > 0) and (len(sen) >= 3):
            sentences.append(sen)                
    return sentences
