from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from pythainlp.util import normalize
from pythainlp.ulmfit import *

def text_process(text):
    words = re.sub(r'[^ก-๙]', '', text)
    words = normalize(words)
    words = process_thai(words,
                            pre_rules=[replace_rep_after, fix_html, rm_useless_spaces],
                            post_rules=[ungroup_emoji,
                            replace_wrep_post_nonum,
                            remove_space]
                        )
    words = ' '.join(word for word in words)
    return words

max_length = 484
def create_tokenizer(words, filters = ''):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))
def padding_doc(encoded_doc, max_length):

    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

def cleansingTextToken(text):
    text = process_thai(text,
                            pre_rules=[replace_rep_after, fix_html, rm_useless_spaces],
                            post_rules=[ungroup_emoji,
                            replace_wrep_post_nonum,
                            remove_space]
                        )
    # text = ''.join([i for i in text])
    # text = re.sub(r'[a-z0-9]', '', text)
    # text = process_thai(text,
    #                         pre_rules=[replace_rep_after, fix_html, rm_useless_spaces],
    #                         post_rules=[ungroup_emoji,
    #                         replace_wrep_post_nonum,
    #                         remove_space]
    #                     )
    return text