from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
thai_stopwords = list(thai_stopwords())
from string import punctuation
from pythainlp.ulmfit import (process_thai,
    replace_rep_after,
    fix_html,
    ungroup_emoji,
    replace_wrep_post,
    remove_space)

def cleaning(sentences):
    words = []
    temp = []
    for s in sentences:
        clean = re.sub(r'[^ก-๙]', "", s)
        # w = word_tokenize(clean, engine='deepcut')
        # w = word_tokenize(clean, engine='attacut')
        w = process_thai(clean,
                pre_rules=[replace_rep_after, fix_html],
                    post_rules=[ungroup_emoji,
                                replace_wrep_post,
                                remove_space])
        temp.append([i.lower() for i in w])
        words.append(' '.join(w).lower())

    return words, temp

def create_tokenizer(words, filters = ''):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

def max_length(words):
    return(len(max(words, key = len)))
