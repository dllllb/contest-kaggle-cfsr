# coding=utf-8
# https://www.kaggle.com/c/crowdflower-search-relevance
"""
Solutions:
42-th place: https://github.com/marknagelberg/search-relevance/
107-th place: https://www.kaggle.com/lancerts/crowdflower-search-relevance/combined

Discussions:
Александр Дьяконов: https://www.youtube.com/watch?v=kzNJEMR4ltY
"""

import pandas as pd
import numpy as np


def preds_to_rank(preds):
    splits = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    res = np.digitize(preds, splits)
    return res


def same_terms_count(left, right):
    return [len(set(l).intersection(set(r))) for l, r in zip(left, right)]


def text2ngrams(text):
    return [''.join(ng) for ng in ngrams(text, 3)]


def query_match(df):
    res = pd.DataFrame()

    query_ngrams = df['query'].fillna('').map(text2ngrams)
    title_ngrams = df.product_title.fillna('').map(text2ngrams)
    desc_ngrams = df.product_description.fillna('').map(text2ngrams)

    res['query_len'] = query_ngrams.map(len)
    res['title_len'] = title_ngrams.map(len)
    res['desc_len'] = desc_ngrams.map(len)

    res['query_ngrams_in_title'] = same_terms_count(query_ngrams, title_ngrams)
    res['query_ngrams_in_desc'] = same_terms_count(query_ngrams, desc_ngrams)

    res['ratio_title'] = res['query_ngrams_in_title']/(res['query_len']+.00001)
    res['ratio_description'] = res['query_ngrams_in_desc']/(res['query_len']+.00001)

    return res
