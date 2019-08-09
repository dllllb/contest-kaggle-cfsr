import pandas as pd
import numpy as np
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer

from sklearn.tree import DecisionTreeClassifier

from nltk.util import ngrams

import ds_tools.dstools.ml.metrics as m
import ds_tools.dstools.ml.ensemble as ens
import ds_tools.dstools.ml.xgboost_tools as xgb


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


def qwk_score(est, features, labels):
    pred = preds_to_rank(est.predict(features))
    return m.quadratic_weighted_kappa(labels, pred)


def qwk_evalerror(preds, dtrain):
    pred = preds_to_rank(preds)
    return 'qwk',  m.quadratic_weighted_kappa(dtrain.get_label(), pred)

  
def column_transformer(name):
    return FunctionTransformer(partial(pd.DataFrame.__getitem__, key=name), validate=False)


def init_xgb_est(params):
    keys = {
        'eta',
        'num_rounds',
        'max_depth',
        'min_child_weight',
        'gamma',
        'subsample',
        'colsample_bytree',
        'num_es_rounds',
        'es_share'
    }
        
    xgb_params = {
        "objective": "reg:linear",
        "verbose": 120,
        **{k: v for k, v in params.items() if k in keys},
    }
    
    if params['es_metric'] == 'qwk':    
        xgb_params['eval_func'] = qwk_evalerror
        xgb_params['maximize_eval'] = True
    
    return xgb.XGBoostRegressor(**xgb_params)


def init_tc_transf(params):
    return make_union(
        make_pipeline(
            column_transformer('query'),
            CountVectorizer(),
        ),
        make_pipeline(
            column_transformer('product_title'),
            CountVectorizer(),
        ),
        make_pipeline(
            column_transformer('product_description'),
            CountVectorizer(),
        ),
    )


def cfsr_dataset(_):
    import os
    df = pd.read_csv(f'{os.path.dirname(__file__)}/train.csv.gz')
    df.fillna({'query': '', 'product_title': '', 'product_description': ''}, inplace=True)
    features = df.drop(['median_relevance', 'relevance_variance'], axis=1)
    return features, df.median_relevance


def cfsr_estimator(params):    
    transf_type = params['transf_type']
    if transf_type == 'term_cnt':
        transf = init_tc_transf(params)
    elif transf_type == 'qmatch':
        transf = FunctionTransformer(query_match, validate=False)
    elif transf_type == 'qm+tc':
        transf = make_union(
            FunctionTransformer(query_match, validate=False),
            init_tc_transf(params)
        )
    else:
        raise AssertionError(f'unknown transformer: "{transf_type}"')
    
    est_type = params['est_type']
    if est_type == 'rfr':
        est = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            n_jobs=params['n_rfr_jobs'],
            min_samples_split=params['min_samples_split'],
            random_state=1,
            verbose=0)
    elif est_type == 'xgb':
        est = init_xgb_est(params)
    elif est_type == 'xgb/dt':
        est = ens.ModelEnsembleRegressor(
            intermediate_estimators=[init_xgb_est(params)],
            assembly_estimator=DecisionTreeClassifier(max_depth=2),
            ensemble_train_size=1)
    else:
        raise AssertionError(f'unknown estimator: "{est_type}"')

    pl = make_pipeline(transf, est)
    return pl


def cfsr_params(overrides):
    defaults = {
        'n_folds': 3,
        'valid_type': 'cv',
        "eta": 0.01,
        "min_child_weight": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "silent": 1,
        "max_depth": 6,
        "num_rounds": 10000,
        "num_es_rounds": 120,
        "es_share": .05,
        'n_estimators': 200,
        'n_rfr_jobs': 2,
        'min_samples_split': 4,
        'es_metric': 'rmse',
    }
    return {**defaults, **overrides}


def cfsr_experiment(overrides):
    params = cfsr_params(overrides)

    results = run_experiment(
        params=params,
        est=cfsr_estimator,
        dataset=cfsr_dataset,
        scorer=qwk_score)

    update_model_stats('results.json', params, results)


def test_cfsr_experiment():
    params = {
        'n_folds': 2,
        'est_type': 'xgb',
        'transf_type':'qmatch',
        "num_rounds": 100,
    }

    params = cfsr_params(params)

    results = run_experiment(
        params=params,
        est=cfsr_estimator,
        dataset=cfsr_dataset,
        scorer=qwk_score)

    print(results)


def update_model_stats(stats_file, params, results):
    import json
    import os.path

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []

    stats.append({**results, **params})

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)


def run_experiment(est, dataset, scorer, params):
    import time

    start = time.time()
    if params['valid_type'] == 'cv':
        cv = params['n_folds']
        features, target = dataset(params)
        scores = cv_test(est(params), features, target, scorer, cv)
    exec_time = time.time() - start
    return {**scores, 'exec-time-sec': exec_time}


def cv_test(est, features, target, scorer, cv):
    scores = cross_val_score(est, features, target, scoring=scorer, cv=cv)
    return {'score-mean': scores.mean(), 'score-std': scores.std()}
