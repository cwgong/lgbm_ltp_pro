# -*- encoding:utf8 -*-
import io
import json
import re
import codecs
import os
from gensim import corpora, models, similarities
from jieba.analyse import textrank
import jieba
import logging.config
import time_utils
import torch
import torch.nn as nn
import numpy as np

jieba.load_userdict('WordsDic/userdict_.txt')
jieba.analyse.set_stop_words('WordsDic/stopwords.txt')

logging.config.fileConfig("logging.conf")
logger = logging.getLogger('main')

def load_filter_list():
    with codecs.open('config/filter_title', 'r', encoding='utf-8') as f:
        filter_list = f.readlines()
    return filter_list

def load_stop_words():
    with codecs.open('WordsDic/stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [x.strip() for x in f.readlines()]
    return stop_words

def filter_title(title, title_filter_list):
    if len(title.strip()) == 0:
        return False
    for item in title_filter_list:
        if item.strip() in title:
            return False
    if len(re.findall("([0-9]*)月([0-9]*)日", title)) > 0:
        return False
    if '同比' in title and '净利' in title:
        return False
    if '同比' in title and '利润' in title:
        return False
    if '同比' in title and '产量' in title:
        return False
    if '同比' in title and '增长' in title:
        return False
    if '营收' in title and '净利' in title:
        return False
    if '快讯' in title and '报于' in title:
        return False
    if '目标价' in title and '评级' in title:
        return False
    if '买入' in title and '评级' in title:
        return False
    if '增持' in title and '评级' in title:
        return False
    if '维持' in title and '评级' in title:
        return False
    if '质押' in title and '万股票' in title:
        return False
    if '本周国内' in title and '价格' in title:
        return False
    #if '利润' in title or '净利' in title and '降' in title or '减' in title or '增' in title or '涨' in title:
     #   return False
    if '营收' in title and '净赚' in title:
        return False
    if '质押' in title and '股' in title:
        return False
    return True

'''
def fetch_data(text_file):
    title_filter_list = load_filter_list()
    title_filter_list = [ele.strip() for ele in title_filter_list]
    _text_data = []
    _raw_data = []
    print(text_file)
    with io.open(text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                if 'title' in json_data and 'content' in json_data and filter_title(json_data['title'], title_filter_list):
                        _raw_data.append(json_data)
                        title_ner = [term['word'] for term in json_data['seg_title'] if term['ner'] != 'O']
                        content_ner = [term['word'] for term in json_data['seg_content'] if term['ner'] != 'O']
                        content_ner.extend(title_ner)
                        _text_data.append(" ".join(content_ner))  # get content ner
            else:
                break
    return _text_data, _raw_data
'''

def fetch_data(text_file):
    temp_ners = ['证监会','商务部','习近平','政治局','会议','中共中央政治局','政治局','美国','特朗普','央行','人民币','上市公司','座谈会','声明','货币','中国','住建部','李克强','民营企业','民企','名单','国务院','保监会']
    stop_words = load_stop_words()
    title_filter_list = load_filter_list()
    title_filter_list = [ele.strip() for ele in title_filter_list]
    _text_data = []
    _raw_data = []
    
    if not os.path.exists(text_file):
        return _text_data, _raw_data
    
    count = 0
    with io.open(text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                try:
                    json_data = json.loads(line)
                    if 'title' in json_data and 'content' in json_data and filter_title(json_data['title'], title_filter_list):
                        count += 1
                        #print(count)
                        # _raw_data.append(json_data)
                        title_ner = [term['word'] for term in json_data['seg_title'] if term['word'] in temp_ners]
                        content_ner = [term['word'] for term in json_data['seg_content'] if term['word'] in temp_ners or term['nature'] == 'hy' or term['nature'] == 'st' or term['nature'] == 'nr' or term['nature'] == 'nto' or term['nature'] == 'nis' or term['nature'] == 'ns' or term['nature'] == 'nz' or term['nature'] == 'nt']
                        content_ner.extend(title_ner)
                        _text_data.append(content_ner)  # get content ner
                        #t_w = [term['word'] for term in json_data['seg_title'] if term['nature'].startswith('n') or term['nature'].startswith('v')]
                        t_w = [term['word'] for term in json_data['seg_title'] if term['nature'] != 'w' and term['nature'] != 't' and term['nature'] != 'f' and term['nature'] != 'm' and term['nature'] != 'c' and term['nature'] != 'cc' and term['nature'] != 'd' and term['word'] not in stop_words]
                        c_w = [term['word'] for term in json_data['seg_content'] if term['nature'] != 'w' and term['nature'] != 't' and term['nature'] != 'f' and term['nature'] != 'm' and term['nature'] != 'c' and term['nature'] != 'cc' and term['nature'] != 'd' and term['word'] not in stop_words]
                        c_w.extend(t_w)
                        json_data.pop('seg_title')
                        json_data.pop('seg_content')
                        json_data['ners'] = content_ner
                        json_data['words'] = c_w
                        json_data['title_words'] = t_w
                        _raw_data.append(json_data)
                except Exception as e:
                    continue
            else:
                break
    return _text_data, _raw_data

# end_time 应该指当前时间
def get_origin_cluster_result(origin_cluster_file_path, end_time, n_reserve_days_for_1size_cluster, n_reserve_days):
    origin_cluster_result = []
    if not os.path.exists(origin_cluster_file_path):
        logger.info("origin_cluster_result ids: {}".format(len(origin_cluster_result)))
        return origin_cluster_result
    with io.open(origin_cluster_file_path, 'r', encoding='utf-8') as f: 
        while True:
            line = f.readline()
            if len(line) > 0:
                try:
                    json_data = json.loads(line)
                except:
                    continue
                info_ids = json_data['info_ids']
                length = len(info_ids)
                publish_time = json_data['publish_time']
                # 簇大小为1：去掉时间跨度大于一天的簇
                if length == 1:
                    if end_time - publish_time > int(n_reserve_days_for_1size_cluster*24*60*60*1000):
                        continue
                # 簇大小大于1：去掉时间跨度大于三天的簇
                else:
                    if end_time - publish_time > int(n_reserve_days*24*60*60*1000):
                        continue
                origin_cluster_result.append(json_data)            
            else:
                break
    logger.info("origin_cluster_result ids: {}".format(len(origin_cluster_result)))
    return origin_cluster_result

def get_ner_lsi_online(corpus):
    dic_ner = corpora.Dictionary.load('model/dictionary_ner_model')
    corpus_ner = [dic_ner.doc2bow(text) for text in corpus]
    tfidf_ner = models.TfidfModel(corpus_ner)
    corpus_ner_tfidf = tfidf_ner[corpus_ner]
    lsi_ner_model = models.LsiModel.load('model/ner_lsi_model')
    lsi_ner_model.add_documents(corpus_ner_tfidf)
    lsi_ner_model.save('model/ner_lsi_model')
    corpus_ner_lsi = lsi_ner_model[corpus_ner_tfidf]
    return corpus_ner_lsi

def get_word_lsi_online(corpus):
    dic_word = corpora.Dictionary.load('model/dictionary_word_model')
    corpus_word = [dic_word.doc2bow(text) for text in corpus]
    tfidf_word = models.TfidfModel(corpus_word)
    corpus_word_tfidf = tfidf_word[corpus_word]
    lsi_word_model = models.LsiModel.load('model/word_lsi_model')
    lsi_word_model.add_documents(corpus_word_tfidf)
    lsi_word_model.save('model/word_lsi_model')
    corpus_word_lsi = lsi_word_model[corpus_word_tfidf]
    return corpus_word_lsi

'''
def get_tfidf_and_lsi(corpus):
    dictionary = corpora.Dictionary(corpus)
    length_of_dictionary = len(dictionary)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # TF-IDF特征
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # LSI特征
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=500)
    lsi_vectors = lsi[tfidf_vectors]
    return lsi_vectors
'''

# 更改为 torch
def get_tfidf_and_lsi(corpus):
    dictionary = corpora.Dictionary(corpus)
    length_of_dictionary = len(dictionary)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # TF-IDF特征
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # LSI特征
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=500)
    lsi_vectors = lsi[tfidf_vectors]
    vec = []
    num_topics = 500
    for i, ele in enumerate(lsi_vectors):
        feature = np.zeros(num_topics)
        for idx, val in ele:
            feature[idx] = val
        vec.append(feature)
    return vec

def computeSimilarity_lsm(X, query):
    index = similarities.MatrixSimilarity(X)
    sims = index[query]
    scoreList = list(enumerate(sims))
    rankList = [scoreList[i][1] for i in range(len(scoreList))]
    return rankList

def get_clusters_score(_cluster_result, words, num_topics):
    dictionary = corpora.Dictionary(words)
    doc_vectors = [dictionary.doc2bow(text) for text in words]
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=num_topics)
    lsi_vectors = lsi[tfidf_vectors]

    _cluster_result_score = {}
    for key in _cluster_result:
        score = 0
        indexs = _cluster_result[key]
        length = len(indexs)
        if length == 1:
            _cluster_result_score[key] = 1
        else:
            X = []
            for _id in indexs:
                X.append(lsi_vectors[_id])
            X_score = []
            for j in range(length):
                query = X[j]
                scoreList = computeSimilarity_lsm(X, query)
                X_score.append(scoreList)
            num_of_compute = length * length - length
            # 一个簇的平均文本相似度
            score = (sum([sum(ele) for ele in X_score]) - length) / num_of_compute
            _cluster_result_score[key] = score
    return _cluster_result_score

def list2dict(listObj, num):
    dictObj = {}
    for ele in listObj:
        if ele[0] not in dictObj:
            dictObj[ele[0]] = ele[1]
        else:
            dictObj[ele[0]] += ele[1]
    for k in dictObj.keys():
        v = dictObj[k]
        v = v / num
        dictObj[k] = v
    dictObj = sorted(dictObj.items(), key=lambda d: d[1], reverse=True)
    return dictObj

def get_cluster_keywords(_cluster_result, _cluster_result_score, texts, Threshold):
    _cluster_keywords = {}
    for key in _cluster_result:
        #if _cluster_result_score[key] < Threshold:
        #     continue
        indexs = _cluster_result[key]
        k = []
        for _id in indexs:
            news = texts[_id]
            news_k = textrank(news, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'nrt', 'j', 'v', 'vn'))
            k.extend(news_k)
        k_sort = list2dict(k, len(indexs))
        k_sort_5 = [ele[0] for ele in k_sort][:10]
        _cluster_keywords[key] = k_sort_5
    return _cluster_keywords

def get_cluster_keywords_from_titles(_cluster_result, _cluster_result_score, texts, Threshold):
    _cluster_keywords = {}
    for key in _cluster_result:
        #if _cluster_result_score[key] < Threshold:
        #    continue
        indexs = _cluster_result[key]
        words = {}
        for _id in indexs:
            news = texts[_id]
            for word in news:
                if word not in words.keys():
                    words[word] = 1
                else:
                    words[word] += 1 
        
        #logger.info('words_'+str(words))
        words_ = sorted(words.items(), key=lambda d:d[1], reverse = True)
        # words__ 可能数量可能为 1
        words__ = [word[0] for word in words_][0:5]
        # 不应该用 dict，打印出来会没有顺序
        title_keywords_dic = {}
        for word, count in words_:
            title_keywords_dic[word] = count
                              
        _cluster_keywords[key] = {"keywords": words__,
                                 "title_keywords_dic": title_keywords_dic}
    return _cluster_keywords

def get_keywords_all_title_words(all_title_words):
    
    x = {}
    for title_words in all_title_words:
        for word in title_words:
            if word not in x.keys():
                x[word] = 1
            else:
                x[word] += 1 

    x_ = sorted(x.items(), key=lambda d:d[1], reverse = True)
    title_keywords = [word[0] for word in x_][0:5]
    title_keywords_dic = {}
    for word, count in x_:
        title_keywords_dic[word] = count
    
    return title_keywords, title_keywords_dic

def title_choose_from_all_title(all_title, all_title_words, all_publish_time, keywords):
    
    m = {}
    x = []
    for i in range(0,len(all_publish_time)):
        x.append((all_title[i], all_publish_time[i]))
        m[all_title[i]] = all_title_words[i]
    x_ = sorted(x, key=lambda d:d[1], reverse = True)
    x__ = [word[0] for word in x_][0:3]
    
    title_ = ''
    max_size = 0
    for title in x__:
        size = len(set(keywords) & set(m[title]))
        if size >= max_size:
            max_size = size
            title_ = title
    
    return title_

def cluster(origin_cluster_result, text_data, raw_data):
    center_texts = []
    center_text_data = []
    center_words = []

    origin_texts = []
    cluster_texts = []
    origin_cluster_texts = []
    cluster_titles = []
    count = -1
    for ele in origin_cluster_result:
        condition = []
        key = ele['id']
        info_ids_to_data = ele['info_ids_to_data']
        origin_cluster = []
        for item in info_ids_to_data:
            count += 1
            title = item['title']
            content = item['content']
            text = title + '' + content
            title_words = item['title_words']
            if item['id'] == key:   
                if key not in condition:
                    origin_cluster.append(count)
                    condition.append(key)
                    center_texts.append(text)
                    cluster_titles.append(title_words)
                    cluster_texts.append(text)
            else:
                origin_cluster.append(count)
                origin_texts.append(text)
                cluster_titles.append(title_words)
                cluster_texts.append(text)
        origin_cluster_texts.append(origin_cluster)
        
    origin_text_data = []
    for ele in origin_cluster_result:
        condition = []
        key = ele['id']
        info_ids_to_data = ele['info_ids_to_data']
        for item in info_ids_to_data:
            content_ner = item['ners']
            if item['id'] == key:
                if key not in condition:
                    condition.append(key)
                    center_text_data.append(content_ner)
            else:
                origin_text_data.append(content_ner)

    origin_words = []
    cluster_words = []
    for ele in origin_cluster_result:
        condition = []
        key = ele['id']
        info_ids_to_data = ele['info_ids_to_data']
        for item in info_ids_to_data:
            c_w = item['words']
            if item['id'] == key:
                if key not in condition:
                    condition.append(key)
                    center_words.append(c_w)
                    cluster_words.append(c_w)
            else:
                origin_words.append(c_w)
                cluster_words.append(c_w)
    #logger.info("center_texts: "+str(len(center_texts)))
    #logger.info("center_text_data: "+str(len(center_text_data)))
    #logger.info("center_words: "+str(len(center_words)))
    texts = [ele['title'] + ' ' + ele['content'] for ele in raw_data]
    titles = [ele['title_words'] for ele in raw_data]
    #text_data = [ele.split() for ele in text_data]
    words = [ele['words'] for ele in raw_data]

    len_of_center = len(center_text_data)
    #logger.info("len_of_center: "+str(len_of_center))
    len_of_origin = len_of_center + len(origin_text_data)
    len_of_new = len(raw_data)

    all_text_data = center_text_data.copy()
    all_text_data.extend(origin_text_data)
    all_text_data.extend(text_data)

    all_words = center_words.copy()
    all_words.extend(origin_words)
    all_words.extend(words)
    logger.info("lsi_vectors_ner = get_tfidf_and_lsi(all_text_data) ")
    lsi_vectors_ner = get_tfidf_and_lsi(all_text_data)
    lsi_vectors_word = get_tfidf_and_lsi(all_words)
    logger.info("lsi_vectors_word = get_tfidf_and_lsi(all_words) finished")
    #del all_text_data
    #del all_words
    all_text_data = None
    all_words = None
    # feature_ner = lsi_vectors_ner[:len_of_center]
    # feature_ner_new = lsi_vectors_ner[len_of_origin:]
    # feature_ner.extend(feature_ner_new)
    feature_ner = []
    for i in range(len(lsi_vectors_ner)):
        if i < len_of_center or i >= len_of_origin:
            feature_ner.append(lsi_vectors_ner[i])
    # feature_word = lsi_vectors_word[:len_of_center]
    # feature_word_new = lsi_vectors_word[len_of_origin:]
    # feature_word.extend(feature_word_new)
    feature_word = []
    for i in range(len(lsi_vectors_word)):
        if i < len_of_center or i >= len_of_origin:
            feature_word.append(lsi_vectors_word[i])
    
    #del lsi_vectors_ner
    #del lsi_vectors_word
    lsi_vectors_ner = None
    lsi_vectors_word = None
    
    result = {}
    if len_of_origin == 0:
        result = {0: [0]}
    else:
        for i in range(len_of_center):
            result[i] = [i]
    '''
    for i, _ in enumerate(feature_ner):
        print(i)
        if len_of_center == 0 and i == 0:
            continue
        if i < len_of_center:
            continue
        feature_ner_lsi_now = feature_ner[i]
        feature_word_lsi_now = feature_word[i]
        feature_ner_lsi = []
        feature_word_lsi = []
        for key in result:
            index = result[key][0]
            vec_lsi = feature_ner[index]
            vec_word = feature_word[index]
            feature_ner_lsi.append(vec_lsi)
            feature_word_lsi.append(vec_word)

        index_ner_lsi = similarities.MatrixSimilarity(feature_ner_lsi)
        sims_ner_lsi = index_ner_lsi[feature_ner_lsi_now]
        index_word_lsi = similarities.MatrixSimilarity(feature_word_lsi)
        sims_word_lsi = index_word_lsi[feature_word_lsi_now]

        sims_ner = dict(enumerate(sims_ner_lsi))
        sims_word = dict(enumerate(sims_word_lsi))

        sims_ner_sort = sorted(sims_ner.items(), key=lambda d: d[1], reverse=True)
        sims_word_sort = sorted(sims_word.items(), key=lambda d: d[1], reverse=True)

        max_score_word = sims_word_sort[0][1]
        max_score_word_index = sims_word_sort[0][0]
        max_score_ner = sims_ner[max_score_word_index]

        if max_score_word >= 0.8 and max_score_ner >= 0.35:
            result[max_score_word_index].append(i)
        elif max_score_word >= 0.9 and max_score_ner == 0.0:
            result[max_score_word_index].append(i)
        else:
            result[len(result)] = [i]
    '''
    # 更改为 torch
    for i, _ in enumerate(feature_ner):
        print(i)
        if len_of_center == 0 and i == 0:
            continue
        if i < len_of_center:
            continue
        feature_ner_lsi_now = feature_ner[i]
        feature_word_lsi_now = feature_word[i]
        feature_ner_lsi = []
        feature_word_lsi = []
        for key in result:
            index = result[key][0]
            vec_lsi = feature_ner[index]
            vec_word = feature_word[index]
            feature_ner_lsi.append(vec_lsi)
            feature_word_lsi.append(vec_word)
        
        feature_ner_lsi_now_t = torch.Tensor(feature_ner_lsi_now).unsqueeze(0)
        feature_word_lsi_now_t = torch.Tensor(feature_word_lsi_now).unsqueeze(0)
        feature_ner_lsi_t = torch.Tensor(feature_ner_lsi)
        feature_word_lsi_t = torch.Tensor(feature_word_lsi)
        feature_ner_lsi_t = feature_ner_lsi_t.view(-1, 500)
        feature_word_lsi_t = feature_word_lsi_t.view(-1, 500)

        sims_ner_lsi = nn.functional.cosine_similarity(feature_ner_lsi_t, feature_ner_lsi_now_t)
        sims_word_lsi = nn.functional.cosine_similarity(feature_word_lsi_t, feature_word_lsi_now_t)
        sims_ner_lsi = sims_ner_lsi.numpy().tolist()
        sims_word_lsi = sims_word_lsi.numpy().tolist()

        sims_ner = dict(enumerate(sims_ner_lsi))
        sims_word = dict(enumerate(sims_word_lsi))

        sims_ner_sort = sorted(sims_ner.items(), key=lambda d: d[1], reverse=True)
        sims_word_sort = sorted(sims_word.items(), key=lambda d: d[1], reverse=True)

        max_score_word = sims_word_sort[0][1]
        max_score_word_index = sims_word_sort[0][0]
        max_score_ner = sims_ner[max_score_word_index]

        if max_score_word >= 0.8 and max_score_ner >= 0.35:
            result[max_score_word_index].append(i)
        elif max_score_word >= 0.9 and max_score_ner == 0.0:
            result[max_score_word_index].append(i)
        else:
            result[len(result)] = [i]
    
    _cluster_result = result
    num_of_clusters = len(_cluster_result)
    Threshold = 0.5
    num_topics = 500

    _cluster_result_for_keywords = {}
    for key in _cluster_result:
        if key < len_of_center:
            indexs = _cluster_result[key][1:]
            indexs = [ele + len(origin_text_data) for ele in indexs]
            origin_indexs = origin_cluster_texts[key]
            origin_indexs.extend(indexs)
            _cluster_result_for_keywords[key] = origin_indexs
        else:
            indexs = _cluster_result[key]
            indexs = [ele + len(origin_text_data) for ele in indexs]
            _cluster_result_for_keywords[key] = indexs
    
    # 获取每个簇的得分
    cluster_words.extend(words)
    #_cluster_result_score = get_clusters_score(_cluster_result, center_words, num_topics)
    _cluster_result_score = get_clusters_score(_cluster_result_for_keywords, cluster_words, num_topics)
    cluster_words = None
    # 获取每个簇的关键词
    cluster_titles.extend(titles)
    _cluster_title_keywords = get_cluster_keywords_from_titles(_cluster_result_for_keywords, _cluster_result_score, cluster_titles, Threshold)
    cluster_titles = None
    # 获取每个簇的关键词
    cluster_texts.extend(texts)
    _cluster_keywords = get_cluster_keywords(_cluster_result_for_keywords, _cluster_result_score, cluster_texts, Threshold)
    cluster_texts = None
    #logger.info('_cluster_result: '+str(len(_cluster_result)))
    for key in _cluster_result:
        # 更新簇
        if key < len_of_center:
            origin_cluster_result[key]['score'] = _cluster_result_score[key]
            origin_cluster_result[key]['content_keywords'] = _cluster_keywords[key]
            indexs = _cluster_result[key][1:]
            #logger.info("indexs: "+str(indexs))
            publish_time = origin_cluster_result[key]['publish_time']
            all_titles_in_cluster = origin_cluster_result[key]['all_titles']
            all_titlewords_in_cluster = origin_cluster_result[key]['all_title_words']
            all_publishtime_in_cluster = origin_cluster_result[key]['all_publishtime']
            for index in indexs:
                data = raw_data[index-len_of_center]
                _id = data['id']
                origin_cluster_result[key]['info_ids'].append(_id)
                origin_cluster_result[key]['info_ids_to_data'].append(data)

                all_titles_in_cluster.append(data['title'])
                all_titlewords_in_cluster.append(data['title_words'])
                all_publishtime_in_cluster.append(data['publishAt'])
                publish_time = max(publish_time, data['publishAt'])
            
            keywords, title_keywords_dic = get_keywords_all_title_words(all_titlewords_in_cluster)
            
            # 根据标题关键词选取标题
            title = ''
            max_size = 0
            for i in range(0, len(all_titlewords_in_cluster)):
                size = len(set(keywords) & set(all_titlewords_in_cluster[i]))
                if  size > max_size:
                    max_size = size
                    title = all_titles_in_cluster[i]
                    
            origin_cluster_result[key]['keywords'] = keywords
            origin_cluster_result[key]['title_keywords_dic'] = title_keywords_dic
            origin_cluster_result[key]['title'] = title
            origin_cluster_result[key]['publish_time'] = publish_time
            origin_cluster_result[key]['all_titles'] = all_titles_in_cluster
            origin_cluster_result[key]['all_title_words'] = all_titlewords_in_cluster
            origin_cluster_result[key]['all_publishtime'] = all_publishtime_in_cluster
            origin_cluster_result[key]['min_publishtime'] = min(all_publishtime_in_cluster)
            origin_cluster_result[key]['max_publishtime'] = max(all_publishtime_in_cluster)
        # 未能与已有簇进行合并的数据的聚类处理
        else:
            #if len(_cluster_result[key]) < 2:
            #    continue
            if _cluster_result_score[key] < Threshold:
                continue
            '''
            top_term = dict()
            terms = _cluster_title_keywords[key]['keywords']
            for term in terms:
                top_term[term] = 0

            for index in _cluster_result[key]:
                for term in top_term.keys():
                    word_list = raw_data[index-len_of_center]['words']
                    if term in word_list:
                        top_term[term] += 1

            filter_term = ['减持', '市场', '中国', '增持', '半年报', '业绩', '价格',
                           '投资', '项目', '融资', '完成', '协议', '合作', '股份',
                           '回购', '战略', '股市', '亿元', '股东', '经济', '震荡',
                           '评级', '上半年', '净利', '机会', '美元', '人民币', '持续',
                           '企业', '基金', '如何', '关注', '中标', '预期', '利润',
                           '欧元', '英镑', '两市', '多赚', '调研', '接待', '机构',
                           '订单', '纯利', '年报', '半年报', '资本', '私募', '发展',
                           '科技', 'a股', '美国', '继续', '上市', '建设', '反弹',
                           '资金', '实现', '积极', '万元', '银行', '成交', '增长',
                           '辞职', '平台', '发布', '同比', '亏损', '国家', '净利润',
                           '涨停', '消费', '港股', 'ipo', '分公司', '城市', '公司',
                           '上涨', '金融', '世界', '分析', '溢利', '政策', '黄金',
                           '应占', '活跃', '质押', '盈利', '回落', '公告', '创新',
                           'gdp', '数据', '扩大', '调整', '减少', '关于', '走势',
                           '助力', '携手', '布局', '国际', '打造', '指数', '收购',
                           '第一', '期货', '控股', '目标价', '签署', '披露', '总股本',
                           '募资', '万亿', '耗资', '风险', '板块', '子公司', '市值',
                           '质量', '设立', '整体', '弱势', '提示', '规模', '开盘',
                           '销售额', '研发', '合同', '公开', '合约', '建议', '跌幅',
                           '重大', '购买', '政府', '新高', '可期', '央行', '销售',
                           '主席', '走强', '机遇', '维持', '期权']
            event_cluster = False
            '''
            '''
            # 过滤掉杂簇
            logger.info('---------------')
            logger.info("Cluster size %d:" % len(_cluster_result[key]))
            logger.info(top_term)
            logger.info('---------------')
            '''
            '''
            # 关键词占比大于一定阈值，认为是事件簇
            for term in top_term.keys():
                if term not in filter_term and top_term[term] * 1.0 / len(_cluster_result[key]) > 0.75:
                    event_cluster = True
                    break

            # 第一个关键词的词频比第二个大很多的，认为不是事件簇
            if len(top_term) == 0:
                continue
            
            try:
                sorted_terms = [v for v in sorted(top_term.values(), reverse=True)]
                first_term_count = sorted_terms[0]
                second_term_count = sorted_terms[1]
                if ((first_term_count - second_term_count) * 1.0 / len(_cluster_result[key])) > 0.75:
                    event_cluster = False
            except Exception as e:
                logger.info("len(top_term) == 1 关键词只有1个 " % top_term)
                continue

            if not event_cluster:
                continue
            '''

            score = _cluster_result_score[key]
            keywords = []
            info_ids = []
            info_ids_to_data = []
            publish_time = 0
            #logger.info("Cluster %d:" % key)
            for ele in _cluster_title_keywords[key]['keywords']:
                #logger.info(' %s' % ele)
                keywords.append(ele)
                
            # 拿到所有标题以及标题的时间
            all_titles_in_cluster = []
            all_titlewords_in_cluster = []
            all_publishtime_in_cluster = []
            for index in _cluster_result[key]:
                #logger.info(raw_data[index-len_of_center]['title'])
                info_ids.append(raw_data[index-len_of_center]['id'])
                info_ids_to_data.append(raw_data[index-len_of_center])
                all_titles_in_cluster.append(raw_data[index-len_of_center]['title'])
                all_titlewords_in_cluster.append(raw_data[index-len_of_center]['title_words'])
                all_publishtime_in_cluster.append(raw_data[index-len_of_center]['publishAt'])
                
                publish_time = max(publish_time, raw_data[index-len_of_center]['publishAt'])
            
            # 根据标题关键词选取标题
            title = ''
            max_size = 0
            for i in range(0, len(all_titlewords_in_cluster)):
                size = len(set(keywords) & set(all_titlewords_in_cluster[i]))
                if  size > max_size:
                    max_size = size
                    title = all_titles_in_cluster[i]
                    
            cluster_dict = dict()
            cluster_dict['id'] = info_ids[0]
            cluster_dict['keywords'] = keywords
            cluster_dict['title_keywords_dic'] =_cluster_title_keywords[key]['title_keywords_dic']
            cluster_dict['info_ids'] = info_ids
            cluster_dict['info_ids_to_data'] = info_ids_to_data
            cluster_dict['title'] = title
            cluster_dict['publish_time'] = publish_time
            cluster_dict['all_titles'] = all_titles_in_cluster
            cluster_dict['all_title_words'] = all_titlewords_in_cluster
            cluster_dict['all_publishtime'] = all_publishtime_in_cluster
            cluster_dict['score'] = score
            cluster_dict['content_keywords'] = _cluster_keywords[key]
            cluster_dict['min_publishtime'] = min(all_publishtime_in_cluster)
            cluster_dict['max_publishtime'] = max(all_publishtime_in_cluster)
            
            origin_cluster_result.append(cluster_dict)
            
    return origin_cluster_result

if __name__ == '__main__':
    data_file = 'logs/1.txt'
    ner_content_data, raw_data = fetch_data(data_file)
    logger.info('cluster size: ' + str(len(raw_data)))
    #end_time = time_utils.current_milli_time()
    end_time = 1540161900000
    origin_cluster_result = get_origin_cluster_result(end_time)
    length_data = len(origin_cluster_result)
    logger.info('origin cluster size: ' + str(length_data))
    cluster_result = cluster(origin_cluster_result, ner_content_data, raw_data)
    fout = codecs.open('logs/origin_cluster.txt', 'w', encoding='utf-8')
    for ele in cluster_result:
        strObj = json.dumps(ele, ensure_ascii=False)
        fout.write(strObj+'\n')
    fout.close()
    for ele in cluster_result:
        score = ele['score']
        keywords = ele['keywords']
        print(score)
        print(keywords)
        info_ids_to_data = ele['info_ids_to_data']
        for ele in info_ids_to_data:
            title = ele['title']
            print(title)
        print('---------------------------------------')
