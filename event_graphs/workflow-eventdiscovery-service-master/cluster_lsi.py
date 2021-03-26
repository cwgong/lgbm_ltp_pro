# -*- encoding:utf8 -*-
'''
实体和关键词相似度合并,用textrank找出关键词，在计算tfidf向量时增加关键词的权重，pytorch计算相似度，聚类中心用所有文本LSI向量的均值表示
'''
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
import pickle
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from political_title_supervision.political_title_supervision import Political_Title_Supervision
from event_description_class import Event_Description_Supervision

jieba.load_userdict('WordsDic/userdict_.txt')
jieba.analyse.set_stop_words('WordsDic/stopwords.txt')

logging.config.fileConfig("logging.conf")
logger = logging.getLogger('main')

def load_stop_words():
    
    with codecs.open('WordsDic/stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [x.strip() for x in f.readlines()]
    return stop_words

def fetch_data(text_file):

    # 过滤掉「偏公司行业经济类」资讯
    political_title_supervision = Political_Title_Supervision()
    # 寻找文章转述型概述
    event_description_supervison = Event_Description_Supervision()
    # ners 实体列表
    temp_ners = []
    stop_words = load_stop_words()
    
    _ner_data = []
    _word_data = []
    _text_data = []
    _title_word_data = []
    _raw_data = []
    p_count = 0
    n_count = 0
    with io.open(text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                if 'title' in json_data and 'content' in json_data:
                    logger.info(json_data['title'] + ' ' + str(political_title_supervision.f(json_data['title'])))
                    # 判断是否为政策类事件，f() 需要分词
                    if political_title_supervision.f(json_data['title'], title_seg = json_data['seg_title']):
                        p_count += 1
                        title_ner = [term['word'] for term in json_data['seg_title'] if term['ner'] != 'O' or term['word'] in temp_ners]
                        content_ner = [term['word'] for term in json_data['seg_content'] if term['ner'] != 'O' or term['word'] in temp_ners]
                        content_ner.extend(title_ner)
                        _ner_data.append(content_ner)
                        t_w = [term['word'] for term in json_data['seg_title'] if
                               term['word'] not in stop_words and term['nature'].startswith('n') or term['nature'].startswith('v')]
                        c_w = [term['word'] for term in json_data['seg_content'] if
                               term['word'] not in stop_words and term['nature'].startswith('n') or term['nature'].startswith('v')]
                        c_w.extend(t_w)
                        _title_word_data.append(t_w)
                        _word_data.append(c_w)
                        _text_data.append(json_data['title'] + ' ' +json_data['content'])
                        json_data['ners'] = content_ner
                        json_data['words'] = c_w
                        json_data['title_words'] = t_w
                        json_data['text'] = json_data['title'] + ' ' +json_data['content']
                        _raw_data.append(json_data)
                    else:
                        n_count += 1
            else:
                break
    
    logger.info("fetch_data p_count: {}".format(p_count))
    logger.info("fetch_data n_count: {}".format(n_count))
    return _ner_data, _word_data, _text_data, _title_word_data, _raw_data

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

def get_tfidf_and_lsi(corpus, texts):
    # 根据texts获取每个text的textrank关键词，将corpus中关键词复制weight份，即提升关键词的权重
    keywords = []
    for i, text in enumerate(texts):
        text_k = textrank(text, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'nrt', 'j', 'v', 'vn'))
        keywords.append(text_k)
        words = corpus[i]
        weight = len(text_k)
        for word in text_k:
            if word[0] in words:
                words.extend(weight*[word[0]])
                weight -= 1
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
    for i, ele in enumerate(lsi_vectors):
        feature = np.zeros(500)
        for idx, val in ele:
            feature[idx] = val
        vec.append(feature)
    return vec, lsi_vectors, keywords

def get_ner_lsi(corpus):
    dic_ner = corpora.Dictionary.load('model/dictionary_ner_model')
    corpus_ner = [dic_ner.doc2bow(text) for text in corpus]
    tfidf_ner = models.TfidfModel(corpus_ner)
    corpus_ner_tfidf = tfidf_ner[corpus_ner]
    lsi_ner_model = models.LsiModel.load('model/ner_lsi_model')
    corpus_ner_lsi = lsi_ner_model[corpus_ner_tfidf]
    return corpus_ner_lsi

#只用预训练的文本构建的词典
def get_word_lsi(corpus):
    dic_word = corpora.Dictionary.load('model/dictionary_word_model')
    corpus_word = [dic_word.doc2bow(text) for text in corpus]
    tfidf_word = models.TfidfModel(corpus_word)
    corpus_word_tfidf = tfidf_word[corpus_word]
    lsi_word_model = models.LsiModel.load('model/word_lsi_model')
    corpus_word_lsi = lsi_word_model[corpus_word_tfidf]
    return corpus_word_lsi

#新的文本会加入词典中进行预训练
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

def get_cluster_keywords(_cluster_result, keywords):
    _cluster_keywords = {}
    for key in _cluster_result:
        indexs = _cluster_result[key]
        k = []
        for _id in indexs:
            news_k = keywords[_id]
            k.extend(news_k)
        k_sort = list2dict(k, len(indexs))
        k_sort_5 = [ele[0] for ele in k_sort][:5]
        _cluster_keywords[key] = k_sort_5
    return _cluster_keywords

def get_cluster_keywords_from_titles(_cluster_result, texts):
    _cluster_keywords = {}
    for key in _cluster_result:
        indexs = _cluster_result[key]
        words = {}
        for _id in indexs:
            news = texts[_id]
            for word in news:
                if word not in words.keys():
                    words[word] = 1
                else:
                    words[word] += 1
        words_ = sorted(words.items(), key=lambda d: d[1], reverse=True)
        # words__ 可能数量可能为 1
        words__ = [word[0] for word in words_][0:5]
        title_keywords_dic = {}
        for word, count in words_:
            title_keywords_dic[word] = count
        _cluster_keywords[key] = {"keywords": words__, "title_keywords_dic": title_keywords_dic}
    return _cluster_keywords

# def get_filter_terms():
#     terms = []
#     with open('WordsDic/filter_term', 'r', encoding='utf8') as f:
#         for line in f:
#             terms.append(line.strip())
#     return terms
#用于后处理的代码，已无效

def cluster(origin_cluster_result, ner_content_data, word_content_data, text_data, word_title_data, raw_data):
    origin_ners = []
    origin_words = []
    origin_texts = []
    origin_title_words = []
    origin_cluster_index = []
    origin_raw_data = []
    count = 0
    for ele in origin_cluster_result:
        indexs = []
        info_ids_to_data = ele['info_ids_to_data']
        for item in info_ids_to_data:
            indexs.append(count)
            count += 1
            origin_ners.append(item['ners'])#ner，命名实体识别
            origin_words.append(item['words'])#分词
            origin_texts.append(item['text'])#所有文本
            origin_title_words.append(item['title_words'])#标题关键词
            origin_raw_data.append(item)#所有的信息
        origin_cluster_index.append(indexs)

    all_ners = origin_ners.copy()
    all_ners.extend(ner_content_data)
    all_words = origin_words.copy()
    all_words.extend(word_content_data)
    all_texts = origin_texts.copy()
    all_texts.extend(text_data)
    all_title_words = origin_title_words.copy()
    all_title_words.extend(word_title_data)
    all_raw_data = origin_raw_data.copy()
    all_raw_data.extend(raw_data)

    num_of_origin_clusters = len(origin_cluster_index)
    len_of_origin = len(origin_ners)
    len_of_all = len(all_ners)

    for i in range(len_of_all):
        all_ners[i].extend(all_words[i])
    lsi, _, all_keywords = get_tfidf_and_lsi(all_ners, all_texts)
    result = {0: [0]}
    for i in range(num_of_origin_clusters):
        result[i] = origin_cluster_index[i]
    for i in range(len_of_all):
        print(i)
        if i <len_of_origin or i == 0:
            continue
        feature_lsi_now = lsi[i]
        feature_lsi = []
        for key in result:
            ids = result[key]
            lsi_ = np.array([lsi[_id] for _id in ids])
            lsi_center = np.mean(lsi_, axis=0)
            feature_lsi.append(lsi_center)

        feature_lsi_now_t = torch.Tensor(feature_lsi_now).unsqueeze(0)
        feature_lsi_t = torch.Tensor(feature_lsi)
        feature_lsi_t = feature_lsi_t.view(-1, 500)
        sims_lsi = nn.functional.cosine_similarity(feature_lsi_t, feature_lsi_now_t)
        max_score, max_score_index = torch.max(sims_lsi, 0)
        max_score = max_score.item()
        max_score_index = max_score_index.item()

        if max_score >= 0.7:
            result[max_score_index].append(i)
        else:
            result[len(result)] = [i]

    _cluster_result = result
    num_of_clusters = len(_cluster_result)
    Threshold = 0.0
    num_topics = 500

    # 获取每个簇的得分（words相似度）
    _cluster_result_score = get_clusters_score(_cluster_result, all_ners, num_topics)
    # 获取每个簇的标题关键词
    _cluster_title_keywords = get_cluster_keywords_from_titles(_cluster_result, all_title_words)
    # 获取每个簇的textrank关键词
    _cluster_keywords = get_cluster_keywords(_cluster_result, all_keywords)

    new_origin_cluster_result = []

    for key in _cluster_result:
        #if key < len_of_center:
        #更新簇或者加入新簇
        if _cluster_result_score[key] < Threshold:
            continue

        # all_titles_in_cluster = origin_cluster_result[key]['all_titles']
        # all_titlewords_in_cluster = origin_cluster_result[key]['all_title_words']
        # all_publishtime_in_cluster = origin_cluster_result[key]['all_publishtime']

        keywords = []
        info_ids = []
        info_ids_to_data = []
        title = ''
        publish_time = 0
        all_titles_in_cluster = []
        all_titlewords_in_cluster = []
        all_publishtime_in_cluster = []
        #publish_time = 0
        #输出可视结果，保存到文件
        #logger.info("Cluster %d:" % key)
        max_size = 0
        min_words = 0
        for ele in _cluster_title_keywords[key]['keywords']:
            #logger.info(' %s' % ele)
            keywords.append(ele)
        for index in _cluster_result[key]:
            #logger.info(all_raw_data[index]['title'])
            info_ids.append(all_raw_data[index]['id'])
            info_ids_to_data.append(all_raw_data[index])
            title_words = all_raw_data[index]['title_words']

            # 根据标题关键词选取标题,原则1：关键词覆盖度，原则二：关键词覆盖相同，取title-keywords少的那种
            size = len(set(keywords) & set(title_words))
            if size > max_size:
                max_size = size
                min_words=len(set(title_words))
                title = all_raw_data[index]['title']
            elif size == max_size:
                if min_words>len(set(title_words)):
                    min_words=len(set(title_words))
                    title=all_raw_data[index]['title']
            # size = len(top_term.keys() & set(title_words))
            # if size > max_size:
            #     max_size = size
            #     title = all_raw_data[index]['title']
            all_titles_in_cluster.append(all_raw_data[index]['title'])
            all_titlewords_in_cluster.append(all_raw_data[index]['title_words'])
            all_publishtime_in_cluster.append(all_raw_data[index]['publishAt'])
            publish_time = max(publish_time, all_raw_data[index]['publishAt'])


        cluster_dict = dict()
        cluster_dict['keywords']=keywords#title-keywords
        cluster_dict['id'] = info_ids[0]
        cluster_dict['info_ids'] = info_ids
        cluster_dict['info_ids_to_data'] = info_ids_to_data
        cluster_dict['title'] = title
        cluster_dict['publish_time'] = publish_time
        cluster_dict['score'] = _cluster_result_score[key]
        cluster_dict['title_keywords_dic'] = _cluster_title_keywords[key]['title_keywords_dic']#有词频信息的title-keywords
        cluster_dict['content_keywords'] = _cluster_keywords[key]#title-keywords+content-keywords
        cluster_dict['all_titles'] = all_titles_in_cluster
        cluster_dict['all_title_words'] = all_titlewords_in_cluster
        cluster_dict['all_publishtime'] = all_publishtime_in_cluster
        cluster_dict['min_publishtime'] = min(all_publishtime_in_cluster)
        cluster_dict['max_publishtime'] = max(all_publishtime_in_cluster)
        new_origin_cluster_result.append(cluster_dict)

    s_sort = sorted(new_origin_cluster_result, key=lambda x: len(x['info_ids']), reverse = True)
    return s_sort




# 每次聚类后更新 publishAt、title cluster keywords、triple

feature_engineering:
    
cluster_info:
    cluster_id = ''
    score = ''
    article_ids = []
    article_ids_detail = [{'ori_info':,
                           'event_description':,
                           'title_segment':,
                           'event_description_segment':,
                           'features': 去停用词 v n and ner and n_gram
                           
                           
                            }] sort_dict
    title_keywords = [{word: count}] sort_dict
    content_keywords = [{word: count}] sort_dict
    min_publishAt = ''
    max_publishAt = ''
    
    



paragraphs = event_description_supervison.split_content_to_paragraph(json_data['content'], json_data['dataSource'])
                        event_description = event_description_supervison.get_event_description_from_article(paragraphs)
                        
                        if event_description != '':
                            





if __name__ == '__main__':
    t0 = datetime.now()
    data_file = 'logs/20181112.txt'
    ner_content_data, word_content_data, text_data, word_title_data, raw_data = fetch_data(data_file)
    logger.info('cluster size: ' + str(len(raw_data)))
    end_time = time_utils.current_milli_time()
    origin_cluster_result = get_origin_cluster_result(end_time)
    length_data = len(origin_cluster_result)
    logger.info('origin cluster size: ' + str(length_data))
    cluster_result = cluster(origin_cluster_result, ner_content_data, word_content_data, text_data, word_title_data, raw_data)

    fout = codecs.open('logs/origin_cluster.txt', 'w', encoding='utf-8')
    for ele in cluster_result:
        strObj = json.dumps(ele, ensure_ascii=False)
        fout.write(strObj+'\n')
    fout.close()

    for ele in cluster_result:
        score = ele['score']
        keywords = ele['title_keywords_dic']
        keywords_content = ele['keywords']
        info_ids_to_data = ele['info_ids_to_data']
        if len(info_ids_to_data) == 1:
            continue
        print(score)
        print(keywords)
        print(keywords_content)
        for ele in info_ids_to_data:
            title = ele['title']
            print(title)
        print('---------------------------------------')
    t1 = datetime.now()
    print(t0, t1, t1 - t0)
