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
from event_project_utils import semantic_clean, get_special_chunk, event_merge, cluster_event_complete
import requests
from ltp_class import HIT_LTP
from get_date_chunk import Date_chunk_handle
import time



jieba.load_userdict('WordsDic/userdict_.txt')
jieba.analyse.set_stop_words('WordsDic/stopwords.txt')

logging.config.fileConfig("logging.conf")
logger = logging.getLogger('main')

def split_sentence(sen):
    nlp_url = 'http://hanlp-nlp-service:31001/hanlp/segment/nlp'
    try:
        cut_sen = dict()
        cut_sen['content'] = sen
        cut_sen['customDicEnable'] = True
        data = json.dumps(cut_sen).encode("UTF-8")
        cut_response = requests.post(nlp_url, data=data, headers={'Connection':'close'})
        cut_response_json = cut_response.json()
        return cut_response_json['data']
    except Exception as e:
        logger.exception("Exception: {}".format(e))
        logger.exception("sentence: {}".format(sen))
        return []
    
def load_stop_words():
    with codecs.open('WordsDic/stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [x.strip() for x in f.readlines()]
    return stop_words

def fetch_data(text_file):
    print('text_file: ', text_file)
    temp_ners = []
    stop_words = load_stop_words()
    
    # 过滤掉「偏公司行业经济类」资讯
    #political_title_supervision = Political_Title_Supervision()
    
    _ner_data = []
    _word_data = []
    _text_data = []
    _title_word_data = []
    _raw_data = []
    count = 0
    
    with io.open(text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                if 'title' in json_data and 'content' in json_data:
                    
                    title_ner = [term['word'] for term in json_data['seg_title'] if term['nature'] in ['ni', 'ns', 'nh', 'nz'] or term['word'] in temp_ners]
                    content_ner = [term['word'] for term in json_data['seg_content'] if term['nature'] in ['ni', 'ns', 'nh', 'nz'] or term['word'] in temp_ners]
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
                    json_data.pop('seg_title')
                    json_data.pop('seg_content')
                    json_data['ners'] = content_ner
                    json_data['words'] = c_w
                    json_data['title_words'] = t_w
                    json_data['text'] = json_data['title'] + ' ' +json_data['content']
                    _raw_data.append(json_data)
                    count += 1
            else:
                break
    
    logger.info("fetch_data count: {}".format(count))
    
    return _ner_data, _word_data, _text_data, _title_word_data, _raw_data

def get_standard_time(time_stamp):
    try:
        timeArray = time.localtime(time_stamp/1000)
        otherStyleTime = time.strftime("%Y年%m月%d日", timeArray)
        return otherStyleTime
    except Exception as e:
        logger.info(e)
        logger.info("time_stamp error: "+ str(time_stamp))
    return ''



def special_entity_merge_segment(s, segs):
    
    signs_infos_list = get_special_chunk(s)
    words_info = {}
    for sign_info in signs_infos_list:
        offset_start = sign_info['offset'][0]
        words_info[offset_start] = {}
        words_info[offset_start]['chunk_str'] = sign_info['chunk_str']
        words_info[offset_start]['offset_end'] = sign_info['offset'][1]
        words_info[offset_start]['type'] = sign_info['type']

    segs_ = []
    new_word = None
    end_offset = None
    for seg in segs:
        
        offset_start = seg['offset']
        
        if offset_start == end_offset:
            new_word = None
            end_offset = None
            
        if offset_start in words_info and end_offset is None:
            new_word = words_info[offset_start]['chunk_str']
            seg['word'] = new_word
            seg['offset'] = offset_start
            seg['nature'] = 'n'
            segs_.append(seg)
            end_offset = words_info[offset_start]['offset_end']

        if new_word is None:
            segs_.append(seg)
            
    return segs_
    
# 1、对于更准确的分词的处理，（如 人名、动词的 处理）
# 2、关于核心动词合再做三元组，以及主语 COO 缺失、宾语 COO 缺失 的问题
# 3、完整界限的问题
# 4、同一句话中，连续位置的两个核心词进行合并，并将其主语和宾语进行合并

# 时间信息 common sense，优先取句中时间，再取段落时间 TMP，再取文章时间
# 在 triple 中携带 相应句子、标题、文章等信息
def data_event_process(ori_text_file, text_file, MODELDIR = None):
    
    if MODELDIR is None:
        MODELDIR = 'ltp_data_v3.4.0'
    hit_ltp =  HIT_LTP(MODELDIR)
    date_chunk_handle = Date_chunk_handle()
    # 过滤掉「偏公司行业经济类」资讯
    political_title_supervision = Political_Title_Supervision()
    # 寻找文章转述型概述
    event_description_supervison = Event_Description_Supervision()
    
    count_of_event_description = 0
    p_count = 0
    n_count = 0
    
    f1 = codecs.open(text_file, 'w', encoding='utf-8')
    with io.open(ori_text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                if 'title' in json_data and 'content' in json_data and 'dataSource' in json_data:
                    title = json_data['title'].strip()
                    
                    seg_title_split = []
                    seg_title = []
                    # 处理标题中的 ‘ ’ 句义隔断问题
                    title_s_list = title.strip().split(' ')
                    for title_s in title_s_list:
                        title_s_ = ''
                        for char in title_s:
                            char = char.strip()
                            if len(char) != 0:
                                title_s_ += char
                        title_s  = title_s_
                        if title_s != '':
                            #seg_title_s = split_sentence(title_s)
                            seg_title_s = hit_ltp.std_seg(title_s)
                            seg_title_s = special_entity_merge_segment(title_s, seg_title_s)
                            seg_title_split.append(seg_title_s)
                            seg_title += seg_title_s
                    
                    # 只选政策类数据，因为观测分析与程序运行时间问题
                    #p_c = political_title_supervision.f(title, title_seg = seg_title)
                    #logger.info('political judge: ' + title + ' ' + str(p_c))
                    p_c = True
                    if p_c:
                        p_count += 1
                        content = json_data['content']
                        dataSource = json_data['dataSource']
                        publishAt = json_data['publishAt']
                        # 找转述型概述
                        paragraphs = event_description_supervison.split_content_to_paragraph(content, dataSource)
                        event_description = event_description_supervison.get_event_description_from_article(paragraphs)
                        if event_description != '':
                            json_data['event_description'] = event_description
                            # 获取 RPT
                            RPT = ''
                            TMP = ''
                            try:
                                time_chunk_list = date_chunk_handle.get_date_chunk(event_description)
                                normalize_list = date_chunk_handle.normalize_datetime(time_chunk_list)
                                time_stamp_list = date_chunk_handle.parse_datetime(normalize_list, publishAt)
                            except Exception as e:
                                logger.info(e)
                                logger.info("event_description: " + event_description)
                                time_stamp_list = []
                                
                            for time_stamp in time_stamp_list:
                                if time_stamp['type'] == 'RPT':
                                    RPT = time_stamp['time_stamp'][0]
                                    break
                            for time_stamp in time_stamp_list:
                                if time_stamp['type'] == 'TMP':
                                    TMP = time_stamp['time_stamp'][0]
                                    break
                                
                            if RPT == '':
                                RPT = publishAt
                                
                            # title 处理
                            titles_info = []
                            for seg_title_s in seg_title_split:
                                words = [term['word'] for term in seg_title_s]
                                postags = [term['nature'] for term in seg_title_s]
                                data = hit_ltp.get_parser_triple(title, words = words, postags = postags)
                                temp_dic = {}
                                temp_dic['sentence'] = ''.join(words)
                                temp_dic['sentence_info'] = data
                                temp_dic['TMP'] = TMP
                                temp_dic['RPT'] = RPT
                                titles_info.append(temp_dic)
                                
                            json_data['titles_info'] = titles_info
                            # content 处理
                            # 用 semantic_clean 后的 event_description 覆盖掉 原来的 content，
                            content = semantic_clean(event_description)
                            json_data['seg_title'] = seg_title
                            #seg_content = split_sentence(content)
                            seg_content = hit_ltp.std_seg(content)
                            #seg_content = special_entity_merge_segment(content, seg_content)
                            json_data['seg_content'] = seg_content
                            json_data['content'] = content
                            content = content.replace('；','。').replace('，','。')
                            sentences = content.split("。")
                            sentences_info = []
                            for s in sentences:
                                # parser triple 的 s 中不能有 空格，否则导致会错误，因为分词和词性标注是分开的
                                # nlp 分词中，一般需要去掉句子中的空格，否则空格的词性可能会出错
                                s_ = ''
                                for char in s:
                                    char = char.strip()
                                    if len(char) != 0:
                                        s_ += char
                                s = s_
                                if len(s) > 0 and len(s) < 500:
                                    # 利用自己领域的分词 + ltp 的 parser
                                    #s_seg = split_sentence(s)
                                    s_seg = hit_ltp.std_seg(s)
                                    s_seg = special_entity_merge_segment(s, s_seg)
                                    words = [term['word'] for term in s_seg]
                                    postags = [term['nature'] for term in s_seg]
                                    data = hit_ltp.get_parser_triple(s, words = words, postags = postags)
                                    # TMP for Parser
                                    # --------------------------------------------
                                    TMP_ = ''
                                    try:
                                        time_chunk_list = date_chunk_handle.get_date_chunk(s)
                                        normalize_list = date_chunk_handle.normalize_datetime(time_chunk_list)
                                        time_stamp_list = date_chunk_handle.parse_datetime(normalize_list, publishAt)
                                    except Exception as e:
                                        logger.info(e)
                                        logger.info("s: " + s)
                                        time_chunk_list = []
                                        time_stamp_list = []
                                    if len(time_chunk_list) != 0:
                                        for time_stamp in time_stamp_list:
                                            if time_stamp['type'] == 'TMP':
                                                TMP_ = time_stamp['time_stamp'][0]
                                                break
                                    if TMP_ == '':
                                        TMP_ = TMP
                                        
                                    # --------------------------------------------
                                    temp_dic = {}
                                    temp_dic['sentence'] = s
                                    temp_dic['sentence_info'] = data
                                    temp_dic['TMP'] = TMP_
                                    temp_dic['RPT'] = RPT
                                            
                                    sentences_info.append(temp_dic)
                            
                            json_data['sentences_info'] = sentences_info
                            f1.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                            count_of_event_description += 1
                        else:
                            logger.info("no_event_description: {}".format(json_data['id'] + ' ' + title))
                    else:
                        n_count += 1
            else:
                break
    f1.close()
    hit_ltp.release()
    logger.info(" data_event_process count_of_event_description: {}".format(count_of_event_description))
    logger.info("data_event_process p_count: {}".format(p_count))
    logger.info("data_event_processa n_count: {}".format(n_count))
    return text_file

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
    # 每两个文章计算相似 score 再 均值
    _cluster_result_score = get_clusters_score(_cluster_result, all_ners, num_topics)
    # 获取每个簇的标题关键词
    _cluster_title_keywords = get_cluster_keywords_from_titles(_cluster_result, all_title_words)
    # 获取每个簇的textrank关键词
    _cluster_keywords = get_cluster_keywords(_cluster_result, all_keywords)

    new_origin_cluster_result = []
    # 过滤前
    #all_cluster_triples_info = {}
    # 过滤后
    #filter_all_cluster_triples_info = {}
    # 所有簇的事件 event_info 用于进一步聚类
    #all_cluster_event_infos = []
    
    for key in _cluster_result:
        #if key < len_of_center:
        #更新簇或者加入新簇
        if _cluster_result_score[key] < Threshold:
            continue

        keywords = []
        info_ids = []
        info_ids_to_data = []
        title = ''
        publish_time = 0
        all_titles_in_cluster = []
        all_titlewords_in_cluster = []
        all_publishtime_in_cluster = []
        max_size = 0
        min_words = 0
        
        for ele in _cluster_title_keywords[key]['keywords']:
            keywords.append(ele)
        
        # 簇内原始 triple
        #cluster_triples_info = {}
        # 簇内过滤后（句子核心动词过滤后）的 triple
        #filter_cluster_triples_info = {}
        # 用于进一步的事件聚类
        #cluster_event_infos = []
       
        for index in _cluster_result[key]:
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
                    
            all_titles_in_cluster.append(all_raw_data[index]['title'])
            all_titlewords_in_cluster.append(all_raw_data[index]['title_words'])
            all_publishtime_in_cluster.append(all_raw_data[index]['publishAt'])
            publish_time = max(publish_time, all_raw_data[index]['publishAt'])
            
            '''
            # 拿到外部处理
            # ---------- cluster_triples_info 处理 -----------
            titles_info = all_raw_data[index]['titles_info']
            sentences_info = all_raw_data[index]['sentences_info']
            
            for title_info in titles_info:
                
                sentence = title_info['sentence']
                TMP = title_info['TMP']
                RPT = title_info['RPT']
                sentence_info = title_info['sentence_info']
                
                core_words_info = sentence_info['core_words_info']
                core_words = [item['word'] for item in core_words_info]
                
                for a in sentence_info['triple_info']:
                    
                    triple = a['triple']
                    ner_info = a['ner_info']
                    
                    if str(triple) not in cluster_triples_info:
                        cluster_triples_info[str(triple)] = {'count': 1,
                                                              'TMP': {TMP: 1},
                                                              'RPT': {RPT: 1}}
                    else:
                        cluster_triples_info[str(triple)]['count'] += 1
                    
                    if str(triple) not in all_cluster_triples_info:
                        all_cluster_triples_info[str(triple)] = {'count': 1,
                                                              'TMP': {TMP: 1},
                                                              'RPT': {RPT: 1}}
                    else:
                        all_cluster_triples_info[str(triple)]['count'] += 1   
                    
                    # 根据 core_words 过滤 triple
                    if triple[1] in core_words:
                        
                        if str(triple) not in filter_cluster_triples_info:
                            filter_cluster_triples_info[str(triple)] = {'count': 1,
                                                                      'TMP': {TMP: 1},
                                                                      'RPT': {RPT: 1}}
                        else:
                            filter_cluster_triples_info[str(triple)]['count'] += 1
                        
                        if str(triple) not in filter_all_cluster_triples_info:
                            filter_all_cluster_triples_info[str(triple)] = {'count': 1,
                                                                          'TMP': {TMP: 1},
                                                                          'RPT': {RPT: 1}}
                        else:
                            filter_all_cluster_triples_info[str(triple)]['count'] += 1
                                                            
                        # 用于之后的进一步事件聚类
                        event_info = {}
                        event_info['triple_info'] = {'triple': triple,
                                                     'sentence': sentence}
                        event_info['ner_info'] = ner_info
                        event_info['TMP'] = TMP
                        event_info['RPT'] = RPT
                        cluster_event_infos.append(event_info)
                    
            for sentence_info in sentences_info:
                
                sentence = sentence_info['sentence']
                RPT = sentence_info['RPT']
                TMP = sentence_info['TMP']
                sentence_info = sentence_info['sentence_info']
                
                core_words_info = sentence_info['core_words_info']
                core_words = [item['word'] for item in core_words_info]
                
                for a in sentence_info['triple_info']:
                    triple = a['triple']
                    if str(triple) not in cluster_triples_info:
                        cluster_triples_info[str(triple)] = {'count': 1,
                                                              'TMP': {TMP: 1},
                                                              'RPT': {RPT: 1}
                                                              }
                    else:
                        cluster_triples_info[str(triple)]['count'] += 1
                        if TMP not in cluster_triples_info[str(triple)]['TMP']:
                            cluster_triples_info[str(triple)]['TMP'][TMP] = 1
                        else:
                            cluster_triples_info[str(triple)]['TMP'][TMP] += 1
                        
                        if RPT not in cluster_triples_info[str(triple)]['RPT']:
                            cluster_triples_info[str(triple)]['RPT'][RPT] = 1
                        else:
                            cluster_triples_info[str(triple)]['RPT'][RPT] += 1
                                                 
                    if str(triple) not in all_cluster_triples_info:
                        all_cluster_triples_info[str(triple)] = {'count': 1,
                                                                  'TMP': {TMP:1},
                                                                  'RPT': {RPT:1}
                                                                  }
                    else:
                        all_cluster_triples_info[triple]['count'] += 1
                        if TMP not in all_cluster_triples_info[str(a['triple'])]['TMP']:
                            all_cluster_triples_info[str(triple)]['TMP'][TMP] = 1
                        else:
                            all_cluster_triples_info[str(triple)]['TMP'][TMP] += 1
                        
                        if RPT not in all_cluster_triples_info[str(a['triple'])]['RPT']:
                            all_cluster_triples_info[str(triple)]['RPT'][RPT] = 1
                        else:
                            all_cluster_triples_info[str(triple)]['RPT'][RPT] += 1
                    
                    # 根据 core_words 过滤 triple
                    if triple[1] in core_words:
                        
                        if str(triple) not in filter_cluster_triples_info:
                            filter_cluster_triples_info[str(triple)] = {'count': 1,
                                                                      'TMP': {TMP: 1},
                                                                      'RPT': {RPT: 1}
                                                                      }
                        else:
                            filter_cluster_triples_info[str(triple)]['count'] += 1
                            if TMP not in filter_cluster_triples_info[str(triple)]['TMP']:
                                filter_cluster_triples_info[str(triple)]['TMP'][TMP] = 1
                            else:
                                filter_cluster_triples_info[str(triple)]['TMP'][TMP] += 1
                            
                            if RPT not in filter_cluster_triples_info[str(triple)]['RPT']:
                                filter_cluster_triples_info[str(triple)]['RPT'][RPT] = 1
                            else:
                                filter_cluster_triples_info[str(triple)]['RPT'][RPT] += 1
                                                     
                        if str(triple) not in filter_all_cluster_triples_info:
                            filter_all_cluster_triples_info[str(triple)] = {'count': 1,
                                                                      'TMP': {TMP:1},
                                                                      'RPT': {RPT:1}
                                                                      }
                        else:
                            filter_all_cluster_triples_info[str(a['triple'])]['count'] += 1
                            if TMP not in filter_all_cluster_triples_info[str(a['triple'])]['TMP']:
                                filter_all_cluster_triples_info[str(triple)]['TMP'][TMP] = 1
                            else:
                                filter_all_cluster_triples_info[str(triple)]['TMP'][TMP] += 1
                            
                            if RPT not in filter_all_cluster_triples_info[str(a['triple'])]['RPT']:
                                filter_all_cluster_triples_info[str(triple)]['RPT'][RPT] = 1
                            else:
                                filter_all_cluster_triples_info[str(triple)]['RPT'][RPT] += 1
                        
                        # 用于之后的进一步事件聚类                             
                        event_info = {}
                        event_info['triple_info'] = {'triple': triple,
                                                     'sentence': sentence}
                        event_info['TMP'] = TMP
                        event_info['RPT'] = RPT
                        cluster_event_infos.append(event_info)
        
        # 拿到外部处理
        # 补全与簇内事件再聚类
        # pass
        
        all_cluster_event_infos += cluster_event_infos
         
        # ---------- cluster_triples_info 排序 -----------
        cluster_triples_info_ = []
        cluster_triples_info = sorted(cluster_triples_info.items(), key=lambda d:d[1]['count'], reverse = True)
        for triple, dic in cluster_triples_info:
            TMP_dic = dic['TMP']
            TMP_dic = sorted(TMP_dic.items(), key=lambda d:d[1], reverse = True)
            RPT_dic = dic['RPT']
            RPT_dic = sorted(RPT_dic.items(), key=lambda d:d[1], reverse = True)
            
            
            cluster_triples_info_.append([triple, dic['count'], TMP_dic, RPT_dic])
        
        filter_cluster_triples_info_ = []
        filter_cluster_triples_info = sorted(filter_cluster_triples_info.items(), key=lambda d:d[1]['count'], reverse = True)
        for triple, dic in filter_cluster_triples_info:
            TMP_dic = dic['TMP']
            TMP_dic = sorted(TMP_dic.items(), key=lambda d:d[1]['count'], reverse = True)
            RPT_dic = dic['RPT']
            RPT_dic = sorted(RPT_dic.items(), key=lambda d:d[1]['count'], reverse = True)
            filter_cluster_triples_info_.append([triple, dic['count'], TMP_dic, RPT_dic])

        '''
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
        #cluster_dict['ori_triples_info'] = cluster_triples_info_
        #cluster_dict['filter_triples_info'] = filter_cluster_triples_info_
        #cluster_dict['cluster_event_infos'] = cluster_event_infos
                    
        new_origin_cluster_result.append(cluster_dict)

    # 按簇的大小排序
    s_sort = sorted(new_origin_cluster_result, key=lambda x: len(x['info_ids']), reverse = True)
    
    '''
    # ---------- all_cluster_triples_info 排序 -----------
    all_cluster_triples_info_ = []
    all_cluster_triples_info = sorted(all_cluster_triples_info.items(), key=lambda d:d[1]['count'], reverse = True)
    for triple, dic in all_cluster_triples_info:
        TMP_dic = dic['TMP']
        TMP_dic = sorted(TMP_dic.items(), key=lambda d:d[1]['count'], reverse = True)
        RPT_dic = dic['RPT']
        RPT_dic = sorted(RPT_dic.items(), key=lambda d:d[1]['count'], reverse = True)
        all_cluster_triples_info_.append([triple, dic['count'], TMP_dic, RPT_dic])
    
    filter_all_cluster_triples_info_ = []
    filter_all_cluster_triples_info = sorted(filter_all_cluster_triples_info.items(), key=lambda d:d[1]['count'], reverse = True)
    for triple, dic in filter_all_cluster_triples_info:
        TMP_dic = dic['TMP']
        TMP_dic = sorted(TMP_dic.items(), key=lambda d:d[1]['count'], reverse = True)
        RPT_dic = dic['RPT']
        RPT_dic = sorted(RPT_dic.items(), key=lambda d:d[1]['count'], reverse = True)
        filter_all_cluster_triples_info_.append([triple, dic['count'], TMP_dic, RPT_dic])
    
    '''
    
    return s_sort

if __name__ == '__main__':
    
    '''
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
    '''
    
    
    
    s = '(中共中央办公厅)近日[印发]《关于解决形式（主义突出问题）》为基层减负的通》'
    segs = split_sentence(s)
    print(special_entity_merge_segment(s, segs))
    
    #print(get_special_chunk(s))
