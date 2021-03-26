# -*- coding: utf-8 -*-

import io
import json
import random

def load_json_line_data(filepath):
    
    data = []
    with io.open(filepath, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                data.append(json.loads(line.strip()))  
            else:
                break
    return data

def random_choose(triples):
    
    ids = []
    size = len(triples)
    x = list(range(0,size))
    if size > 50:
        random.shuffle(x) 
        ids = x[0:50]
    else:
        ids = x
        
    return ids

if __name__ == '__main__':
    
    filepath = './data_oneyear_crawl/2019-04-17_events_ltp.txt'
    filepath1 = './data_oneyear_crawl/2019-04-17_political_events_ltp.txt'
    #data = load_json_line_data(filepath)
    #print('len(data):', len(data)) # 148046
    
    # 找“动作”词，通过该词定义事件类别（事件触发词）
    # 先过滤掉明显错误的 “动作” 词，或找到高频 “动作” 词
    action_words_dic = {}
    c = 0
    with io.open(filepath1, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_line = json.loads(line.strip())
                c += 1
                if c % 1000 == 1:
                    print(c)
                #print(json_line)
                # title
                title = json_line['title']
                title_srl_result = json_line['title_srl_result']
                title_srl_info = title_srl_result['srl_info']
                for a in title_srl_info:
                    #print('triple: ', a['triple'])
                    #print('TMP: ', a['TMP'])
                    #print('LOC: ', a['LOC'])
                    #print('-----------')
                    triple_str = str(a['triple']) + ' ' + 'TMP: ' + a['TMP'] + ' ' + 'LOC: ' + a['LOC'] + '\t|\t'
                    act_w = a['triple'][1]
                    if act_w not in action_words_dic.keys():
                        action_words_dic[act_w] = {}
                        action_words_dic[act_w]['count'] = 1
                        action_words_dic[act_w]['triples'] = [triple_str]
                    else:
                        action_words_dic[act_w]['count'] += 1
                        if triple_str not in action_words_dic[act_w]['triples']:
                            action_words_dic[act_w]['triples'].append(triple_str)
                # all sentences
                sentences_srl_result = json_line['sentences_srl_result']
                # sentence
                for s_s_r in sentences_srl_result:
                    s = s_s_r['sentence']
                    #print(s)
                    # sentence_srl_result
                    s_srl_result = s_s_r['sentence_srl_result']
                    s_words = s_srl_result['words']
                    postags = s_srl_result['postags']
                    s_srl_info = s_srl_result['srl_info']
                    for b in s_srl_info:
                        #print('triple: ', b['triple'])
                        #print('TMP: ', b['TMP'])
                        #print('LOC: ', b['LOC'])
                        #print('-----------')
                        triple_str = str(b['triple']) + ' ' + 'TMP: ' + b['TMP'] + ' ' + 'LOC: ' + b['LOC'] + '\t|\t'
                        act_w = b['triple'][1]
                        if act_w not in action_words_dic.keys():
                            action_words_dic[act_w] = {}
                            action_words_dic[act_w]['count'] = 1
                            action_words_dic[act_w]['triples'] = [triple_str]
                        else:
                            action_words_dic[act_w]['count'] += 1
                            if triple_str not in action_words_dic[act_w]['triples']:
                                action_words_dic[act_w]['triples'].append(triple_str)
                            
                #break
            else:
                break
            
    a_w_t = sorted(action_words_dic.items(), key=lambda d:d[1]['count'], reverse = True)
    
    with io.open('./data_oneyear_crawl/action_words_p.txt', 'a', encoding='utf-8') as f:
        for word, dic in a_w_t:
            ids = random_choose(dic['triples'])
            string = ''
            for i in ids:
                string += dic['triples'][i]
            f.write(word + '\t' + str(dic['count']) +  '\t' + string + '\n')
        
        
        
        
        
            
            