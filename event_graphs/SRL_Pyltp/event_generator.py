# -*- coding: utf-8 -*-

import os
import io
import json
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer

class Event_Ltp_Generator():

    def __init__(self, MODELDIR = 'ltp_data_v3.4.0'):
        
        self.segmentor = None
        self.postagger = None
        self.parser = None
        self.recognizer = None
        
        print("正在加载LTP模型... ...")
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(MODELDIR, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(MODELDIR, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(MODELDIR, "parser.model"))
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(MODELDIR, "ner.model"))
        
    def fact_triple_extract(self, sentence):
        """
        对于给定的句子进行事实三元组抽取
        Args:
            sentence: 要处理的语句
        """
        #print(sentence)
        words = self.segmentor.segment(sentence)
        #print("\t".join(words))
        words = '\t'.join(words)        #分词
        words = words.split('\t')
        postags = self.postagger.postag(words)      #词性标注
        postags = '\t'.join(postags)
        postags = postags.split('\t')
        netags = self.recognizer.recognize(words, postags)      #命名实体识别 S-Ni，O，BIESO
        #print("\t".join(netags))
        netags = '\t'.join(netags)
        netags = netags.split('\t')
        arcs = self.parser.parse(words, postags)                #依存句法分析，SBV
        #print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        arcs_ = "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        arcs_ = arcs_.split('\t')
        child_dict_list = self.build_parse_child_dict(words, postags, arcs)
        #print(child_dict_list)
        
        sentence_result = {}
        sentence_result['sentence'] = sentence
        sentence_result['words'] = words
        sentence_result['postags'] = postags
        sentence_result['netags'] = netags
        sentence_result['arcs'] = arcs_
        sentence_result['child_dict_list'] = child_dict_list
                       
        event_triples_info = []
        entity_triples = []
        for index in range(len(postags)):
            # 抽取以谓词为中心的事实三元组
            if postags[index] == 'v':
                child_dict = child_dict_list[index]
                
                # 主谓宾
                if 'SBV' in child_dict and 'VOB' in child_dict:
                    e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    r = words[index]
                    e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    event_triples_info.append({'triple':(e1, r, e2),
                                               'structure': '主谓宾'})
                    
                # 定语后置，动宾关系
                # 进行v 正式 访问vob 的 缅甸国务资政昂山素季sbv
                # 动宾，补主语
                elif arcs[index].relation == 'ATT':
                    if 'VOB' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, arcs[index].head - 1)
                        r = words[index]
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        temp_string = r+e2
                        if temp_string == e1[:len(temp_string)]:
                            e1 = e1[len(temp_string):]
                        if temp_string not in e1:
                            event_triples_info.append({'triple':(e1, r, e2),
                                               'structure': '补主'})
    
                # 含有介宾关系的主谓动补关系
                # 哈立德sbv 居住 在cmp(动补结构) 土耳其pob
                # 主谓，补宾语
                elif 'SBV' in child_dict and 'CMP' in child_dict:
                    #e1 = words[child_dict['SBV'][0]]
                    e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    cmp_index = child_dict['CMP'][0]
                    r = words[index] + words[cmp_index]
                    if 'POB' in child_dict_list[cmp_index]:
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                        event_triples_info.append({'triple':(e1, r, e2),
                                               'structure': '补宾'})
                # 主谓
                elif 'SBV' in child_dict:
                    e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    r = words[index]
                    event_triples_info.append({'triple':(e1, r, ''),
                                               'structure': '主谓'})
                # 谓宾
                elif 'VOB' in child_dict:
                    r = words[index]
                    e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    event_triples_info.append({'triple':('', r, e2),
                                               'structure': '谓宾'})
            # 尝试抽取命名实体有关的三元组
            if netags[index][0] == 'S' or netags[index][0] == 'B':
                ni = index
                if netags[ni][0] == 'B':
                    while netags[ni][0] != 'E':
                        ni += 1
                    e1 = ''.join(words[index:ni+1])
                else:
                    e1 = words[ni]
                if arcs[ni].relation == 'ATT' and postags[arcs[ni].head-1] == 'n' and netags[arcs[ni].head-1] == 'O':
                    r = self.complete_e(words, postags, child_dict_list, arcs[ni].head-1)
                    if e1 in r:
                        r = r[(r.index(e1)+len(e1)):]
                    if arcs[arcs[ni].head-1].relation == 'ATT' and netags[arcs[arcs[ni].head-1].head-1] != 'O':
                        e2 = self.complete_e(words, postags, child_dict_list, arcs[arcs[ni].head-1].head-1)
                        mi = arcs[arcs[ni].head-1].head-1
                        li = mi
                        if netags[mi][0] == 'B':
                            while netags[mi][0] != 'E':
                                mi += 1
                            e = ''.join(words[li+1:mi+1])
                            e2 += e
                        if r in e2:
                            e2 = e2[(e2.index(r)+len(r)):]
                        if r+e2 in sentence:
                            entity_triples.append(('实体关系: ', e1, r, e2))
                   
        # ner(人名、地名、机构名)
        entities = {'人物':[],
                    '地点':[],
                    '机构':[]}
        for index in range(len(netags)):
            if netags[index] == 'S-Nh':
                if words[index] not in entities['人物']:
                    entities['人物'].append(words[index])
            if netags[index] == 'S-Ni':
                if words[index] not in entities['机构']:
                    entities['机构'].append(words[index])
            if netags[index] == 'S-Ns':
                if words[index] not in entities['地点']:
                    entities['地点'].append(words[index])
                
        sentence_result['event_triples_info'] = event_triples_info
        sentence_result['entity_triples'] = entity_triples
        sentence_result['entities'] = entities
                
        
        return sentence_result

    def collect_triple(self,sentence_result):
        triple_list = []
        for event_triple in sentence_result['event_triples_info']:
            triple_item = event_triple['structure'] + "\t" + str((event_triple['triple'][0],event_triple['triple'][1],event_triple['triple'][2]))
            triple_list.append(triple_item)
        return triple_list


    def build_parse_child_dict(self, words, postags, arcs):
        """
        为句子中的每个词语维护一个保存句法依存儿子节点的字典
        为每一个词创建一个字典，字典中包含所有依赖该词的其他词的关系和索引
        Args:
            words: 分词列表
            postags: 词性列表
            arcs: 句法依存列表
        """
        child_dict_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index + 1:
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            #if child_dict.has_key('SBV'):
            #    print words[index],child_dict['SBV']
            child_dict_list.append(child_dict)
        return child_dict_list
    
    
    # 1、ATT定中关系，2、动宾短语实体，3、从父节点向子节点遍历
    def complete_e(self, words, postags, child_dict_list, word_index):
        """
        完善识别的部分实体
        """
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
        
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix
    
        return prefix + words[word_index] + postfix
        


if __name__ == "__main__":
    
    event_ltp_generator = Event_Ltp_Generator()
    
    read_file = "./long_text_2020-09-07-14-24-09"
    out_file_name = "./triple_result.txt"
    
    # with io.open(read_file, "r", encoding='utf-8') as f:
    #     while True:
    #         line = f.readline()
    #         if len(line) > 0:
    #             sentence_result = event_ltp_generator.fact_triple_extract(line)
    #             with io.open(out_file_name, 'a', encoding='utf-8') as f1:
    #                 f1.write(json.dumps(sentence_result, ensure_ascii=False) + "\n")
    #
    #         else:
    #             break
    with io.open(read_file, "r", encoding='utf-8') as f:
        file_list = json.load(f)
        for line in file_list:
            if len(line) > 0:
                sentence_result = event_ltp_generator.fact_triple_extract(line)
                triple_list = event_ltp_generator.collect_triple(sentence_result)
                with io.open(out_file_name, 'a', encoding='utf-8') as f1:
                    f1.write(json.dumps(triple_list, ensure_ascii=False) + "\n")

            else:
                break
    
    
    
    
    
    
    
    
