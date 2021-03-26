# -*- encoding:utf8 -*-
import os
from pyltp import Segmentor, Postagger, Parser, SementicRoleLabeller
import json
import requests


# 分词
def split_sentence(sen):
    nlp_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough'
    try:
        cut_sen = dict()
        cut_sen['content'] = sen
        data = json.dumps(cut_sen).encode("UTF-8")
        cut_response = requests.post(nlp_url, data=data, headers={'Connection':'close'})
        cut_response_json = cut_response.json()
        return cut_response_json['data']
    except Exception as e:
        print.exception("Exception: {}".format(e))
        return []
    
def get_event_sentences():#读取文件中的sentence
    sentences_lists = []
    # f = open('event_sentences_for_day/%d.txt' % day, 'r', encoding='utf-8')
    f = open('event_sentences_for_day/conference.txt', 'r', encoding='utf-8')
    # for line in f.readlines():
    #     line_dict = eval(line)
    #     sentences_lists.append(list(line_dict.values())[0])

    for line in f.readlines():
        # line = eval(line)
        sentences_lists.append(line)
    return sentences_lists


def get_event_triples_srl(sentence):
    words = segmentor.segment(sentence)  # 分词
    # print(words)
    words = '\t'.join(words)
    words = words.split('\t')
    postags = postagger.postag(words)  # 词性标注
    postags = '\t'.join(postags)
    postags = postags.split('\t')
    print(words)
    print(postags)
    arcs = parser.parse(words, postags)  # 句法分析
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    triplesAll = []
    for role in roles:
        triples = ['', '', '']
        role = role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
        print(role)
        predicate = words[role[0]]
        print(predicate)
        triples[1] = predicate
        args = role[1].split(")")
        args.remove('')
        for ele in args:
            ele = ele.split(":")
            if ele[0] == "A0":
                index = ele[1][1:].split(",")
                A0 = words[int(index[0]):int(index[1]) + 1]
                A0_str = "".join(A0)
                triples[0] = A0_str
            if ele[0] == "A1":
                index = ele[1][1:].split(",")
                A1 = words[int(index[0]):int(index[1]) + 1]
                A1_str = "".join(A1)
                triples[2] = A1_str
        print(triples)
        triplesAll.append(triples)
    return triplesAll


def get_event_triples_dp():
    words = segmentor.segment(sentence)  # 分词
    words = '\t'.join(words)
    words = words.split('\t')
    print('ltp: ',words)
    postags = postagger.postag(words)  # 词性标注
    postags = '\t'.join(postags)
    postags = postags.split('\t')
    arcs = parser.parse(words, postags)  # 句法分析
    arcs = '\t'.join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
    arcs = arcs.split('\t')
    print(arcs)
    # return event_triple


if __name__ == '__main__':
    
    
    num_of_days = 1
    for i in range(num_of_days):
        day = i + 4

        LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
        segmentor = Segmentor()
        #segmentor.load(cws_model_path)  # 加载模型，第二个参数是外部词典文件路径
        segmentor.load_with_lexicon(cws_model_path, './dict_for_ltp/ltp_customer.txt') 
        postagger = Postagger()
        #postagger.load(pos_model_path)
        postagger.load_with_lexicon(pos_model_path, './dict_for_ltp/ltp_customer.txt')
        parser = Parser()
        parser.load(par_model_path)
        labeller = SementicRoleLabeller()
        labeller.load(srl_model_path)

        # ------------------------------------------------------------------------------
        # pyltp-srl
        # sentences_lists = get_event_sentences()
        # print(len(sentences_lists))
        sentences_lists =['李克强将出席博鳌亚洲论坛2019年年会','新华社北京3月22日电3月21日14时48分许，江苏盐城市响水县陈家港镇天嘉宜化工有限公司化学储罐发生爆炸事故',
        '正赴国外访问途中的中共中央总书记、国家主席、中央军委主席习近平立即作出重要指示',
        '国务院反假货币工作联席会议第六次会议在北京召开',
        '换言之，最迟于3月22日，科创板的第一批IPO企业名单将正式出炉',
        '中国政府网22日公布《国务院办公厅关于调整2019年劳动节假期安排的通知》',
        '同日，财政部、国家税务总局、海关总署联合发布《深化增值税改革有关政策的公告》，对增值税相关抵扣政策等细节问题一一予以明确',
        '十二届全国人大常委会第二十二次会议29日上午举行第一次全体会议',
        '财政部、税务总局、海关总署21日联合发布《关于深化增值税改革有关政策的公告》（以下简称“公告”），推进增值税实质性减税',
        '2019年3月21日，全国有线电视网络融合发展战略合作签约活动在中国国际展览中心举行']
        #读取sentence
        #fout = open('event_triples_for_day/%d_srl.txt' % day, 'w', encoding='utf-8')
        # for i in range(len(sentences_lists)):
        #     sentences = sentences_lists[i]
        #     print(sentences)
        #     # keyPredicates = keyPredicates_lists[i]
        #     # keyEntities = keyEntities_lists[i]
        #     for sentence in sentences:
        #         if len(sentence) > 500:
        #             continue
        #         event_triple = get_event_triples_srl()
        #         for ele in event_triple:
        #             fout.write(str(ele))
        #             fout.write('\n')
        #     fout.write('\n')
        # fout.close()
        for i in range(len(sentences_lists)):
            sentence = sentences_lists[i]
            x = split_sentence(sentence)
            print([x['word'] for x in x])
            if len(sentence) > 500:
                continue
            event_triple = get_event_triples_srl(sentence)
            
        # ------------------------------------------------------------------------------

        segmentor.release()
        postagger.release()
        parser.release()
        labeller.release()
