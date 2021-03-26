import json
#读取srl实验结果中的事件
f = open('event_triples_for_day/2019-04-17_events_ltp.txt', 'r', encoding='utf-8')
contents=f.readlines()[0:100]
title_parser = []
sentence_parser = []
#dic_contents=[]
filter_content={}
def get_triples_from_srl(lines):
    contents = f.readlines()[0:lines]
    for list in contents:
        dic = json.loads(list)
        title_parser.append(dic['title_srl_result'])
        sentence_parser.append(dic['sentences_srl_result'])
    filter_content['title_triples_info'] = title_parser
    filter_content['sentences_triples_info'] = sentence_parser
    all_triples_list=[]
    for i in range(len(filter_content['sentences_triples_info'])):
        all_triples={}
        title_triples = filter_content['title_triples_info'][i]['srl_info']
        sentences_triples = filter_content['sentences_triples_info'][i][0]['sentence_srl_result']['srl_info']
        all_triples['title']=title_triples
        all_triples['content']=sentences_triples
        all_triples_list.append(all_triples)
        print(i)
    return all_triples_list
if __name__ == '__main__':
    get_triples_from_srl(1000)