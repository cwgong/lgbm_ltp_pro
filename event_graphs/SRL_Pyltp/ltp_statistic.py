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
    x = list(range(0, size))
    if size > 50:
        random.shuffle(x)
        ids = x[0:50]
    else:
        ids = x

    return ids


if __name__ == '__main__':

    filepath = './data_oneyear_crawl/2019-04-17_events_ltp.txt'
    filepath1 = './data_oneyear_crawl/2019-04-17_political_events_ltp.txt'
    filepath2 = './ltp_data/2019-04-17_events_ltp_parser.txt'
    filepath3 = './ltp_data/action_words_p.txt'

    action_words_dic = {}
    c = 0

    with io.open(filepath2,'r',encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_line = json.loads(line.strip())
                c += 1
                if c % 1000 == 1:
                    print(c)

                title = json_line['title']
                title_parser_result = json_line['title_parser_result']
                title_parser_info = title_parser_result['event_triples_info']

                for a in title_parser_info:

                    # triple_str = str(a['triple']) + ' ' + 'TMP: ' + a['TMP'] + ' ' + 'LOC: ' + a['LOC'] + '\t|\t'
                    triple_str = str(a['triple']) + ' ' + '\t|\t'
                    act_w = a['triple'][1]

                    if act_w not in action_words_dic.keys():
                        action_words_dic[act_w] = {}
                        action_words_dic[act_w]['count'] = 1
                        action_words_dic[act_w]['triples'] = [triple_str]
                    else:
                        action_words_dic[act_w]['count'] += 1
                        if triple_str not in action_words_dic[act_w]['triples']:
                            action_words_dic[act_w]['triples'].append(triple_str)

                sentences_parser_result = json_line['sentences_parser_result']
                for item in sentences_parser_result:
                    sentence_srl_result = item['sentence_srl_result']
                    s = sentence_srl_result['sentence']
                    s_words = sentence_srl_result['words']
                    postags = sentence_srl_result['postags']
                    s_srl_info = sentence_srl_result['event_triples_info']

                    for b in s_srl_info:
                        # triple_str = str(b['triple']) + ' ' + 'TMP: ' + b['TMP'] + ' ' + 'LOC: ' + b['LOC'] + '\t|\t'
                        triple_str = str(b['triple']) + ' ' + '\t|\t'
                        act_w = b['triple'][1]
                        if act_w not in action_words_dic.keys():
                            action_words_dic[act_w] = {}
                            action_words_dic[act_w]['count'] = 1
                            action_words_dic[act_w]['triples'] = [triple_str]
                        else:
                            action_words_dic[act_w]['count'] += 1
                            if triple_str not in action_words_dic[act_w]['triples']:
                                action_words_dic[act_w]['triples'].append(triple_str)

                if c > 100000:
                    break
            else:
                break

    a_w_t = sorted(action_words_dic.items(), key=lambda d:d[1]['count'], reverse = True)

    with io.open(filepath3, 'a', encoding='utf-8') as f:
        for word, dic in a_w_t:
            ids = random_choose(dic['triples'])
            string = ''
            for i in ids:
                string += dic['triples'][i]
            f.write(word + '\t' + str(dic['count']) +  '\t' + string + '\n')

    # for word, dic in a_w_t:
    #     ids = random_choose(dic['triples'])
    #     string = ''
    #     for i in ids:
    #         string += dic['triples'][i]
    #     print(word + '\t' + str(dic['count']) +  '\t' + string + '\n')



