import io
import  json

def load_line_data(filepath):
    event_data = []
    i = 0

    with io.open(filepath,'r',encoding='utf-8') as f:
        while True:
            i += 1
            line = f.readline()
            if len(line) > 0:
                event_data.append(line)
            if i > 1000:
                break
    return event_data


def get_general_verbs(filepath1):
    verbs = []

    with io.open(filepath1,'r',encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                line_list = line.split()
                if len(line_list) > 0:
                    verbs.append(line_list[0])

            else:
                break
    return verbs


def get_parser_events(verbs,filepath):
    event_data = load_line_data(filepath)
    parser_list = []

    for item in event_data:
        if len(item) > 0:
            dic_item ={}
            json_result = json.loads(item)
            dic_item['sentences_parser_result'] = json_result['sentences_parser_result']
            dic_item['title_parser_result'] = json_result['title_parser_result']
            parser_list.append(dic_item)

    for dic in parser_list:
        sentence_start = 0

        if len(dic) > 0:
            event_triples_infos = dic['title_parser_result']['event_triples_info']
            for event_triples_info in event_triples_infos:
                triple = event_triples_info['triple']



            sentences_parser_results = dic['sentences_parser_result']
            for sentences_parser_result in sentences_parser_results:
                index_item = []
                triple_list = []
                arcs = sentences_parser_result['sentence_srl_result']['arcs']
                print(sentences_parser_result['sentence_srl_result']['sentence'])
                for arc in arcs:
                    arc_list = arc.split(':')
                    if arc_list[1] == 'HED':
                        sentence_start = arcs.index(arc)
                postags = sentences_parser_result['sentence_srl_result']['postags']
                words = sentences_parser_result['sentence_srl_result']['words']
                print(words[sentence_start])
                for i in range(0,len(postags)):
                    if postags[i] == 'v' and i <= sentence_start:
                        index_item.append(i)
                        # print(index_item)
                event_triples_infos = sentences_parser_result['sentence_srl_result']['event_triples_info']
                for event_triples_info in event_triples_infos:

                    triple = event_triples_info['triple']
                    # print(words.index(triple[1]))
                    # print(index_item)
                    # print("******************")
                    # if triple[1] in words:
                    if words.index(triple[1]) in index_item:
                        if triple[0] != '' or triple[2] != '':
                            if triple[1] in verbs:
                                triple_list.append(triple)
                print(triple_list)
            print('***************************************')

if __name__ == '__main__':
    verbs = get_general_verbs('./ltp_data/event_verbs.txt')
    get_parser_events(verbs,'./ltp_data/2019-04-17_events_ltp_parser.txt')


