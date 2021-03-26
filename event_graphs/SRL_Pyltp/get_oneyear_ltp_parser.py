# -*- coding: utf-8 -*-

import io
import json
from event_generator import Event_Ltp_Generator

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

if __name__ == '__main__':
    
    # load ltp =============================================
    event_ltp_generator = Event_Ltp_Generator()
    # ======================================================
     
    read_file = './data_oneyear_crawl/2019-04-17_events.txt'
    write_file = './data_oneyear_crawl/2019-04-17_events_ltp_parser.txt'
    
    #articles = load_json_line_data(read_file)
    #print('len(articles): ', len(articles))
    
    killed_count = 0

    count = 0

    print('read file ... ')
    with io.open(read_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                count += 1
                if count <= killed_count:
                    continue
                try:
                    article = json.loads(line.strip())
                    temp_dic = {}
                    title = article['title']
                    title_srl_result = event_ltp_generator.fact_triple_extract(title)
                    temp_dic['title'] = title
                    temp_dic['title_parser_result'] = title_srl_result
                    temp_dic['url'] = article['url']
                    temp_dic['publishAt'] = article['publishAt']
                    
                    sentences_parser_result = []
                    p = article['event_discription']
                    for s in p.split('。'):
                        for s1 in s.split('；'):
                            
                            if len(s1) > 500:
                                continue
                            
                            if len(s1.strip()) > 0:
                                s1_parser_result = event_ltp_generator.fact_triple_extract(s1)
                                
                                sentences_parser_result.append({'sentence_srl_result': s1_parser_result})
                    
                    temp_dic['sentences_parser_result'] = sentences_parser_result
                    
                    with io.open(write_file, 'a', encoding='utf-8') as f1:
                        f1.write(json.dumps(temp_dic, ensure_ascii=False) + "\n")

                    if count % 1000 == 1:
                        print('count: ', count)
                    print('count: ', count)
                except Exception as e:
                    print('ltp error')
                    print("Exception: {}".format(e))
            else:
                break
    # -------------------------------------------------------
   
    

    
    

















