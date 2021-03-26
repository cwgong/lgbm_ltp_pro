from get_events_from_parser import get_triples_from_parser
from get_events_from_srl import get_triples_from_srl

f=open('WordsDic/irrelevant_verb.txt','r',encoding='utf-8')
verbs=f.readlines()
irrelevant_verbs=[]
for verb in verbs:
    verb=verb.strip('\n')
    irrelevant_verbs.append(verb)

def filter_srl():
    all_triples_srl=get_triples_from_srl(1000)
    #print(all_triples)
    for triple in all_triples_srl:
        #print(triple)
        content_triples=triple['content']
        #title_triples=triple['title']
        for content_triple in content_triples:
            if content_triple['triple'][0]=='' and content_triple['triple'][2]=='':#删除主语和宾语均缺失的情况
                continue
            if content_triple['triple'][1] in irrelevant_verbs:#删除非实意动词所在的事件三元组
                continue
            for i in range(3):
                print(content_triple['triple'][i]+"  ",end='')
            print()
        print()

def filter_parser_title():
    all_triples_parser=get_triples_from_parser(1000)
    for triple in all_triples_parser:
        #print(triple)
        title_triples=triple['title']
        title_arcs=triple['title_parser']['arcs']
        hed_child_dict={}
        start = 0
        end = 0
        for i in range(len(title_arcs)):
            arc=title_arcs[i]
            arc=arc.split(':')
            if arc[1]=='HED':
                hed_child_dict=triple['title_parser']['child_dict_list'][i]
                start = i
                print(hed_child_dict)
                break
        if 'SBV' in hed_child_dict and 'VOB' in hed_child_dict:
            #start= int(hed_child_dict['SBV'][0])
            end=int(hed_child_dict['VOB'][0])
        print(start)
        print(end)
        hed_event_verbs=[]
        for i in range(start+1,end):
            if triple['title_parser']['postags'][i]=='v':
                hed_event_verbs.append(triple['title_parser']['words'][i])
        print(hed_event_verbs)
        #title_triples=triple['title']
        for title_triple in title_triples:
            if title_triple['triple'][0]=='' and title_triple['triple'][2]=='':#删除主语和宾语均缺失的情况
                continue
            if title_triple['triple'][1] in irrelevant_verbs:#删除非实意动词所在的事件三元组
                continue
            if title_triple['triple'][1] in hed_event_verbs:#删除在核心动词从句中的事件
                continue
            for i in range(3):
                print(title_triple['triple'][i]+"  ",end='')
            print('structure:'+title_triple['structure'])
            # print()
        print()

def filter_parser_content():
    all_triples_parser=get_triples_from_parser(1000)#读取前1000个事件
    for triple in all_triples_parser:
        #print(triple)
        content_triples=triple['content']
        content_arcs=triple['content_parser']['arcs']
        hed_child_dict={}
        start = 0
        end = 0
        for i in range(len(content_arcs)):
            arc=content_arcs[i]
            arc=arc.split(':')
            if arc[1]=='HED':
                hed_child_dict=triple['content_parser']['child_dict_list'][i]
                start = i
                print(hed_child_dict)
                break
        if 'SBV' in hed_child_dict and 'VOB' in hed_child_dict:
            #start= int(hed_child_dict['SBV'][0])
            end=int(hed_child_dict['VOB'][0])
        print(start)
        print(end)
        hed_event_verbs=[]
        for i in range(start+1,end):
            if triple['content_parser']['postags'][i]=='v':
                hed_event_verbs.append(triple['content_parser']['words'][i])
        print(hed_event_verbs)
        #title_triples=triple['title']
        for content_triple in content_triples:
            if content_triple['triple'][0]=='' and content_triple['triple'][2]=='':#删除主语和宾语均缺失的情况
                continue
            if content_triple['triple'][1] in irrelevant_verbs:#删除非实意动词所在的事件三元组
                continue
            if content_triple['triple'][1] in hed_event_verbs:#删除在核心动词从句中的事件
                continue
            for i in range(3):
                print(content_triple['triple'][i]+"  ",end='')
            print('structure:'+content_triple['structure'])
            # print()
        print()

if __name__ == '__main__':
    filter_parser_content()
