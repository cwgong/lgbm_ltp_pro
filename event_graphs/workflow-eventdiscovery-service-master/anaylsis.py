# -*- coding: utf-8 -*-

import io
import json
from event_project_utils import semantic_clean, get_special_chunk, event_merge, cluster_event_complete

'''
with io.open('./logs/d3.txt', "a", encoding='utf-8') as f1:
    with io.open('./logs/a3.txt', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                
                cluster_title = json_data['title']
                cluster_keywords = json_data['keywords']
                hot = len(json_data['info_ids'])
                f1.write('cluster_title:' + cluster_title + '\n')
                f1.write('hot:' + str(hot) + '\n')
                f1.write('cluster_keywords:' + str(cluster_keywords) + '\n')
                
                f1.write('------------' + '\n')
                f1.write('cluster_info:' + '\n')
                for raw_data in json_data['info_ids_to_data']:
                    title = raw_data['title']
                    content = raw_data['content']
                    f1.write('********' + '\n')
                    f1.write('title:' + title + '\n')
                    f1.write('content:' + content + '\n')
                    
                f1.write('------------' + '\n')
                min_publishtime = json_data['min_publishtime']
                max_publishtime = json_data['max_publishtime']
                f1.write('min_publishtime:' + str(min_publishtime) + '\n')
                f1.write('max_publishtime:' + str(max_publishtime) + '\n')
                ori_cluster_triples_info = json_data['ori_triples_info']
                f1.write('------------' + '\n')
                f1.write('\noricluster_triples_info:' + '\n\n')
                for triple_info in ori_cluster_triples_info:
                    f1.write(str(triple_info) + '\n')
                
                filter_cluster_triples_info = json_data['filter_triples_info']
                f1.write('------------' + '\n')
                f1.write('\nfilter_cluster_triples_info:' + '\n\n')
                for triple_info in filter_cluster_triples_info:
                    f1.write(str(triple_info) + '\n')
                    
                cluster_event_infos = json_data['cluster_event_infos']
                
                # 补全与簇内事件再聚类
                # 补全
                cluster_event_infos_complete = cluster_event_complete(cluster_event_infos)
                f1.write('------------' + '\n')
                f1.write('\ncluster_event_infos_complete:' + '\n\n')
                for event_info in cluster_event_infos_complete:
                    f1.write(str(event_info) + '\n')
                    with io.open('./logs/c22.txt', "a", encoding='utf-8') as f2:
                        f2.write(json.dumps(event_info, ensure_ascii=False) + "\n")
                # 聚类
                cluster_event_clusters = event_merge(cluster_event_infos_complete)
                
                # 聚类后按簇大小排序
                cluster_event_clusters = sorted(cluster_event_clusters, key=lambda x:len(x), reverse = True)
                       
                f1.write('------------' + '\n')
                f1.write('\ncluster_event_clusters:' + '\n\n')
                for cluster_event_cluster in cluster_event_clusters:
                    f1.write('-----' + '\n')
                    
                    # cluster_event_cluster 按 triple 统计排序 方便观察
                    triples_count = {}
                    
                    for item in cluster_event_cluster:
                        m = item['triple_info'] ['triple']
                        if str(m) not in triples_count:
                            triples_count[str(m)] = 1
                        else:
                            triples_count[str(m)] += 1
                                          
                    triples_count = sorted(triples_count.items(), key=lambda x:x[1], reverse = True)
                    
                    for item, count in triples_count:
                        f1.write(str(item) + 'count: ' + str(count) + '\n')
                
                
                
                    
                f1.write('\n================================\n\n\n')
            
            else:
                break
            
    
with io.open('./logs/e1.txt', "a", encoding='utf-8') as f1:
    with io.open('./logs/b1.txt', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line) # [triple, dic['count'], TMP_dic, RPT_dic]
                
                
                triple = json_data[0]
                count = json_data[1]
                TMP = json_data[2]
                RPT = json_data[3]
                
                
                f1.write('triple: ' + triple + '\n')
                f1.write('count: ' + str(count) + '\n')
                f1.write('TMP:' + str(TMP) + '\n')
                f1.write('RPT:' + str(RPT) + '\n')
                f1.write('\n================================\n\n')
            
            else:
                break


with io.open('./logs/f1.txt', "a", encoding='utf-8') as f1:
    with io.open('./logs/c1.txt', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line) # [triple, dic['count'], TMP_dic, RPT_dic]
                
                
                triple = json_data[0]
                count = json_data[1]
                TMP = json_data[2]
                RPT = json_data[3]
                
                
                f1.write('triple: ' + triple + '\n')
                f1.write('count: ' + str(count) + '\n')
                f1.write('TMP:' + str(TMP) + '\n')
                f1.write('RPT:' + str(RPT) + '\n')
                f1.write('\n================================\n\n')
            
            else:
                break
            
'''    

cluster_event_infos_complete = []
with io.open('./logs/c22.txt', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                json_data = json.loads(line)
                cluster_event_infos_complete.append(json_data)
            else:
                break
                
                
with io.open('./logs/f33.txt', "a", encoding='utf-8') as f1:
     # [triple, dic['count'], TMP_dic, RPT_dic]
                
    # 聚类
    cluster_event_clusters = event_merge(cluster_event_infos_complete)
    
    # 聚类后按簇大小排序
    cluster_event_clusters = sorted(cluster_event_clusters, key=lambda x:len(x), reverse = True)
           
    f1.write('------------' + '\n')
    f1.write('\ncluster_event_clusters:' + '\n\n')
    for cluster_event_cluster in cluster_event_clusters:
        f1.write('-----' + '\n')
        
        # cluster_event_cluster 按 triple 统计排序 方便观察
        triples_count = {}
        
        for item in cluster_event_cluster:
            m = item['triple_info'] ['triple']
            if str(m) not in triples_count:
                triples_count[str(m)] = 1
            else:
                triples_count[str(m)] += 1
                              
        triples_count = sorted(triples_count.items(), key=lambda x:x[1], reverse = True)
        
        for item, count in triples_count:
            f1.write(str(item) + 'count: ' + str(count) + '\n')
                


            
            
            
            
            