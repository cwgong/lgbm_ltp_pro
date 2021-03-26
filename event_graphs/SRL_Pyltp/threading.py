# -*- coding: utf-8 -*-
import threading
import json
import requests
import time

'''
构造方法： 
Thread(group=None, target=None, name=None, args=(), kwargs={}) 

　　group: 线程组，目前还没有实现，库引用中提示必须是None； 
　　target: 要执行的方法； 
　　name: 线程名； 
　　args/kwargs: 要传入方法的参数。

实例方法： 
　　isAlive(): 返回线程是否在运行。正在运行指启动后、终止前。 
　　get/setName(name): 获取/设置线程名。 

　　start():  线程准备就绪，等待CPU调度
　　is/setDaemon(bool): 获取/设置是后台线程（默认前台线程（False））。（在start之前设置）

　　　　如果是后台线程，主线程执行过程中，后台线程也在进行，主线程执行完毕后，后台线程不论成功与否，主线程和后台线程均停止
       　　如果是前台线程，主线程执行过程中，前台线程也在进行，主线程执行完毕后，等待前台线程也执行完成后，程序停止
　　start(): 启动线程。 
　　join([timeout]): 阻塞当前上下文环境的线程，直到调用此方法的线程终止或到达指定的timeout（可选参数）。
'''

# 分词
def split_sentence(sen,a):
    nlp_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough'
    try:
        cut_sen = dict()
        cut_sen['content'] = sen
        data = json.dumps(cut_sen).encode("UTF-8")
        cut_response = requests.post(nlp_url, data=data, headers={'Connection':'close'})
        cut_response_json = cut_response.json()
        a.append(cut_response_json['data'][0]['word'])
        return cut_response_json['data']
        
    except Exception as e:
        print("Exception: {}".format(e))
        return []
    
# 多线程分词
def split_sentence_for_thread(i, sen, result):
    nlp_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough'
    try:
        cut_sen = dict()
        cut_sen['content'] = sen
        data = json.dumps(cut_sen).encode("UTF-8")
        cut_response = requests.post(nlp_url, data=data, headers={'Connection':'close'})
        cut_response_json = cut_response.json()
        result[i] = cut_response_json['data']
    except Exception as e:
        print("Exception: {}".format(e))
        result[i] = cut_response_json['data']
    
def for_test():
    print('for_test():\n')
    
    s1 = '全国政协副主席张庆黎辜胜阻、刘新成、何维、邵鸿、高云龙出席会议辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    s2 = '刘奇葆、董建华、何厚铧、'
    s3 = '卢展工、王正伟、马飚、陈晓光、'
    s4 = '梁振英、杨传堂、李斌、汪永清、'
    s5 = '苏辉、郑建邦、辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    
    jobStart = time.time()
    a = []
    for s in [s1,s2,s3,s4,s5,s1,s2,s3,s4,s5,s1,s2,s3,s4,s5]*10:
        split_sentence(s,a)
    jobEnd = time.time()
    print(a)
    print("batch spendTime : ", time.localtime(jobEnd - jobStart).tm_sec, "sec")
                 
def thread_test():
    print('thread_test():\n')

    s1 = '全国政协副主席张庆黎辜胜阻、刘新成、何维、邵鸿、高云龙出席会议辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    s2 = '刘奇葆、董建华、何厚铧、'
    s3 = '卢展工、王正伟、马飚、陈晓光、'
    s4 = '梁振英、杨传堂、李斌、汪永清、'
    s5 = '苏辉、郑建邦、辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    
    
    jobStart = time.time()
    a = []
    threads = []
    for s in [s1,s2,s3,s4,s5,s1,s2,s3,s4,s5,s1,s2,s3,s4,s5]*10:
        t = threading.Thread(target=split_sentence, args=(s,a))
        threads.append(t)
    
    jobStart = time.time()
    for t in threads:
        t.setDaemon(True) # 设置为后台线程，随主线程共存亡
        t.start()
    
    for t in threads:
        t.join() # 等待每一个线程结果
    
    print(a)
    jobEnd = time.time()
    print("batch spendTime : ", time.localtime(jobEnd - jobStart).tm_sec, "sec")
    
def thread_sort_test():
    print('thread_sort_test():\n')

    s1 = '全国政协副主席张庆黎辜胜阻、刘新成、何维、邵鸿、高云龙出席会议辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    s2 = '刘奇葆、董建华、何厚铧、'
    s3 = '卢展工、王正伟、马飚、陈晓光、'
    s4 = '梁振英、杨传堂、李斌、汪永清、'
    s5 = '苏辉、郑建邦、辜胜阻、刘新成、何维、邵鸿、高云龙出席会议'
    
    
    jobStart = time.time()
    result = {}
    threads = []
    s_list = [s1,s2,s3,s4,s5,s1,s2,s3,s4,s5,s1,s2,s3,s4,s5]*10
    for i in range(len(s_list)):
        s = s_list[i]
        t = threading.Thread(target=split_sentence_for_thread, args=(i,s,result))
        threads.append(t)
    
    jobStart = time.time()
    for t in threads:
        t.setDaemon(True) # 设置为后台线程，随主线程共存亡
        t.start()
    
    for t in threads:
        t.join() # 等待每一个线程结果
    
    result = sorted(result.items(), key=lambda x:x[0], reverse = False)
    for key, value in result:
        print(key, value[0]['word'])
        
    jobEnd = time.time()
    print("batch spendTime : ", time.localtime(jobEnd - jobStart).tm_sec, "sec")
             
            
if __name__ == '__main__':
    
    for_test()  
    print()
    thread_test()
    print()
    thread_sort_test()
            
            
            