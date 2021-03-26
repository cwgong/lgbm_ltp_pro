import io
from LAC import LAC
import json

def loading_data(input_file):
    with io.open(input_file,"w",encoding="utf-8") as f:
        ori_data_list = json.load(f)
    ori_data_str = "".join(ori_data_list)
    data_list = ori_data_str.split("。")
    return data_list

def choose_suitable_reg(seg_words):
    pass


def extract_relation_from_data(input_file):
    lac = LAC()
    seg_words_filter = []
    data_list = loading_data(input_file)
    seg_words_lac = lac.run(data_list)
    for seg_words in seg_words_lac:
        # if ("PER" or "LOC" or "ORG") not in seg_words[1]:
        #     continue
        idx = 0
        for character in seg_words[1]:
            idx += 1
            if character == ("PER" or "LOC" or "ORG"):
                pass



