# 读
import json

with open('../8-ner_nre/pytorch_nn_rnn-master/train_data_me.json', 'r') as file:
    str = file.read()
    data = json.loads(str)
    for i in data:
        print(i)

# 写
result_triples = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}

with open('result_triples.json', 'w') as f:
    json.dump(result_triples, f)


# 首先是读文件，有readline和readlines两种方式，readline是逐行读，readlines是直接读入所有行返回一个列表，当文件很大时，readlines会很慢，不建议使用。

# 再将读进来的每行用json.loads()解析，会返回一个字典。

import json
root = "path"
def do_something():
    pass
with open(root, encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: # 到 EOF，返回空字符串，则终止循环
            break
        js = json.loads(line)
# 也可以用fileinput库

import fileinput

for line in fileinput.input(['filename']):
    do_something(line)

# 然后是写，刚开始我用pandas，把新的数据一行行添加到dataframe的后面。这样也存在同样的问题，当数据过多时，会占用很多内存。所以改用一行一行地写。

with open('./allData.txt', "a+") as outFile: # 用追加的方式打开要写入的文件，没有会自动创建
    with open(root, encoding='utf-8') as f: # 打开要读的文件
        while True:
            line = f.readline()
            if not line:
                break
            js = json.loads(line)
            string = js['content']
            outFile.write(string+'\n') # 逐行写入