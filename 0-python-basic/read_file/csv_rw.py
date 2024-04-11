import numpy as np
import pandas as pd

# 读
path = '../../6-bert/dataset/test.csv'
df = pd.read_csv('../../6-bert/dataset/test.csv')
print(df.head())


col_names = ['Id',
             'Survived',
             'Passenger Class',
             'Full Name',
             'Gender', ]
titanic_data = pd.read_csv(path, names=col_names, header=None)
print(titanic_data.head())
# 但是问题是，原来的标题变成了第一行

# 写
li = [['Sacramento', 'California'], ['Miami', 'Florida']]
city = pd.DataFrame(li, columns=['City', 'State'])
city.to_csv('city.csv')

# 遍历
for index, row in titanic_data.iterrows():
    print(row['Full Name'])

# 假设我们要删除的列的名称为 ‘观众ID’,‘评分’ :
# df = df.drop(['观众ID', '评分'], axis=1)
# 删除指定列
# df.drop(columns=["城市"])

# 删除某几行
df.drop([1, 2])  # 删除1,2行的整行数据
# 删除行（某个范围）
df.drop(df.index[3:6], inplace=True)

# 合并csv

# sales1 = 'salesdata1.csv'
# sales2 = 'salesdata2.csv'

# merge files
# dataFrame = pd.concat(map(pd.read_csv, [sales1, sales2]), ignore_index=True)
# df.to_csv("Alldatas.csv", index=False, header=False, mode='a+')
# print(dataFrame)

# # 单列的内连接

# # 定义df1
df1 = pd.DataFrame({'alpha': ['A', 'B', 'B', 'C', 'D', 'E'],
                    'feature1': [1, 1, 2, 3, 3, 1],
                    'feature2': ['low', 'medium', 'medium', 'high', 'low', 'high']})
# 定义df2
df2 = pd.DataFrame({'alpha': ['A', 'A', 'B', 'F'],
                    'pazham': ['apple', 'orange', 'pine', 'pear'],
                    'kilo': ['high', 'low', 'high', 'medium'],
                    'price': np.array([5, 6, 5, 7])})
# 基于共同列alpha的内连接
df3 = pd.merge(df1, df2, how='inner', on='alpha')



