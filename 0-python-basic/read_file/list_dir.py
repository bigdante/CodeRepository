import os
path = ".."
data_name = list(map(lambda x: path + '/' + x, os.listdir(path)))
print(data_name)

# 功能更强大的walk
for curDir, dirs, files in os.walk(".."):
    print("====================")
    print("现在的目录：" + curDir)
    print("该目录下包含的子目录：" + str(dirs))
    print("该目录下包含的文件：" + str(files))