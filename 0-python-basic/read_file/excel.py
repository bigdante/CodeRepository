import json
from pandas import json_normalize
def json_to_excel(path, save):
    result_json = json.load(open(path))
    dat_table = json_normalize(result_json)
    dat_table.to_excel(save, index=False)


def excel_to_json(path, save):
    df = pd.read_excel(path)
    cols = [colName for colName in df.columns]
    json_list = []
    for row in df.itertuples():
        json_dict = {}
        for index in range(len(cols)):
            json_dict[cols[index]] = getattr(row, cols[index])
        json_list.append(json_dict)
    with open(save, "w", encoding="utf-8") as fw:
        json.dump(json_list, fw, ensure_ascii=False, indent=4)