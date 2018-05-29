# -*- coding:utf-8 -*-

from load_data import load_data
from const import const

class preprocess(object):
    '''
    简单的文本预处理：小写化，空值检测
    '''

    def lower_case(self, data_dict):
        self._null_check(data_dict)
        for index,element in data_dict.items():
            data_dict[index]['question'] = element['question'].lower()
        return data_dict

    def _null_check(self, data_dict):
        '''
        空值检测，去除空问题
        '''
        for index,element in data_dict.items():
            if len(element['question'].strip()) == 0:
                data_dict.pop(index)
        return data_dict

if __name__ == '__main__':
    l = load_data()
    r = l.load_train_set(const.REPOSITORY_FILE_PATH, const.REPOSITORY_SHEET_NAME)
    # print(r)
    p = preprocess()
    d = p.lower_case(r)
    print(d)