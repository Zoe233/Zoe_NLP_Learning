# -*- coding:utf-8 -*-
import os

class Constants(object):
    '''
    常量
    '''
    class load_data(object):
        BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
        REPOSITORY_FILE_PATH = os.path.join(BASE_DIR, 'repository.xls')
        TEST_FILE_PATH = os.path.join(BASE_DIR, 'test_data.xlsx')
        STOP_WORDS_FILE_PATH = os.path.join(BASE_DIR, 'stop_words.xlsx')
        USERDICT_FILE_PATH = os.path.join(BASE_DIR, 'userdict.txt')
        REPOSITORY_SHEET_NAME = 'main'
        TEST_SHEET_NAME_A = '2834'
        TEST_SHEET_NAME_B = '3110'
        TEST_SHEET_NAME_C = '7927'

class _const:
  class ConstError(TypeError): pass
  class ConstCaseError(ConstError): pass

  def __setattr__(self, name, value):
      if name in self.__dict__:
          raise self.ConstError("can't change const %s" % name)
      if not name.isupper():
          raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
      self.__dict__[name] = value

const = _const()
const.PI = 3.14

# import sys
# sys.modules[__name__] = _const()