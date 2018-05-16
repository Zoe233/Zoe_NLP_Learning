# -*- coding:utf-8 -*-

import urllib.request
import urllib.parse
def call_rest_api():
    url_get_base = 'https://api.ltp-cloud.com/analysis/'
    args = {
        'api_key':'n1K8v6W0P4RhIB4krUVWouPu2ObecDjsHniDFItp',
        'text':'我是中国人',
        'pattern':'dp',
        'format':'plain'
    }
    result = urllib.request.urlopen(url_get_base, urllib.parse.urlencode(args).encode())  # POST method
    content = result.read().strip()
    print(content.decode())

if __name__ == '__main__':
    call_rest_api()
