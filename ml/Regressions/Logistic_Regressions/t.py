import numpy as np
house_size = [1,2104,1,1416,1,1534,1,852]

house_size = np.array(house_size).reshape(4,2)
print('house_size：')
print(house_size)

hypothis_models_matrix = [ -40,0.25,200,0.1,-159,0.4]
hypothis_models_matrix = np.array(hypothis_models_matrix).reshape(3,2)
print('hypothis_models_matrix:')
print(hypothis_models_matrix)

trans_H = np.transpose(hypothis_models_matrix)
print('trans_H')
print(trans_H)

res = np.dot(house_size,trans_H)
print('res')
print(res)