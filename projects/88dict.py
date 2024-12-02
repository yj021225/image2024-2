sales = {'colar':100, 'cider':20, 'juice':{'orange':3, 'apple':9}}

print(sales['cider'])
print(sales.keys())
print(sales.values())

for key, val in sales.items():
    print(f'{key=}')
    print(f'{val=}')

resp = {'face_1': {'score': 0.9995705485343933,
                   'facial_area': [400, 95, 490, 207],
                   'landmarks': {'right_eye': [431.66342, 137.51898],
                                 'left_eye': [473.41107, 138.92555]}
        },
        'face_2': {'score': 0.9995139241218567,
                   'facial_area': [155, 93, 251, 212],
                   'landmarks': {'right_eye': [182.71143, 143.53116],
                                 'left_eye': [227.43085, 143.12755]}
        }}

for key, face in  resp.items():
    print(face['score'])
    print(face['facial_area'][0])

import time
start = time.time()
print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
end = time.time()
print((end-start)*1000)