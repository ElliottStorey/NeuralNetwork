import numpy as np

'''
Sample Square

##
#*

Input: [1, 1, 1, 0.2]
Output: [0.8]
'''

def generate(amount):
    inputs = []
    outputs = []
    for _ in range(amount):
        square = np.array([np.random.random() for _ in range(4)])
        inputs.append(square)
        brightness = np.array(np.mean(square))
        outputs.append(brightness)
    return dict(inputs=inputs, outputs=outputs)