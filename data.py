import random, statistics

# Sample Square
# Input: [1, 1, 1, 0.2]
# Output: [0.8]

##
#*

def generate(amount):
    inputs = []
    outputs = []
    for _ in range(amount):
        square = [random.random() for _ in range(4)]
        inputs.append(square)
        outputs.append(statistics.mean(square))
    return dict(inputs=inputs, outputs=outputs)