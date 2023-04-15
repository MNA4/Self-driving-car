# imports
import math
import numpy
import random

# classes
class Node:
    def __init__(self, value = None, bias = 0):
        self.value = value
        self.bias = bias
        self.input_paths = []
        
    def calculate(self):
        n = self.bias
        for p in self.input_paths:
            #if p.input.value == None:
                #p.input.calculate()
            n += p.weight * p.input.value
            
        self.value = activation(n)

class Path:
    def __init__(self, input_, output, weight = 0):
        self.weight = weight
        self.input = input_
        self.output = output

        self.output.input_paths.append(self)

class NeuralNetwork:
    def __init__(self, n_input, n_output, n_hlayers, l_hlayers):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hlayers = n_hlayers
        self.l_hlayers = l_hlayers
#        self.inodes = []
        self.hnodes = []
        self.onodes = []
        self.n_path = n_hlayers * n_input + n_output * n_hlayers + n_hlayers**2 * (l_hlayers - 1)
        self.n_node = n_input + n_output + n_hlayers * l_hlayers

#        for i in range(n_input):
#            self.inodes.append(Node())
        for i in range(l_hlayers):
            self.hnodes.append([])
            for j in range(n_hlayers):
                self.hnodes[i].append(0)
        for i in range(n_output):
            self.onodes.append(0)
    def calculate(self, biases, weights, inputs):
        w_index = 0
        b_index = 0
        for i in range(self.l_hlayers):
            for j in range(self.n_hlayers):
                if i == 0:
                    self.hnodes[i][j] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+len(inputs)]*inputs))
                    w_index += len(inputs)
                else:
                    self.hnodes[i][j] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+self.n_hlayers]*self.hnodes[i-1]))
                    w_index += self.n_hlayers
                b_index += 1
        for i in range(self.n_output):
            self.onodes[i] = activation(biases[b_index] + numpy.sum(weights[w_index:w_index+self.n_hlayers]*self.hnodes[-1]))
            w_index += self.n_hlayers
            b_index += 1
        return self.onodes
            


#        for i in range(len(self.inodes)):
#           self.inodes[i].value = inputs[i]
#        for i in range(len(self.hnodes)):
#            for j in range(len(self.hnodes[0])):
#               self.hnodes[i][j].bias = biases[b_index]
#                for x in self.hnodes[i][j].input_paths:
#                    x.weight = weights[w_index]
#                    w_index += 1
#                b_index += 1
#        for i in range(len(self.onodes)):
#            self.onodes[i].bias = biases[b_index]
#            for x in self.onodes[i].input_paths:
#                x.weight = weights[w_index]
#                w_index += 1
#            b_index += 1
#            
#        for i in range(len(self.hnodes)):
#            for j in range(len(self.hnodes[0])):
#                self.hnodes[i][j].calculate()
#        for i in range(len(self.onodes)):
#            self.onodes[i].calculate()
#        return [i.value for i in self.onodes]
        
        
        
    
# functions
def activation(x): # Thanks to: https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    return 1 / (1 + math.exp(-x))

def r():
    return random.random()*2-1


nn = NeuralNetwork(1,1,3, 2)
mutation_rate = 1/1
b_biases = 0
b_weights = 0
outputs = []
while 1:
    for i in range(10):
        biases = b_biases + (numpy.random.rand(nn.n_node)*2-1)*mutation_rate
        weights = b_weights + (numpy.random.rand(nn.n_path)*2-1)*mutation_rate
        score = 0
        for j in range(10):
            inputs = [j/10]
            output = nn.calculate(biases, weights, inputs)[0]
            if inputs[0] >= .5:
                score += abs(output-1)
            else:
                score += abs(output)
        score/=10
        outputs.append([biases, weights, score])

    winner = min(outputs, key = lambda x:x[2])
    print(winner[-1])
    outputs = [winner]
    b_biases = winner[0]
    b_weights = winner[1]
