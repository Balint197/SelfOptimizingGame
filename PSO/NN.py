import numpy as np
from copy import deepcopy



def format_np_array(arr):
    s = ''
    arr = arr.flatten()
    for element in arr:
        if isinstance(element, np.int32):
            s += str(element) + ' '
        else:
            s += '{:.8f}'.format(element) + ' '
    s = s.strip() + '\n'
    return s

def np_from_str(s):
    arr = []
    pieces = s.strip().split()
    for piece in pieces:
        if '.' in piece:
            arr.append(float(piece))
        else:
            arr.append(int(piece))
    arr = np.array(arr)
    return arr

def relu(x):
    return x*(x>0)

class NeuralNetwork():
    def __init__(self, kwargs = None):
        """
        Builds the Neural network based on given struct, (and example, if given)
        Example-based building also needs std.
        Necessary arguments for buildign NN:
          - inputs
          - outputs
          - nn_struct
              n pieces of integers, where n is the number of layers,
              and the integers specify the number of neurons in each layer
          - activation_function
        """
        
        if kwargs and 'nn_struct' in kwargs.keys():
            if 'nn_std' in kwargs.keys() and 'nn_example' in kwargs.keys():
                self.generate_by_example(kwargs['inputs'], kwargs['outputs'], kwargs['nn_struct'],
                                         kwargs['nn_example'], kwargs['nn_std'])
            elif 'copy' in kwargs.keys():
                self.copy(kwargs['copy'])
            else:
                self.generate_by_struct(kwargs['inputs'], kwargs['outputs'], kwargs['nn_struct'],
                                        kwargs['activation_function'])
        elif kwargs and 'file' in kwargs.keys():
            self.generate_from_file(kwargs['file'])


    def generate_from_file(self, filename, place = None, seek_best = False, act_func = None):
        if not filename:
            raise Exception('No filename was given for importing NeuralNetwork')
        with open(filename, 'r') as f:
            lines = f.readlines()
            if place and not seek_best:
                line_i = 0
                struct_i = np_from_str(lines[line_i])
                for i in range(place):
                    line_i += len(struct_i) + 3
                    struct_i = np_from_str(lines[line_i])
                self.inputs = struct_i[0]
                self.outputs = struct_i[-1]
                self.struct = struct_i[1:-1]
                layers = []
                for i in range(len(self.struct)):
                    layer = np_from_str(lines[i+line_i+1])
                    if i==0:
                        layer = layer.reshape(self.struct[0],self.inputs+1)
                    else:
                        layer = layer.reshape(self.struct[i], self.struct[i-1]+1)
                    layers.append(layer)
                layer = np_from_str(lines[line_i+len(self.struct)+1])
                layer = layer.reshape(self.outputs,self.struct[-1]+1)
                layers.append(layer)
                self.layers = np.array(layers,dtype=object)
                if lines[line_i + len(self.struct) + 2][:-1] == 'tanh':
                    self.act_func = np.tanh
                elif lines[line_i + len(self.struct) + 2][:-1] == 'relu':
                    self.act_func = relu
                elif act_func:
                    self.act_func = act_func
                else:
                    raise Exception('No activation function given')
            elif seek_best:
                indices_points = []
                line_i = 0
                while True:
                    struct_i = np_from_str(lines[line_i])
                    ind = line_i
                    p = np_from_str(lines[line_i+len(struct_i)+1])
                    indices_points.append([ind,p])
                    if line_i + len(struct_i) + 3 > len(lines):
                        break
                    else:
                        line_i += len(struct_i) + 3
            else:
                struct = np_from_str(lines[0])
                self.inputs = struct[0]
                self.outputs = struct[-1]
                self.struct = struct[1:-1]
                layers = []
                for i in range(len(self.struct)):
                    layer = np_from_str(lines[i+1])
                    if i==0:
                        layer = layer.reshape(self.struct[0],self.inputs+1)
                    else:
                        layer = layer.reshape(self.struct[i],self.struct[i-1]+1)
                    layers.append(layer)
                layer = np_from_str(lines[len(self.struct)+1])
                layer = layer.reshape(self.outputs,self.struct[-1]+1)
                layers.append(layer)
                self.layers = np.array(layers,dtype=object)
                if lines[len(self.struct) + 2][:-1] == 'tanh':
                    self.act_func = np.tanh
                elif act_func:
                    self.act_func = act_func
                else:
                    raise Exception('No activation function given') 
        if seek_best:
            indices_points = indices_points.sort(key=lambda x: x[1])
            if place:
                best_i = indices_points[place][0]
            else:
                best_i = indices_points[0][0]
            with open(filename, 'r') as f:
                lines = f.readlines()
                struct_i = np_from_str(lines[best_i])
                self.inputs = struct_i[0]
                self.outputs = struct_i[-1]
                self.struct = struct_i[1:-1]
                layers = []
                for i in range(len(self.struct)):
                    layer = np_from_str(lines[line_i+i])
                    if i==0:
                        layer = layer.reshape(self.struct[0],self.inputs+1)
                    else:
                        layer = layer.reshape(self.struct[i],self.struct[i-1]+1)
                    layers.append(layer)
                layer = np_from_str(lines[best_i+len(self.struct)+1])
                layer = layer.reshape(self.outputs,self.struct[-1]+1)
                layers.append(layer)
                self.layers = np.array(layers,dtype=object)
                if lines[best_i + len(self.struct) + 2][:-1] == 'tanh':
                    self.act_func = np.tanh
                elif act_func:
                    self.act_func = act_func
                else:
                    raise Exception('No activation function given')
            

    def to_file(self, filename=None, points = -1):
        if filename:
            if '.' in filename:
                if not filename[-4:]=='.txt':
                    raise Exception('Save destination file must be txt')
            else:
                filename += '.txt'
        else:
            filename = 'save'
            for layer in self.struct:
                filename += '_'+str(layer)
            filename += '.txt'
        with open(filename, 'a') as f:
            struct = np.append([self.inputs],self.struct)
            struct = np.append(struct, [self.outputs])
            struct = format_np_array(struct)
            f.write(struct)
            for i, layer in enumerate(self.layers):
                f.write(format_np_array(layer))
            if self.act_func == np.tanh:
                f.write('tanh\n')
            else:
                try:
                    if self.act_func == relu:
                        f.write('relu\n')
                except NameError:
                    f.write('user\n')

            f.write(str(points) + '\n' + '\n')
            

    def generate_by_struct(self, inputs, outputs, struct, act):
        self.inputs = inputs
        self.outputs = outputs
        self.struct = struct
        self.act_func = act
        self.layers = []
        for i, size in enumerate(struct):
            if i == 0:
                self.layers.append(np.random.rand(size,inputs + 1)*2-1)
            else:
                self.layers.append(np.random.rand(size,struct[i-1] + 1)*2-1)
        self.layers.append(np.random.rand(outputs,struct[-1] + 1)*2-1)
        self.layers = np.array(self.layers,dtype=object)


    def generate_by_example(self, inputs, outputs, struct, example, std):
        pass #_____________________________


    def calculate(self, inputs):
        inter = inputs
        for i, layer in enumerate(self.layers):
            inter = self.act_func(np.matmul(layer,np.append([1],inter)))
        return inter


class Particle():
    def __init__(self, kwargs):
        self.NN = NeuralNetwork(kwargs)
        self.score = 0
        self.best_score = 0
        self.best_NN = None
        self.learning_vector = []
        self.learning_vector_ratio = 1/40
        struct = kwargs['nn_struct']
        for i, size in enumerate(struct):
            if i == 0:
                self.learning_vector.append(np.random.rand(size,kwargs['inputs']+1)*self.learning_vector_ratio
                                            -(self.learning_vector_ratio/2))
            else:
                self.learning_vector.append(np.random.rand(size,struct[i-1]+1)*self.learning_vector_ratio
                                            -(self.learning_vector_ratio/2))
        self.learning_vector.append(np.random.rand(kwargs['outputs'],struct[-1]+1)*self.learning_vector_ratio
                                            -(self.learning_vector_ratio/2))
        self.learning_vector = np.array(self.learning_vector,dtype=object)


    def learn(self, global_best, learning_weight, local_weight, global_weight):
        self.learning_vector = self.learning_vector * learning_weight
        if self.best_NN:
            self.learning_vector = self.learning_vector + ((self.best_NN.layers - self.NN.layers) * local_weight * np.random.rand())
        if global_best:
            self.learning_vector = self.learning_vector + ((global_best.layers - self.NN.layers) * global_weight * np.random.rand())
        self.check_boundaries()
        self.NN.layers = self.NN.layers + self.learning_vector     


    def set_score(self, score):
        self.score = score
        if score > self.best_score:
            self.best_score = score
            self.best_NN = deepcopy(self.NN)


    def calculate(self, inputs):
        return self.NN.calculate(inputs)


    def check_boundaries(self):
        """
        Checks if any NN node or value exceeds the (-1;1) boundary.
        If they do, the correspondig learning vector will be mirrored.
        """
        for i,layer in enumerate(self.NN.layers):
            for j, neuron in enumerate(layer):
                for k, weight in enumerate(neuron):
                    if weight > 2 and self.learning_vector[i][j][k]>0:
                        self.NN.layers[i][j][k] = 2
                        self.learning_vector[i][j][k] *= -1
                    if weight < -2 and self.learning_vector[i][j][k]<0:
                        self.NN.layers[i][j][k] = -2
                        self.learning_vector[i][j][k] *= -1
                        

    def global_distance(self, global_best):
        if global_best:
            diff = global_best.layers - self.NN.layers
            dist = sum([np.sum(x) for x in np.abs(diff)])
            return dist
        else:
            return None

    def to_file(self, filename = None, minpoint = 0):
        if self.score>minpoint:
            self.NN.to_file(filename, self.score)


class Swarm():
    def __init__(self, n, local_best_weight = 0.05, global_best_weight = 0.1, **kwargs):
        """
        Creates the swarm with given n specimens.
        The specifications for the neural networks also need to be included here.
        """
        self.n = n
        self.particles = []
        for i in range(n):
            self.particles.append(Particle(kwargs))
        self.best_NN = None
        self.best_score = 0
        self.learning_weight = 1
        self.local_best_weight = local_best_weight
        self.global_best_weight = global_best_weight


    def learn(self):
        for particle in self.particles:
            particle.learn(self.best_NN, self.learning_weight,
                           local_weight=self.local_best_weight,
                           global_weight=self.global_best_weight)


    def get_scores(self):
        points = []
        for particle in self.particles:
            points.append(particle.best_score)


    def set_best(self):
        for particle in self.particles:
            if particle.score > self.best_score:
                self.best_score = particle.score
                self.best_NN = deepcopy(particle.NN)


    def get_global_distances(self):
        if self.best_NN:
            dist = 0
            for particle in self.particles:
                dist += particle.global_distance(self.best_NN)
            nodes_count = sum([x.shape[0]*x.shape[1] for x in self.best_NN.layers])
            return dist/nodes_count/self.n  # normalizing the value (bigger NN would mean bigger distance)
        else:
            return None

    
    def visualize_learning(self):
        pass #_____________________________

    def to_file(self, filename = None, minpoint = 0):
        for particle in self.particles:
            particle.to_file(filename=filename, minpoint=minpoint)

    
