from godot import exposed, export
from godot import *
import numpy as np
import time
from copy import deepcopy
import csv

CAR = ResourceLoader.load("res://game/car_AI.tscn")
NUMBEROFCARS = 80
start_time = 0
numiters = 0
maxiters = 4000
numberOfFinished = 0
evolution_data = []
time_up = False
best_time = 0
trigger_faster = True
ticks = 0


log_run = False
log_points = [10,20,40,70,100,150,200,300,450,600,800,1000,1200,1500, 2000, 2500, 3000, 4000] # can be longer than maxiters
now = time.localtime()
log_file = '../data/generations{}.txt'.format('_' + '_'.join([str(now.tm_mon),str(now.tm_mday),str(now.tm_hour)]))


load_previous = True
one_by_one = True
from_file = '../data/generations_5_4_19_ordered.txt'



@exposed
class controller(Node):
	carArray = []

	def endSim(self, reason):
		if load_previous:
			pass
		else:
			global now
			s = '_' + '_'.join([str(now.tm_mon),str(now.tm_mday),str(now.tm_hour)])
			with open('../data/evolution_data{}.csv'.format(s), 'w', newline='') as f:
				writer = csv.writer(f)
				writer.writerows(evolution_data)
			self.swarm.to_file(filename='../data/NN_particles{}.txt'.format(s),minpoint=0.7)
			print(reason)
		self.carArray[0].quitSim()

	def restartCars(self):
		travel_lengths = []
		global time_up
		global start_time
		global numberOfFinished
		global best_time
		start_time = time.time()
		global numiters
		global trigger_faster
		global ticks
		global log_run
		global log_points
		global log_file
		global one_by_one
		numiters += 1
		print(numiters%NUMBEROFCARS)
		maxp = 0
		maxi = 0
		if load_previous:
			for i, car in enumerate(self.carArray):
				if car.finished == 1:
					numberOfFinished += 1
					car.finished = 0
				# visszaállítás
				car.position = self.position
				car.rotation_degrees = 0
				car.deactivated = 0
				car.maxSpeed = 21
			numberOfFinished = 0
			ticks = 0
			time_up = False
		else:
			for i, car in enumerate(self.carArray):
				# tanulás
				point = car.getTravelLength()
				if point>1:
					point = 1
				if point > 0.98 and trigger_faster:
					self.swarm.global_best_weight = self.swarm.global_best_weight * 8
					self.swarm.learning_weight = 0.994
					trigger_faster = False
				if car.time:
					point += pow(1.1, -car.time / 10)
				if point > maxp:
					maxp = point
					maxi = i
				self.swarm.particles[i].set_score(point)
				car.time = 0
				travel_lengths.append(car.getTravelLength())
				if car.finished == 1:
					numberOfFinished += 1
					car.finished = 0
				# visszaállítás
				car.position = self.position
				car.rotation_degrees = 0
				car.deactivated = 0
				car.maxSpeed = 21
			if log_run and (numiters in log_points or numiters == maxiters):
				self.swarm.particles[maxi].to_file(filename=log_file)
			best_time = 0
			maxtravel = max(travel_lengths)
			print('max: '+str(maxtravel))
			average = np.mean(travel_lengths)
			med = np.median(travel_lengths)
			print('median: '+str(med))
			evolution_data.append([maxtravel,med,average])
			self.swarm.set_best()
			self.swarm.learn()
			if numberOfFinished == NUMBEROFCARS:
				self.endSim("ending simulation: all cars finished")
			else:
				numberOfFinished = 0
			if numiters == maxiters:
				self.endSim("ending simulation: maximum iterations reached")
			ticks = 0
			time_up = False

	def _ready(self):
		#self.carArray = 0
		global one_by_one
		if load_previous:
			print('setting up from previous run')
			global NUMBEROFCARS
			self.swarm = Swarm(0, local_best_weight = 0.005, global_best_weight = 0.01,
								nn_struct=[0], inputs=0, outputs=0,
								activation_function=np.tanh)
			self.swarm.build_from_file(from_file)
			NUMBEROFCARS = len(self.swarm.particles)
		else:
			self.swarm = Swarm(NUMBEROFCARS, local_best_weight = 0.01, global_best_weight = 0.015,
								nn_struct=[8,4], inputs=6, outputs=2,
								activation_function=np.tanh)
		if one_by_one:
			self.car = CAR.instance()	 # instance car
			self.add_child(self.car)	 # add as child
			self.car.translate(Vector2(self.position.x, self.position.y))  # move to start position
			self.carArray.append(self.car) # add to array of cars
		else:
			for i in range(NUMBEROFCARS):
				# create cars
				self.car = CAR.instance()	 # instance car
				self.add_child(self.car)	 # add as child
				self.car.translate(Vector2(self.position.x, self.position.y))  # move to start position
				self.carArray.append(self.car) # add to array of cars



	def _process(self, delta):
		global start_time
		global best_time
		global ticks
		global time_up
		global one_by_one
		if not time_up and time.time() - start_time > 3:
			time_up = True
		ticks += 1
		for i, car in enumerate(self.carArray):
			if not car.deactivated: # if active
				if car.finished and not car.time:
					car.deactivated = 1
					car.time = ticks
					#if not best_time:
					#	best_time = car.time

				if car.output()[0] < 0.02:
					car.deactivated = 1
				# check if they are all still on track
				# travelLength = 0 until crash, then [0 ... 1]
				if car.travelLength != 0: # if crashed
					car.deactivated = 1 # flag car object to be deactivated

				# get active car parameters for PSO input
				getinfo = car.output() ### <Array([0, None, None])>
				getinfo[0] = getinfo[0] / 21 #self.carArray[i].maxSpeed # 21
				if getinfo[1] == None:
					getinfo[1] = np.random.rand()
				if getinfo[2] == None:
					getinfo[2] = np.random.rand()
				if one_by_one:
					outputs = self.swarm.particles[(numiters-1)%NUMBEROFCARS].calculate(np.array(list(getinfo)))
				else:
					outputs = self.swarm.particles[i].calculate(np.array(list(getinfo)))
				car.input(Array(list(outputs)))
		if all([x.deactivated for x in self.carArray]):
			self.restartCars()

### PSO below ###

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
			
		if not kwargs:
			self.inputs = 0
			self.outputs = 0
			self.struct = []

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
	def __init__(self, kwargs = None):
		self.NN = NeuralNetwork(kwargs)
		self.score = 0
		self.best_score = 0
		self.best_NN = None
		self.learning_vector = []
		self.learning_vector_ratio = 1/40
		self.build_learning_vector()

	def build_learning_vector(self):
		inputs = self.NN.inputs
		outputs = self.NN.outputs
		struct = self.NN.struct
		self.learning_vector = []
		if len(struct)>0:
			for i, size in enumerate(struct):
				if i == 0:
					self.learning_vector.append(np.random.rand(size,inputs+1)*self.learning_vector_ratio
												-(self.learning_vector_ratio/2))
				else:
					self.learning_vector.append(np.random.rand(size,struct[i-1]+1)*self.learning_vector_ratio
												-(self.learning_vector_ratio/2))
			self.learning_vector.append(np.random.rand(outputs,struct[-1]+1)*self.learning_vector_ratio
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


	def to_file(self, filename = None, minpoint = 0):
		for particle in self.particles:
			particle.to_file(filename=filename, minpoint=minpoint)


	def build_from_file(self, filename):
		self.n = 0
		self.particles = []
		i = 0
		print('Reading neural-networks')
		while True:
			try:
				nn = NeuralNetwork()
				nn.generate_from_file(filename, place = i)
				p = Particle()
				p.NN = nn
				p.build_learning_vector()
				self.particles.append(p)
				self.n +=1
				i += 1
				print('{} ready'.format(i))
			except Exception as e:
				print(e)
				break
