from numba.np.ufunc import parallel
import numpy as np
from numba import int8, int16, float32    # import the types
from numba import njit, prange
from numba.experimental import jitclass
import struct

def binary(num):
	return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

spec = [
	('inputNeurons', float32[:, :]),
	('activationsInput', int8[:, :, :]),
	('connectionsInput', int16[:, :, :]),
	('middle', float32[:, :]),
	('activationsMiddle', int8[:, :, :]),
	('connectionsMiddle', int16[:, :, :]),
	('outputNeurons', float32[:, :]),
]

@jitclass(spec)
class Network(object):
	def __init__(self, numConnections, numInput, bitsInput, sizeMiddle, numLayers, numOutput, bitsOutput):
		self.inputNeurons = np.zeros((numInput, bitsInput), dtype=np.float32)
		self.activationsInput = np.zeros((numInput, bitsInput, numConnections), dtype=np.int8)
		self.connectionsInput = np.zeros((numInput, bitsInput, numConnections), dtype=np.int16)
		self.middle = np.zeros((numLayers, sizeMiddle), dtype=np.float32)
		self.activationsMiddle = np.zeros((numLayers, sizeMiddle, numConnections), dtype=np.int8)
		self.connectionsMiddle = np.zeros((numLayers, sizeMiddle, numConnections), dtype=np.int16)
		self.outputNeurons = np.zeros((numOutput, bitsOutput), dtype=np.float32)

	def inputVals(self):
		return self.inputNeurons
	
	# @njit(parallel=True)
	def giveInput(self, inputNums):
		print(inputNums)
		for i in prange(len(inputNums)):
			for b in prange(len(inputNums[i])):
				if inputNums[i, b] == '1':
					self.inputNeurons[i, b] = 100.0
				else:
					self.inputNeurons[i, b] = 0.0

	def increment(self, val):
		for i in range(self.size):
				self.array[i] += val
		return self.array

	@staticmethod
	def add(x, y):
		return x + y

mybag = Network(300, 100, 16, 1000, 20, 10, 16)
inputs = []
for i in range(16):
	inputs.append(binary(float32(3.0)))

print(inputs)
mybag.giveInput(inputs)
print(mybag.inputVals())
