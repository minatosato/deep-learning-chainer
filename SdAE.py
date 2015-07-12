# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

from dAE import DenoisingAutoEncoder

N = 60000
N_test = 10000

class SdAE:
	def __init__(self, rng, data, target, n_inputs=784, n_hidden=[784,784,784], n_outputs=10, gpu=-1):
		self.model = FunctionSet(l1=F.Linear(n_inputs, n_hidden[0]),
								 l2=F.Linear(n_hidden[0], n_hidden[1]),
								 l3=F.Linear(n_hidden[1], n_hidden[2]),
								 l4=F.Linear(n_hidden[2], n_outputs))

		if gpu >= 0:
			self.model.to_gpu()

		self.rng = rng
		self.gpu = gpu
		self.data = data
		self.target = target

		self.x_train, self.x_test = np.split(data, [N])
		self.y_train, self.y_test = np.split(target, [N])

		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs

		self.dae1 = None
		self.dae2 = None
		self.dae3 = None
		self.optimizer = None
	
	def setup_optimizer(self):
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model.collect_parameters())

	def pre_train(self, n_epoch=5):
		first_inputs = self.data
		
		# initialize first dAE
		self.dae1 = DenoisingAutoEncoder(self.rng, first_inputs,
										 n_inputs=self.n_inputs,
										 n_hidden=self.n_hidden[0],
										 gpu=self.gpu)
		# train first dAE
		self.dae1.train_and_test(n_epoch=n_epoch, batchsize=100)
		# compute second iputs for second dAE
		second_inputs = self.dae1.compute_hidden(first_inputs)
		# initialize second dAE
		self.dae2 = DenoisingAutoEncoder(self.rng, second_inputs,
										 n_inputs=self.n_hidden[0],
										 n_hidden=self.n_hidden[1],
										 gpu=self.gpu)
		# train second dAE
		self.dae2.train_and_test(n_epoch=n_epoch, batchsize=100)
		# compute third inputs for third dAE
		third_inputs = self.dae2.compute_hidden(second_inputs)
		# initialize third dAE
		self.dae3 = DenoisingAutoEncoder(self.rng, third_inputs,
										 n_inputs=self.n_hidden[1],
										 n_hidden=self.n_hidden[2],
										 gpu=self.gpu)
		# train third dAE
		self.dae3.train_and_test(n_epoch=n_epoch, batchsize=100)

		# update model parameters
		self.model.l1 = self.dae1.encoder()
		self.model.l2 = self.dae2.encoder()
		self.model.l3 = self.dae3.encoder()

		self.setup_optimizer()

	def forward(self, x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.sigmoid(self.model.l1(x)), train=train)
		h2 = F.dropout(F.sigmoid(self.model.l2(h1)), train=train)
		h3 = F.dropout(F.sigmoid(self.model.l3(h2)), train=train)
		y = self.model.l4(h3)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

	def fine_tune(self, n_epoch=20, batchsize=100):
		for epoch in xrange(1, n_epoch+1):
			print 'fine tuning epoch ', epoch

			perm = self.rng.permutation(N)
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, N, batchsize):
				x_batch = self.x_train[perm[i:i+batchsize]]
				y_batch = self.y_train[perm[i:i+batchsize]]

				if self.gpu >= 0:
					x_batch = cuda.to_gpu(x_batch)
					y_batch = cuda.to_gpu(y_batch)

				self.optimizer.zero_grads()
				loss, acc = self.forward(x_batch, y_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

			print 'fine tuning train mean loss={}, accuracy={}'.format(sum_loss/N, sum_accuracy/N)

			# evaluation
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, N_test, batchsize):
				x_batch = self.x_test[i:i+batchsize]
				y_batch = self.y_test[i:i+batchsize]

				if self.gpu >= 0:
					x_batch = cuda.to_gpu(x_batch)
					y_batch = cuda.to_gpu(y_batch)

				loss, acc = self.forward(x_batch, y_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

			print 'fine tuning test mean loss={}, accuracy={}'.format(sum_loss/N_test, sum_accuracy/N_test)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MNIST')
	parser.add_argument('--gpu', '-g', default=-1, type=int,
						help='GPU ID (negative value indicates CPU)')
	args = parser.parse_args()

	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)

	rng = np.random.RandomState(1)
	
	if args.gpu >= 0:
		cuda.init(args.gpu)

	n_hidden = [784,500,200]
	SDA = SdAE(rng=rng, data=mnist.data, target=mnist.target, n_hidden=n_hidden, gpu=args.gpu)
	SDA.pre_train(n_epoch=10)
	SDA.fine_tune(n_epoch=20)

	sys.exit()