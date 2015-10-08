# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F


class MLP:
	def __init__(self, data, target, n_inputs=784, n_hidden=784, n_outputs=10, gpu=-1):

		self.model = FunctionSet(l1=F.Linear(n_inputs, n_hidden),
								 l2=F.Linear(n_hidden, n_hidden),
								 l3=F.Linear(n_hidden, n_outputs))

		if gpu >= 0:
			self.model.to_gpu()
			self.xp = cuda.cupy
		else:
			self.xp = np

		self.x_train, self.x_test = data
		self.y_train, self.y_test = target

		self.n_train = len(self.y_train)
		self.n_test = len(self.y_test)
		
		self.gpu = gpu
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model)

	def forward(self, x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.relu(self.model.l1(x)), train=train)
		h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
		y = self.model.l3(h2)
		return F.softmax_cross_entropy(y, t), F.accuracy(y,t)


	def train_and_test(self, n_epoch=20, batchsize = 100):
		for epoch in xrange(1, n_epoch+1):
			print 'epoch', epoch

			perm = np.random.permutation(self.n_train)
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.xp.asarray(self.x_train[perm[i:i+batchsize]])
				y_batch = self.xp.asarray(self.y_train[perm[i:i+batchsize]])

				real_batchsize = len(x_batch)

				self.optimizer.zero_grads()
				loss, acc = self.forward(x_batch, y_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

			print 'train mean loss={}, accuracy={}'.format(sum_loss/self.n_train, sum_accuracy/self.n_train)

			# evalation
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_test, batchsize):
				x_batch = self.xp.asarray(self.x_test[i:i+batchsize])
				y_batch = self.xp.asarray(self.y_test[i:i+batchsize])

				real_batchsize = len(x_batch)

				loss, acc = self.forward(x_batch, y_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
				sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

			print 'test mean loss={}, accuracy={}'.format(sum_loss/self.n_test, sum_accuracy/self.n_test)

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

	data_train,\
	data_test,\
	target_train,\
	target_test = train_test_split(mnist.data, mnist.target)

	data = [data_train, data_test]
	target = [target_train, target_test]


	start_time = time.time()

	if args.gpu >= 0:
		cuda.check_cuda_available()
		cuda.get_device(args.gpu).use()

	MLP = MLP(data=data, target=target, gpu=args.gpu)
	MLP.train_and_test(n_epoch=1)

	end_time = time.time()

	print "time = {} min".format((end_time-start_time)/60.0)











