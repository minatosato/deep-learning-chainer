# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

class ConvolutionalNN:
	def __init__(self, data, target, n_hidden=100, n_outputs=10, gpu=-1):

		self.model = FunctionSet(conv1=	F.Convolution2D(1, 8, 5),
								 conv2=	F.Convolution2D(8, 16, 5),
								 l3=	F.Linear(144, n_hidden),
								 l4=	F.Linear(n_hidden, n_outputs))
		if gpu >= 0:
			self.model.to_gpu()

		self.gpu = gpu

		self.x_train,\
		self.x_test,\
		self.y_train,\
		self.y_test = train_test_split(data, target, test_size=0.1)

		self.n_train = len(self.y_train)
		self.n_test = len(self.y_test)

		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model.collect_parameters())

	def forward(self, x_data, y_data, train=True):

		if self.gpu >= 0:
			x_data = cuda.to_gpu(x_data)
			y_data = cuda.to_gpu(y_data)

		x, t = Variable(x_data), Variable(y_data)
		h = F.max_pooling_2d(F.relu(self.model.conv1(x)), ksize=2, stride=2)
		h = F.max_pooling_2d(F.relu(self.model.conv2(h)), ksize=3, stride=3)
		h = F.dropout(F.relu(self.model.l3(h)), train=train)
		y = self.model.l4(h)
		return F.softmax_cross_entropy(y, t), F.accuracy(y,t)


	def train_and_test(self, n_epoch=20, batchsize=100):
		for epoch in xrange(1, n_epoch+1):
			print 'epoch', epoch

			perm = np.random.permutation(self.n_train)
			sum_accuracy = 0
			sum_loss = 0
			for i in xrange(0, self.n_train, batchsize):
				x_batch = self.x_train[perm[i:i+batchsize]]
				y_batch = self.y_train[perm[i:i+batchsize]]

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
				x_batch = self.x_test[i:i+batchsize]
				y_batch = self.y_test[i:i+batchsize]

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
	mnist.data = mnist.data.reshape(70000,1,28,28)
	mnist.target = mnist.target.astype(np.int32)

	if args.gpu >= 0:
		cuda.init(args.gpu)

	start_time = time.time()

	CNN = ConvolutionalNN(data=mnist.data, target=mnist.target, gpu=args.gpu)
	CNN.train_and_test()


	end_time = time.time()

	print "time = {} min".format((end_time-start_time)/60.0)









