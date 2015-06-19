# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F


N = 60000
N_test = 10000


class dAE:
	def __init__(self, data, target, n_inputs=784, n_hidden=1000, n_outputs=10):

		self.model = FunctionSet(l1=F.Linear(n_inputs, n_hidden),
								 l2=F.Linear(n_hidden, n_outputs))
		self.x_train, self.x_test = np.split(data,   [N])
		self.y_train, self.y_test = np.split(target, [N])
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model.collect_parameters())

	def forward(self, x_data, y_data, train=True):
		x, t = Variable(x_data), Variable(y_data)
		h1 = F.dropout(F.sigmoid(self.model.l1(x)), train=train)
		y = self.model.l2(h1)
		return F.softmax_cross_entropy(y, t)


	def train_and_test(self, n_epoch=20, batchsize = 100):
		for epoch in xrange(1, n_epoch+1):
			print 'epoch', epoch

			perm = np.random.permutation(N)
			sum_loss = 0
			for i in xrange(0, N, batchsize):
				x_batch = self.x_train[perm[i:i+batchsize]]
				y_batch = self.y_train[perm[i:i+batchsize]]

				self.optimizer.zero_grads()
				loss, acc = self.forward(x_batch, y_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

			print 'train mean loss={}, accuracy={}'.format(sum_loss/N, sum_accuracy/N)

			# evalation
			sum_loss = 0
			for i in xrange(0, N_test, batchsize):
				x_batch = self.x_test[i:i+batchsize]
				y_batch = self.y_test[i:i+batchsize]

				loss, acc = self.forward(x_batch, y_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

			print 'test mean loss={}, accuracy={}'.format(sum_loss/N_test, sum_accuracy/N_test)

if __name__ == '__main__':
	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)

	dae = dAE(data=mnist.data, target=mnist.target)
	dae.train_and_test()