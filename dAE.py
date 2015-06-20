# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F


N = 60000
N_test = 10000


class DenoisingAutoEncoder:
	def __init__(self, data, n_inputs=784, n_hidden=784, noise=0.1):

		self.model = FunctionSet(encoder=F.Linear(n_inputs, n_hidden),
								 decoder=F.Linear(n_hidden, n_inputs))
		self.x_train, self.x_test = np.split(data, [N])
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model.collect_parameters())
		self.noise = noise
		self.rng = np.random.RandomState(1)

	def forward(self, x_data, train=True):
		y_data = x_data
		# add noise (masking noise)
		x_data = self.add_noise(x_data, train=train)

		x, t = Variable(x_data), Variable(y_data)
		# encode
		h = self.encode(x)
		# decode
		y = self.decode(h)
		# compute loss
		loss = F.mean_squared_error(y, t)
		return loss

	def predict(self, x_data):
		x = Variable(x_data)
		# encode
		h = self.encode(x)
		# decode
		y = self.decode(h)
		return y.data

	def encode(self, x):
		return F.sigmoid(self.model.encoder(x))

	def decode(self, h):
		return F.sigmoid(self.model.decoder(h))

	def encoder(self):
		return self.model.encoder

	def decoder(self):
		return self.model.decoder

	# masking noise
	def add_noise(self, x_data, train=True):
		if train:
			return self.rng.binomial(size=x_data.shape, n=1, p=1.0-self.noise) * x_data
		else:
			return x_data


	def train_and_test(self, n_epoch=5, batchsize = 100):
		for epoch in xrange(1, n_epoch+1):
			print 'epoch', epoch

			perm = self.rng.permutation(N)
			sum_loss = 0
			for i in xrange(0, N, batchsize):
				x_batch = self.x_train[perm[i:i+batchsize]]

				self.optimizer.zero_grads()
				loss = self.forward(x_batch)
				loss.backward()
				self.optimizer.update()

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

			print 'train mean loss={}'.format(sum_loss/N)

			# evalation
			sum_loss = 0
			for i in xrange(0, N_test, batchsize):
				x_batch = self.x_test[i:i+batchsize]

				loss = self.forward(x_batch, train=False)

				sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

			print 'test mean loss={}'.format(sum_loss/N_test)

def draw_digit(data):
    size = 28
    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.flag()
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

def draw_digits(data, fname="fig.png"):
	for i in xrange(3*3):
		plt.subplot(331+i)
		draw_digit(data[i])
	plt.savefig(fname)



if __name__ == '__main__':
	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)
	# draw_digits(mnist.data[0:9])

	dAE = DenoisingAutoEncoder(data=mnist.data)

	perm = np.random.permutation(N)
	data = mnist.data[perm[0:9]]

	draw_digits(data, fname="epoch0.png")

	dAE.train_and_test(n_epoch=1)

	draw_digits(dAE.predict(data), fname="epoch1.png")

	dAE.train_and_test(n_epoch=1)

	draw_digits(dAE.predict(data), fname="epoch2.png")

	dAE.train_and_test(n_epoch=1)

	draw_digits(dAE.predict(data), fname="epoch3.png")

	dAE.train_and_test(n_epoch=1)

	draw_digits(dAE.predict(data), fname="epoch4.png")

	dAE.train_and_test(n_epoch=1)

	draw_digits(dAE.predict(data), fname="epoch5.png")