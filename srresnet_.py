import tensorflow as tf
from utils import *

class SRresnet:


	def __init__(self,training, learning_rate=1e-4, n_blocks=16):
		""" Init class attr """
		self.learning_rate = learning_rate
		self.training = training
		self.n_blocks = n_blocks


	def ResidualBlock(self, x, kernel_size, filter_size):
	    """Residual block a la ResNet"""

	    weights = {
			'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size], stddev=1e-3), name='w1'),
			'w2': tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size], stddev=1e-3), name='w2'),
	            # 'w3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim], stddev=1e-3), name='w3')
	    }

	    biases = {
	    	'b1':tf.Variable(tf.zeros([64],name='b1')),
	    	'b2':tf.Variable(tf.zeros([64],name='b2')),
	    }

	    skip = x
	    x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
	    x = tf.layers.batch_normalization(x)
	    x = tf.nn.relu(x)
	    x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
	    x = tf.nn.relu(x)
	    x = tf.layers.batch_normalization(x)

	    x = x + skip
	    return x

	def Upsample2xBlock(self, x, kernel_size, filter_size):
		weights = {
		    'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size, 64, filter_size], stddev=1e-3), name='w1'),
		}
		"""Upsample 2x via SubpixelConv"""
		print('init',x)
		x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
		print('before',x)
		x = tf.depth_to_space(x, 2)
		print('after',x)
		
		x = tf.nn.relu(x)
		return x


	def foward(self, x):
		with tf.variable_scope('sr_edge_net') as scope:

			weights ={
				'w_in': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3), name='w_in'),
		    	'w1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-3), name='w1'),
		    	'w_out': tf.Variable(tf.random_normal([9, 9, 64, 3], stddev=1e-3), name='w_in')
			}

			biases = {
				'b_in': tf.Variable(tf.zeros([64],name='b_in')),
				'b1': tf.Variable(tf.zeros([64],name='b1')),
			}

			# print(x_concate)
			x = tf.nn.conv2d(x, weights['w_in'], strides=[1,1,1,1], padding='SAME', name='x_input')
			x = tf.nn.relu(x, name='x_input')
			skip = x
			
			for i in range(self.n_blocks):
				x = self.ResidualBlock(x, 3, 64)

			x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME', name='w_in_residual_out')
			x = tf.layers.batch_normalization(x, training=True)
			x = x + skip

			for i in range(2):
				x = self.Upsample2xBlock(x, kernel_size=3, filter_size=256)
			
			x = tf.nn.conv2d(x, weights['w_out'], strides=[1,1,1,1], padding='SAME', name='w_in_residual_out')
			
			print(x)
			return x

	# def optimizer(self, loss):
	# 	return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

	def resnetLoss(self, y, y_pred):
		return tf.reduce_mean(tf.square(y - y_pred))

	def gradientLoss(self, y, y_pred):
		# y = cany_oper_batch(y)
		y = tf.image.sobel_edges(y)
		y_pred = tf.image.sobel_edges(y_pred)

		return tf.reduce_mean(tf.square(y - y_pred))

	def totalLoss(self, resnetLoss, gradientLoss):
		return resnetLoss + gradientLoss


	def optimizer(self, loss):
		# tf.control_dependencies([discrim_train
		# update_ops needs to be here for batch normalization to work
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='sr_edge_net')
		with tf.control_dependencies(update_ops):
			return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope='sr_edge_net'))