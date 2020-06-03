import tensorflow as tf
from utils import *

class SRresnet:


	def __init__(self,training, learning_rate=1e-4, n_blocks=16):
		""" Init class attr """
		self.learning_rate = learning_rate
		self.training = training
		self.n_blocks = n_blocks


	# def ResidualBlock(self, x, kernel_size, filter_size):
	# 	"""Residual block a la ResNet"""
	# 	# with tf.variable_scope('sr_edge_net') as scope:		
	# 	weights = {
	# 		'w1':tf.get_variable(name='w1_redidual',\
	# 			shape=[kernel_size, kernel_size, filter_size, filter_size], dtype=tf.float32,\
	# 			initializer=tf.glorot_normal_initializer()),
	# 		'w2':tf.get_variable(name='w2_residual',\
	# 			shape=[kernel_size, kernel_size, filter_size, filter_size], dtype=tf.float32,\
	# 			initializer=tf.glorot_normal_initializer()),
	# 	}

	# 	skip = x
	# 	x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
	# 	x = tf.layers.batch_normalization(x, training=self.training)
	# 	x = tf.nn.relu(x)
	# 	x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
	# 	x = tf.nn.relu(x)
	# 	x = tf.layers.batch_normalization(x, training=self.training)

	# 	x = x + skip
	# 	return x

	# def Upsample2xBlock(self, x, kernel_size, filter_size):
	# 	weights = {
	# 		'w1':tf.get_variable(name='w1_upsample',\
	# 			shape=[kernel_size, kernel_size, 64, filter_size], dtype=tf.float32,\
	# 			initializer=tf.glorot_normal_initializer()),
	# 	}
	# 	"""Upsample 2x via SubpixelConv"""
	# 	print('init',x)
	# 	x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
	# 	print('before',x)
	# 	x = tf.depth_to_space(x, 2)
	# 	print('after',x)
		
	# 	x = tf.nn.relu(x)
	# 	return x


	# def forward(self, x):
	# 	with tf.variable_scope('sr_edge_net',reuse=tf.AUTO_REUSE) as scope:

	# 		weights = {
	# 			'w_in':tf.get_variable(name='w_in', shape=[9, 9, 3, 64], dtype=tf.float32,\
	# 				initializer=tf.glorot_normal_initializer()),
	# 			'w1':tf.get_variable(name='w1', shape=[3, 3, 64, 64], dtype=tf.float32,\
	# 				initializer=tf.glorot_normal_initializer()),
	# 			'w_out':tf.get_variable(name='w_out', shape=[9, 9, 64, 3], dtype=tf.float32,\
	# 				initializer=tf.glorot_normal_initializer()),
	# 		}

	# 		# print(x_concate)
	# 		x = tf.nn.conv2d(x, weights['w_in'], strides=[1,1,1,1], padding='SAME')
	# 		x = tf.nn.relu(x, name='layer_input')
	# 		skip = x
			
	# 		for i in range(self.n_blocks):
	# 			x = self.ResidualBlock(x, 3, 64)

	# 		x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME', name='layer_1')
	# 		x = tf.layers.batch_normalization(x, training=self.training)
	# 		x = x + skip

	# 		for i in range(2):
	# 			x = self.Upsample2xBlock(x, kernel_size=3, filter_size=256)
			
	# 		x = tf.nn.conv2d(x, weights['w_out'], strides=[1,1,1,1], padding='SAME', name='y_predict')
			
	# 		print(x)
	# 		return x

	def ResidualBlock(self, x, kernel_size, filters, strides=1):
		"""Residual block a la ResNet"""
		skip = x
		x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',\
			use_bias=False)
		x = tf.layers.batch_normalization(x, training=self.training)
		x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
		x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',\
			use_bias=False)

		x = tf.layers.batch_normalization(x, training=self.training)
		x = x + skip
		return x


	def Upsample2xBlock(self, x, kernel_size, filters, strides=1):
		"""Upsample 2x via SubpixelConv"""
		x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
		x = tf.depth_to_space(x, 2)
		x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
		return x

	def forward(self, x):
		"""Builds the forward pass network graph"""
		with tf.variable_scope('sr_edge_net') as scope:
			x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
			x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
			skip = x

			# B x ResidualBlocks
			for i in range(self.n_blocks):
				x = self.ResidualBlock(x, kernel_size=3, filters=64, strides=1)

			x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
			x = tf.layers.batch_normalization(x, training=self.training)
			x = x + skip

			# Upsample blocks
			for i in range(2):
				x = self.Upsample2xBlock(x, kernel_size=3, filters=256)

			x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
		return x


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
			return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, \
				var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sr_edge_net'))