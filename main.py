

from srresnet_ import SRresnet
import tensorflow as tf
import argparse
from utils import *
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm,trange
from benchmark import Benchmark

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
	parser.add_argument('--log-path', type=str, default='results/' help='Checkpoint to load all weights from.')
	parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size.')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
	parser.add_argument('--train-dir', type=str, help='Directory containing training images')
	parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
	parser.add_argument('--epoch', type=int, default='100', help='How many iterations ')
	parser.add_argument('--log-freq', type=int, default=1000, help='How many training iterations between validation/checkpoints.')
	parser.add_argument('--block-n', type=int, default=3, help='How many recurenet blocks')
	parser.add_argument('--is-val', action='store_true', help='True for evaluate image')
	parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')

	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	args = parser.parse_args()

	"""
	Testing Variable 
	"""
	hr_ = tf.placeholder(tf.float32, [None, None, None, 3], name='HR_image')
	lr_ = tf.placeholder(tf.float32, [None, None, None, 3], name='LR_image')
	# hr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='HR_edge') 
	# lr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='LR_edge') 

	training_net = tf.placeholder(tf.bool, name='training_net')

	""" DEBUG placeholder parmaters """

	sr_resnet = SRresnet(training=training_net, learning_rate = args.learning_rate)
	y_pred = sr_resnet.foward(lr_)

	resnet_loss = sr_resnet.resnetLoss(hr_, y_pred)
	gradient_loss = sr_resnet.gradientLoss(hr_, y_pred)

	total_loss = sr_resnet.totalLoss(resnet_loss, gradient_loss)

	resnet_opt = sr_resnet.optimizer(total_loss)

	benchmarks = [
		Benchmark('Benchmarks/Set5', name='Set5'),
		Benchmark('Benchmarks/Set14', name='Set14'),
		Benchmark('Benchmarks/BSD100', name='BSD100')
	]

    #Set up h5 dataset path
	train_data_path = 'done_dataset\PreprocessedData.h5'
	val_data_path = 'done_dataset\PreprocessedData_val.h5'
	eval_data_path = 'done_dataset\PreprocessedData_eval.h5'

	"""Train session"""
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		# print(get_batch_folder_list(t_path))
		# print(iter(get_batch_folder_list(t_path)))
		iterator = 0
		saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='sr_edge_net'), max_to_keep=10)

		if args.load:
			# iterator = int(args.load.split('-')[-1])
			# saver.restore(sess, args.load)
			iterator ,sess = load(sess,saver, args.load)
			iterator = int(iterator.split('-')[1])

		train_data_set = get_data_set(train_data_path,'train')
		val_data_set = get_data_set(val_data_path,'val')
		eval_data_set = get_data_set(eval_data_path,'eval')

		val_error_li =[]
		eval_error_li =[]

		if args.is_val:
			""" To validate Benchmarks"""

			for benchmark in benchmarks:
				psnr, ssim, _, _ = benchmark.eval(sess, y_pred, log_path=args.log_path, iteration=iterator)
				print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')

		else:
			""" To Training """

			for epoch in range(args.epoch):
				t =trange(0, len(train_data_set) - args.batch_size + 1, args.batch_size,\
					bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',\
					desc='Iterations')

				for batch_idx in t:
					batch_hr = train_data_set[batch_idx:batch_idx + 16]
					batch_lr = downsample_batch(batch_hr, factor=4)
					batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
					_, err = sess.run([resnet_opt,resnet_loss], feed_dict={training_net: True, lr_: batch_lr, hr_: batch_hr})
					
					t.set_description("[Eopch: %s][Iter: %s][Error: %.4f]" %(epoch, iterator, err))


					""" Log Eval and Val status"""
					if iterator%args.log_freq == 0:
						for batch_idx in range(0, len(val_data_set) - args.batch_size + 1, args.batch_size): 
						# Test every log-freq iterations
							val_error = evaluate_model(total_loss, val_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
							eval_error = evaluate_model(total_loss, eval_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
						
						val_error_li.append(val_error)
						eval_error_li.append(eval_error)

						log_line = ''
						for benchmark in benchmarks:
							psnr,ssim,_,_ = benchmark.evaluate(sess,y_pred,log_path='results',iterator)
							print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
							log_line += ',%.7f, %.7f' % (psnr, ssim)
						record_log(iterator, val_error, eval_error, args.log_path, iterator)
						save(sess,saver,'checkpoint',iterator)
					""""""

					""" Iterator """	
					iterator += 1

				# print(err)

if __name__=='__main__':

	main()