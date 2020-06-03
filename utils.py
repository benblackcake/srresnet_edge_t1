
import numpy as np
import cv2
import glob
import os
import tensorflow as tf
import h5py


def cany_oper(image):
	"""Using cany operator to get image edge map"""
	kernel_size = 3
	low_threshold = 1
	high_threshold = 10

	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size), 0)

	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	return edges

def cany_oper_batch(batch):
	# print(batch.shape) 
	# if batch.shape[1]%2 !=t 0 || batch.shape[2]%2 != 0:
	canyed = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2]))
	for i in range(batch.shape[0]):
		canyed[i, :, :] = cany_oper(batch[i, :, :, :])
	return canyed

def preprocess(lr, hr):
    """Preprocess lr and hr batch"""
    lr = lr / 255.0
    hr = (hr / 255.0) * 2.0 - 1.0
    return lr, hr

def downsample(image, factor):
	"""Using bicubic interpolation to get downsample image"""
	bicbuic_img = cv2.resize(image,None,fx = 1.0/factor ,fy = 1.0/factor, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
	# print(bicbuic_img.shape)

	return bicbuic_img

    
def downsample_batch(batch, factor):
    downsampled = np.zeros((batch.shape[0], batch.shape[1] // factor, batch.shape[2] // factor, 3))
    for i in range(batch.shape[0]):
        downsampled[i, :, :, :] = downsample(batch[i, :, :, :], factor)
    return downsampled

def get_batch_folder_list(folder_path):
    train_filenames = np.array(glob.glob(os.path.join(folder_path, '*'), recursive=True))
    return train_filenames

def get_batch_image(batch_folder_path):
	image_li = list()
	train_file = np.array(glob.glob(os.path.join(batch_folder_path,'*'), recursive=True))

	for image in train_file:
		image = modcrop(cv2.imread(image,cv2.IMREAD_UNCHANGED))
		# print(image.shape)
		image_li.append(image)
	# print(train_file)
	return image_li


def modcrop(img, scale =3):
	"""
	To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
	"""
	# Check the image is gray

	if len(img.shape) ==3:
		h, w, _ = img.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		img = img[0:h, 0:w, :]
	else:
		h, w = img.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		img = img[0:h, 0:w]
	return img


def get_data_set(path,label):
	data_set = h5py.File(path, 'r')
	return data_set[label]


def load(sess,saver,checkpoint_dir):
	"""
	To load the checkpoint use to test or pretrain
	"""
	print("\nReading Checkpoints.....\n\n")
	# model_dir = "%s" % ("srEdge")# give the model name by label_size
	# checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

	# Check the checkpoint is exist 
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
		saver.restore(sess, os.path.join(os.getcwd(), ckpt_path))
		print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
		return ckpt_path,sess
	else:
		print("\n! Checkpoint Loading Failed \n\n")
		raise
def save(sess,saver,checkpoint_dir, step):
	"""
	To save the checkpoint use to test or pretrain
	"""
	model_name = "weights"
	model_dir = "%s" % ("srEdge")
	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step, write_meta_graph=False)


def record_log(iteration, val_error, eval_error, log_path, log_line):
	with open(log_path+'/recorded.csv', 'a') as f:
		f.write('%d, %.15f, %.15f %s\n' % (iteration,val_error,eval_error,log_line))


def evaluate_model(loss_function, get_batch, sess, num_images, batch_size):
	"""Tests the model over all num_images using input tensor get_batch"""
	loss = 0
	total = 0
	for i in range(int(math.ceil(num_images / batch_size))):
		batch_hr = get_batch
		batch_lr = downsample_batch(batch_hr, factor=4)
		batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
	loss += sess.run(loss_function,\
		feed_dict={'training_net:0': False,'LR_image:0': batch_lr, 'HR_image:0': batch_hr})
	total += 1
	loss = loss / total
	return loss