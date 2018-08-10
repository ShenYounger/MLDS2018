# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

#for reproduction of results, should fix seed in random module
from numpy.random import seed
seed(20)
from tensorflow import set_random_seed
set_random_seed(110)

learning_rate = 0.001
point_num = 200 
epoches = 20000
epoches = 20

#sin(5*pi*x)/(5*pi*x)
def func1():
	x = np.linspace(1e-8, 1, point_num)
	y = 5 * np.pi * x
	z = np.sin(y)/y
	return (x, z)

#sgn(sin(5*pi*x))
def func2():
	x = np.linspace(0, 1, point_num)
	y = np.sin(5 * np.pi * x)
	z = np.sign(y)
	return (x, z)

#choose he initialization method
def weight_variable(shape):
	node_in, node_out = shape
	W = tf.Variable(np.random.randn(node_in, node_out).astype('float32')) / np.sqrt((node_in+1.0)/2.0)
	return W
#all biases are initialized zero
def bias_variable(shape):
	W = tf.Variable(np.zeros(shape).astype('float32'))
	return W

def simu1(input, output, epoches):
	print("simulation 1")

	x = tf.placeholder(tf.float32, [None, 1])
	W1 = weight_variable((1,190))
	b1 = bias_variable(190)

	z1 = tf.matmul(x, W1) + b1
	z1 = tf.nn.relu(z1, name="first_layer")

	W2 = weight_variable((190,1))
	b2 = bias_variable(1)

	z2 = tf.matmul(z1, W2) + b2

	y_ = tf.placeholder(tf.float32, [None, 1])

	mse = tf.reduce_mean(tf.square(z2 - y_))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)
	loss_log = []
	for i in range(epoches):
		_, loss = sess.run([train_step, mse], feed_dict={x: input, y_: output})
		print("simu1 loss is ", loss)
		loss_log.append(loss)

	z2_final = sess.run(z2, feed_dict={x: input})
	return (loss_log, z2_final[:, 0])
		
def simu2(input, output, epoches):
	print("simulation 2")

	x = tf.placeholder(tf.float32, [None, 1])
	W1 = weight_variable((1,10))
	b1 = bias_variable(10)

	z1 = tf.matmul(x, W1) + b1
	z1 = tf.nn.relu(z1, name="first_layer")

	W2 = weight_variable((10,18))
	b2 = bias_variable(18)

	z2 = tf.matmul(z1, W2) + b2
	z2 = tf.nn.relu(z2, name="second_layer")

	W3 = weight_variable((18,15))
	b3 = bias_variable(15)

	z3 = tf.matmul(z2, W3) + b3
	z3 = tf.nn.relu(z3, name="third_layer")

	W4 = weight_variable((15,4))
	b4 = bias_variable(4)

	z4 = tf.matmul(z3, W4) + b4
	z4 = tf.nn.relu(z4, name="fourth_layer")

	W5 = weight_variable((4,1))
	b5 = bias_variable(1)

	z5 = tf.matmul(z4, W5) + b5

	y_ = tf.placeholder(tf.float32, [None, 1])

	mse = tf.reduce_mean(tf.square(z5 - y_))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)


	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)
	loss_log = []
	for i in range(epoches):
		_, loss = sess.run([train_step, mse], feed_dict={x: input, y_: output})
		print("simu2 loss is ", loss)
		loss_log.append(loss)

	z5_final = sess.run(z5, feed_dict={x: input})
	return (loss_log, z5_final[:, 0])

def simu3(input, output, epoches):
	print("simulation 3")

	x = tf.placeholder(tf.float32, [None, 1])
	W1 = weight_variable((1,5))
	b1 = bias_variable(5)

	z1 = tf.matmul(x, W1) + b1
	z1 = tf.nn.relu(z1, name="first_layer")

	W2 = weight_variable((5,10))
	b2 = bias_variable(10)

	z2 = tf.matmul(z1, W2) + b2
	z2 = tf.nn.relu(z2, name="second_layer")

	W3 = weight_variable((10,10))
	b3 = bias_variable(10)

	z3 = tf.matmul(z2, W3) + b3
	z3 = tf.nn.relu(z3, name="third_layer")

	W4 = weight_variable((10,10))
	b4 = bias_variable(10)

	z4 = tf.matmul(z3, W4) + b4
	z4 = tf.nn.relu(z4, name="fourth_layer")

	W5 = weight_variable((10,10))
	b5 = bias_variable(10)

	z5 = tf.matmul(z4, W5) + b5
	z5 = tf.nn.relu(z5, name="fifth_layer")

	W6 = weight_variable((10,10))
	b6 = bias_variable(10)

	z6 = tf.matmul(z5, W6) + b6
	z6 = tf.nn.relu(z6, name="sixth_layer")

	W7 = weight_variable((10,5))
	b7 = bias_variable(5)

	z7 = tf.matmul(z6, W7) + b7
	z7 = tf.nn.relu(z7, name="seventh_layer")

	W8 = weight_variable((5,1))
	b8 = bias_variable(1)

	z8 = tf.matmul(z7, W8) + b8

	y_ = tf.placeholder(tf.float32, [None, 1])

	mse = tf.reduce_mean(tf.square(z8 - y_))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)


	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)
	loss_log = []
	for i in range(epoches):
		_, loss = sess.run([train_step, mse], feed_dict={x: input, y_: output})
		print("simu3 loss is ", loss)
		loss_log.append(loss)

	z8_final = sess.run(z8, feed_dict={x: input})
	return (loss_log, z8_final[:, 0])

def make_figure(file_name):
	fr = open(file_name,'rb')

	x = pickle.load(fr)
	loss1 = pickle.load(fr)
	loss2 = pickle.load(fr)
	loss3 = pickle.load(fr)

	
	input = pickle.load(fr)
	output = pickle.load(fr)
	simu1_res = pickle.load(fr)
	simu2_res = pickle.load(fr)
	simu3_res = pickle.load(fr)

	fr.close()

	# plot loss figure
	if file_name == 'simu_func1':
		plt.figure(num = 1)
		plt.ylim((1e-6, 1))
	else:
		plt.figure(num = 3)
		plt.ylim((2*1e-3, 1.2))

	plt.yscale('log')
	plt.xlim((-500, 20500))	
	new_ticks = np.linspace(0, 20000, 9)
	plt.xticks(new_ticks)
	plt.xlabel('epoch')
	plt.ylabel('loss')	
	plt.title('model loss')
	plt.plot(x, loss1, color='blue', linewidth=2.0, linestyle='-', label='model0_loss')
	plt.plot(x, loss2, color='green', linewidth=2.0, linestyle='-', label='model1_loss')
	plt.plot(x, loss3, color='red', linewidth=2.0, linestyle='-', label='model2_loss')

	ax = plt.gca() #get axes
	ax.spines['right'].set_visible(False) #right frame is not visible
	ax.spines['top'].set_visible(False) #top frame is not visible
	ax.xaxis.set_ticks_position('bottom') #bind xaxis to bottom frame
	ax.yaxis.set_ticks_position('left') #bind yaxis to left frame

	plt.tick_params(top='off', bottom='on', left='on', right='off') #turn off ticks on right and top frame
	plt.legend(loc='best')
	save_file_name = file_name + ".loss.png"
	plt.savefig(save_file_name)
	plt.show()

	#plot simu figure
	if file_name == 'simu_func1':
		plt.figure(num = 2)
		plt.ylim((-0.3, 1.1))
		func = r'$\frac{\sin(5\pi x)}{5\pi x}$'
	else:
		plt.figure(num = 4)
		plt.ylim((-1.25, 1.5))
		func = r'$\operatorname{sgn}(\sin(5\pi x))$'

	plt.xlim((-0.05, 1.05))	
	new_ticks = np.linspace(0.0, 1.0, 6)
	plt.xticks(new_ticks)

	plt.plot(input, output, color='magenta', linewidth=2.0, linestyle='-', label=func)
	plt.plot(input, simu1_res, color='blue', linewidth=2.0, linestyle='-', label='model0')
	plt.plot(input, simu2_res, color='green', linewidth=2.0, linestyle='-', label='model1')
	plt.plot(input, simu3_res, color='red', linewidth=2.0, linestyle='-', label='model2')

	ax = plt.gca() #get axes
	ax.spines['right'].set_visible(False) #right frame is not visible
	ax.spines['top'].set_visible(False) #top frame is not visible
	ax.xaxis.set_ticks_position('bottom') #bind xaxis to bottom frame
	ax.yaxis.set_ticks_position('left') #bind yaxis to left frame

	plt.tick_params(top='off', bottom='on', left='on', right='off') #turn off ticks on right and top frame
	plt.legend(loc='best')
	save_file_name = file_name + ".simu.png"
	plt.savefig(save_file_name)
	plt.show()
	

def simu(func, file_name):
	data = func()
	input, output = data
	input = input[:, None]	
	output = output[:, None]	

	x = range(epoches)	

	(loss_simu1, simu1_res) = simu1(input, output, epoches)
	(loss_simu2, simu2_res)  = simu2(input, output, epoches)
	(loss_simu3, simu3_res)  =  simu3(input, output, epoches)

	## store data to file start
	f = open(file_name,'wb')
	
	# store loss
	pickle.dump(x, f)
	pickle.dump(loss_simu1, f)
	pickle.dump(loss_simu2, f)
	pickle.dump(loss_simu3, f)

	# store simulation result
	pickle.dump(input, f)
	pickle.dump(output, f)
	pickle.dump(simu1_res, f)
	pickle.dump(simu2_res, f)
	pickle.dump(simu3_res, f)

	f.close()
	## store data to file end
	
if __name__ == "__main__":

	print("simu data generated by func1()")
	simu(func1, "simu_func1")
	make_figure("simu_func1")

	print("simu data generated by func2()")
	simu(func2, "simu_func2")
	make_figure("simu_func2")

	
		
	
	

