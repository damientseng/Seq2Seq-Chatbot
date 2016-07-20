import theano
import theano.tensor as tensor
import numpy as np
import cPickle as pickle 
import timeit

from utils import *
from seq2seq import Seq2Seq
from chatbot import Chatbot 

import logging
logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
	datefmt='%a, %d %b %Y %H:%M:%S',
	filename='trainingprocess.log',
	filemode='w')


def build_model(
	voca_size = 29331,
	hidden_size = 512,
	lstm_layers_num = 2,
	batch_size = 10,
	max_epochs = 2,
	retrain = False
	):
	"""
	train a model with mini-batch gradient descend
	"""
	model = None

	if not retrain:
		try:
			f = open(path_pkl)
			model = pickle.load(f)
			return model
		except Exception:
			print "Model does not pre-exist..."

	print "Will train a new model..."

	encoderInputs, decoderInputs, decoderTarget = load_train_data(path_train, 20000)
	#(sent_size, example_num)
	num_batchs = encoderInputs.shape[1] // batch_size
	model = Seq2Seq(voca_size, hidden_size, lstm_layers_num, learning_rate=0.1)

	batch_idx = 0
	for ep in xrange(max_epochs):
		enIpt = encoderInputs[:, batch_idx*batch_size:(batch_idx+1)*batch_size]
		deIpt = decoderInputs[:, batch_idx*batch_size:(batch_idx+1)*batch_size]
		deTgt = decoderTarget[:, batch_idx*batch_size:(batch_idx+1)*batch_size]

		enMsk = get_mask(enIpt)
		deMsk = get_mask(deIpt)
		loss, costs = model.train(enIpt, enMsk, deIpt, deMsk, deTgt)

		if ep%20 == 0:
			print "in epoch %d/%d..."%(ep, max_epochs)
		if batch_idx == 0:
			ot = "in epoch %d/%d..."%(ep, max_epochs) + "	loss:	"+str(loss)
			print ot
			logging.info(ot)

		batch_idx = (batch_idx+1) % num_batchs
	"""
	with open(path_pkl, "wb") as mf:
		pickle.dump(model, mf)
	"""
	return model


if __name__ == '__main__':
	print "Initializing chatbot..."
	time_start = timeit.default_timer()

	model = build_model(retrain=True)
	cbot = Chatbot(model)

	time_end = timeit.default_timer()
	print "Done initializing chatbot...Time taken:   ", (time_end-time_start)

	
	while True:
		conva = raw_input("role A: ")
		convB = cbot.utter(conva)
		print "role B: ", convB, "\n"




