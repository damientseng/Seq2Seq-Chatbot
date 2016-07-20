import theano
import theano.tensor as tensor
import numpy as np
import cPickle as pickle 
import timeit

import utils

class LSTM(object):
	def __init__(self, hidden_size):

		self.hidden_size = hidden_size

		# lstm W matrixes, Wf, Wi, Wo, Wc respectively, all config.floatX type
		self.W = theano.shared(name="W", value=utils.init_norm(self.hidden_size, 4*self.hidden_size), borrow=True)
		# lstm U matrixes, Uf, Ui, Uo, Uc respectively, all config.floatX type
		self.U = theano.shared(name="U", value=utils.init_norm(self.hidden_size, 4*self.hidden_size), borrow=True)
		# lstm b vectors, bf, bi, bo, bc respectively, all config.floatX type
		self.b = theano.shared(name="b", value=np.zeros( 4*self.hidden_size, dtype=theano.config.floatX ), borrow=True)

		self.params = [self.W, self.U, self.b]


	def forward(self, inputs, mask, h0=None, C0=None):
		"""
		param inputs: #(max_sent_size, batch_size, hidden_size).
		inputs: state_below
		"""
		if inputs.ndim == 3:
			batch_size = inputs.shape[1]
		else:
			batch_size = 1

		if h0 == None:
			h0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
		if C0 == None:
			C0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

		def _step( m, X,   h_, C_,   W, U, b):
			XW = tensor.dot(X, W)    #(batch_size, 4*hidden_size)
			h_U = tensor.dot(h_, U)  #(batch_size, 4*hidden_size)
			#before activation,       (batch_size, 4*hidden_size)
			bfr_actv = XW + h_U + b
			
			f = tensor.nnet.sigmoid( bfr_actv[:, 0:self.hidden_size] )                     #forget gate (batch_size, hidden_size)
			i = tensor.nnet.sigmoid( bfr_actv[:, 1*self.hidden_size:2*self.hidden_size] )   #input gate (batch_size, hidden_size)
			o = tensor.nnet.sigmoid( bfr_actv[:, 2*self.hidden_size:3*self.hidden_size] ) #output  gate (batch_size, hidden_size)
			Cp = tensor.tanh( bfr_actv[:, 3*self.hidden_size:4*self.hidden_size] )        #candi states (batch_size, hidden_size)

			C = i*Cp + f*C_
			C = m[:, None]*C + (1.0 - m)[:, None]*C_

			h = o*tensor.tanh( C ) 
			h = m[:, None]*h + (1.0 - m)[:, None]*h_

			h, C = tensor.cast(h, theano.config.floatX), tensor.cast(h, theano.config.floatX)
			return h, C

		outputs, updates = theano.scan(
			fn = _step,
			sequences = [mask, inputs],
			outputs_info = [h0, C0],
			non_sequences = [self.W, self.U, self.b]
			)

		hs, Cs = outputs
		return hs, Cs


class Seq2Seq(object):
	def __init__(self, voca_size, hidden_size, lstm_layers_num, learning_rate=0.2):
		self.voca_size = voca_size
		self.hidden_size = hidden_size
		self.lstm_layers_num = lstm_layers_num
		self.learning_rate = learning_rate
		self._train = None
		self._utter = None
		self.params = []
		self.encoder_lstm_layers = []
		self.decoder_lstm_layers = []
		self.hos = []
		self.Cos = []

		encoderInputs, encoderMask = tensor.imatrices(2)
		decoderInputs, decoderMask, decoderTarget = tensor.imatrices(3)

		self.lookuptable = theano.shared(
			name="Encoder LookUpTable",
			value=utils.init_norm(self.voca_size, self.hidden_size),
			borrow=True
			)
		self.linear = theano.shared(
			name="Linear",
			value=utils.init_norm(self.hidden_size, self.voca_size),
			borrow=True
			)
		self.params += [self.lookuptable, self.linear]    #concatenate
		
		#(max_sent_size, batch_size, hidden_size)
		state_below = self.lookuptable[encoderInputs.flatten()].reshape((encoderInputs.shape[0], encoderInputs.shape[1], self.hidden_size))
		for _ in range(self.lstm_layers_num):
			enclstm = LSTM(self.hidden_size)
			self.encoder_lstm_layers += enclstm,    #append
			self.params += enclstm.params    #concatenate
			hs, Cs = enclstm.forward(state_below, encoderMask)
			self.hos += hs[-1],
			self.Cos += Cs[-1],
			state_below = hs

		state_below = self.lookuptable[decoderInputs.flatten()].reshape((decoderInputs.shape[0], decoderInputs.shape[1], self.hidden_size))
		for i in range(self.lstm_layers_num):
			declstm = LSTM(self.hidden_size)
			self.decoder_lstm_layers += declstm,    #append
			self.params += declstm.params    #concatenate
			ho, Co = self.hos[i], self.Cos[i]
			state_below, Cs = declstm.forward(state_below, decoderMask, ho, Co)
		decoder_lstm_outputs = state_below

		ei, em, di, dm, dt = tensor.imatrices(5)    #place holders
		#####################################################
		#####################################################
		linear_outputs = tensor.dot(decoder_lstm_outputs, self.linear)
		softmax_outputs, updates = theano.scan(
			fn=lambda x: tensor.nnet.softmax(x),
			sequences=[linear_outputs],
			)

		def _NLL(pred, y, m):
			return -m * tensor.log(pred[tensor.arange(decoderInputs.shape[1]), y])
		costs, updates = theano.scan(fn=_NLL, sequences=[softmax_outputs, decoderTarget, decoderMask])
		loss = costs.sum() / decoderMask.sum()

		gparams = [tensor.grad(loss, param) for param in self.params]
		updates = [(param, param - self.learning_rate*gparam) for param, gparam in zip(self.params, gparams)]

		self._train = theano.function(
			inputs=[ei, em, di, dm, dt],
			outputs=[loss, costs],
			updates=updates,
			givens={encoderInputs:ei, encoderMask:em, decoderInputs:di, decoderMask:dm, decoderTarget:dt}
			)
		#####################################################
		#####################################################
		hs0, Cs0 = tensor.as_tensor_variable(self.hos, name="hs0"), tensor.as_tensor_variable(self.Cos, name="Cs0")
		token_idxs = tensor.fill( tensor.zeros_like(decoderInputs, dtype="int32"), utils.idx_start)
		msk = tensor.fill( (tensor.zeros_like(decoderInputs, dtype="int32")), 1)

		def _step(token_idxs, hs_, Cs_):
			hs, Cs = [], []
			state_below = self.lookuptable[token_idxs].reshape((decoderInputs.shape[0], decoderInputs.shape[1], self.hidden_size))
			for i, lstm in enumerate(self.decoder_lstm_layers):
				h, C = lstm.forward(state_below, msk, hs_[i], Cs_[i])    #mind msk
				hs += h[-1],
				Cs += C[-1],
				state_below = h
			hs, Cs = tensor.as_tensor_variable(hs), tensor.as_tensor_variable(Cs)
			next_token_idx = tensor.cast( tensor.dot(state_below, self.linear).argmax(axis=-1), "int32" )
			return next_token_idx, hs, Cs

		outputs, updates = theano.scan(
			fn=_step,
			outputs_info=[token_idxs, hs0, Cs0],
			n_steps=utils.max_sent_size
			)
		listof_token_idx = outputs[0]
		self._utter = theano.function(
			inputs=[ei, em, di],
			outputs=listof_token_idx,
			givens={encoderInputs:ei, encoderMask:em, decoderInputs:di}
			#givens={encoderInputs:ei, encoderMask:em}
			)
		#####################################################
		#####################################################
	def train(self, encoderInputs, encoderMask, decoderInputs, decoderMask, decoderTarget):
		return self._train(encoderInputs, encoderMask, decoderInputs, decoderMask, decoderTarget)

	def utter(self, encoderInputs, encoderMask):
		decoderInputs=utils.point_data()
		if encoderInputs.ndim == 1:
			encoderInputs = encoderInputs.reshape((encoderInputs.shape[0], 1))
			encoderMask = encoderMask.reshape((encoderMask.shape[0], 1))
		rez = self._utter(encoderInputs, encoderMask ,decoderInputs)
		return rez.reshape((rez.shape[0],))







		
