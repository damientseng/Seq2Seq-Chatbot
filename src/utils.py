import theano 
import numpy as np
import re 
import json

max_sent_size = np.int32(50)
idx_start = np.int32(1)
idx_end = np.int32(2)
idx_unk = np.int32(3)    #unknown

token_start = "<start>"
token_end = "<end>"
token_unk = "<unk>"

path_train = "../data/train.txt"
path_idx2token = "../data/idx2token.json"
path_token2idx = "../data/token2idx.json"
path_movie_lines = "../data/movie_lines.txt"
path_line_dict = "../data/line_dict.json"
path_corpus = "../data/corpus.txt"
path_movie_conversations ="../data/movie_conversations.txt"
path_convs = "../data/convs.txt"
path_pkl = "../data/model.pkl"
path_log = "../data/log.txt"


def ortho_matrix(size):
	"""
	type shape: two-tuple like (axis1, axis2), both are integers
	para shape: shape of desired matrix  
	"""
	W = np.random.randn(size, size)    #standard normal distribution, mind the order!!!
	u, s, v = np.linalg.svd(W)

	return u.astype(theano.config.floatX)

def init_lstm_W_U(size):
	return np.concatenate(
			[ ortho_matrix( size ),
			  ortho_matrix( size ),
			  ortho_matrix( size ),
			  ortho_matrix( size ), ], axis=1).astype(theano.config.floatX)

def init_uniform(n_in, n_out):
	W = np.asarray(
		np.random.uniform(
			low=-4.*np.sqrt(6. / (n_in + n_out)),
			high=4.*np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
			),
		dtype=theano.config.floatX
		)
	return W 

def init_norm(n_in, n_out):
	W = np.asarray(np.random.randn(n_in, n_out)*0.1, dtype=theano.config.floatX)
	return W

def numpy_floatX(data):
	return np.asarray(data, dtype=theano.config.floatX)

def point_mask():
	return np.ones((1, 1), dtype=theano.config.floatX)

def point_data():
	return np.asarray([[1]], dtype="int32")
	#return np.zeros(max_sent_size, dtype="int32").reshape((max_sent_size, 1))

"""
def load_train_data(datapath):
	ei = [[4],[3],[0]]
	di = [[1],[3],[4]]
	dt = [[3],[4],[2]]
	#(sent_size, 12)
	ei = np.tile(np.asarray(ei, dtype="int32"), 12)
	di = np.tile(np.asarray(di, dtype="int32"), 12)
	dt = np.tile(np.asarray(dt, dtype="int32"), 12)
	return ei, di, dt
"""

def load_train_data(datapath, num):
	"""
	padding and encoderInput reversing is done in make_convs.make_train_data()
	"""
	encIpt, decIpt, decTgt = [], [], []
	with open(datapath) as mf:
		lines = mf.readlines()
		for line in lines[:num]:
			ei, di, dt = line.strip().split("	") #tab
			encIpt += np.fromstring(ei, dtype="int32", sep=" "),
			decIpt += np.fromstring(di, dtype="int32", sep=" "),
			decTgt += np.fromstring(dt, dtype="int32", sep=" "),
	return np.asarray(encIpt).T, np.asarray(decIpt).T, np.asarray(decTgt).T



def load_json(path):
	res =  None
	with open(path) as mf:
		res = json.load(mf)
	return res 


def get_mask(data):
	mask = (np.not_equal(data, 0)).astype("int32")
	return mask 

def tokenize(s):
	"""
	very raw tokenizer
	"""
	s = clearn_str(s)
	return s.strip().split(" ")

def clearn_str(s):
	s = re.sub(r"\s+", " ", s)
	s = re.sub(r"\-", "", s)
	return s

def cut_and_pad(ilist, max_size=max_sent_size):
	ilist = ilist[:max_size]
	rez = ilist + ["0"]*(max_size-len(ilist))
	return rez

def reverse_list(ilist):
	return ilist[::-1]

def str2idxs(s, tkidx=None):
	if not tkidx:
		tkidx = load_json(path_token2idx)
	tokens = tokenize(s)
	rev = [ str(tkidx.get(t, 3)) for t in tokens]  #default to <unk>
	return rev

def idxs2tokens(idxs, idxtk=None):
	if not idxtk:
		idxtk = load_json(path_idx2token)
	rez = []
	for idx in idxs:
		rez += idxtk[str(idx)],
	return rez

def parse_input(conv):
	conv_idxs = np.asarray( reverse_list( cut_and_pad( str2idxs( clearn_str(conv) ) ) ) , dtype="int32")
	return conv_idxs

def cut_end(s):
	fid = s.find(token_end)
	if fid == -1:
		return s
	elif fid == 0:
		return ""
	else:
		return s[:fid-1]


if __name__ == '__main__':
	print "in utils"
