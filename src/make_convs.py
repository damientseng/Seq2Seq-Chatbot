import numpy as np 
from utils import *
import json
import re


def make_line_dict():
	dix = {}
	corpus = []
	with open(path_movie_lines) as mf:
		line_infos = mf.readlines()
		for line in line_infos:
			fields = line.strip().split(" +++$+++ ")
			key, value = fields[0].strip(), unicode(fields[-1].strip(), errors="ignore")
			value = clearn_str(value)
			dix[key] = value
			corpus += value,
	with open(path_line_dict, "wb") as mf:
		json.dump(dix, mf)
	with open(path_corpus, "wb") as mf:
		for line in corpus:
			mf.write(line+"\n")


def make_conversations():
	rev = []
	line_dict = load_json(path_line_dict)

	with open(path_movie_conversations) as mf:
		lines = mf.readlines()
		for line in lines:
			utterance = line.strip().split(" +++$+++ ")[-1][1:-1]
			utterance = utterance.strip().split(", ")
			for i in range(len(utterance)-1):
				ia, ib = utterance[i][1:-1], utterance[i+1][1:-1]
				ca, cb = line_dict[ia], line_dict[ib]
				conv = ca + " +++$+++ " + cb
				rev += conv,
	with open(path_convs, "wb") as mf:
		for conv in rev:
			mf.write(conv + "\n")


def make_dictionary():
	"""
	each line of corpus should be a sentence.
	"""
	dix = {}
	total = 0.
	with open(path_corpus) as mf:
		lines = mf.readlines()
		for line in lines:
			tokens = tokenize(line)
			for tk in tokens:
				dix.setdefault(tk, 0)
				dix[tk] += 1
				total += 1
	dix = {k:v/total for k, v in dix.items() if v >= 5} #

	token2idx = {token_start: 1, token_end: 2, token_unk: 3}    #0 is not used
	idx2token = {1: token_start, 2: token_end, 3: token_unk}
	i = 4

	for k, v in dix.items():
		token2idx[k] = i
		idx2token[i] = k
		i += 1

	print len(token2idx)

	with open(path_token2idx, "wb") as mf:
		json.dump(token2idx, mf)

	with open(path_idx2token, "wb") as mf:
		json.dump(idx2token, mf)

def make_train_data():
	dix = load_json(path_token2idx)
	if not dix:
		print "no token2idx dictionary file..."
		return
	rev = []
	with open(path_convs) as mf:
		convs = mf.readlines()
		for conv in convs:
			conva, convb = conv.strip().split(" +++$+++ ")
			encoderInput, decoderInput, decoderTarget = conva, token_start+" "+convb, convb+" "+token_end
			#the encoderInput is reversed here
			enIptIdxs = reverse_list( cut_and_pad(str2idxs(encoderInput, dix)) )
			deIptIdxs = cut_and_pad(str2idxs(decoderInput, dix))
			deTgtIdxs = cut_and_pad(str2idxs(decoderTarget, dix))
			rev += "	".join([" ".join(enIptIdxs), " ".join(deIptIdxs), " ".join(deTgtIdxs)]),
	with open(path_train, "wb") as mf:
		for t in rev:
			mf.write(t + "\n")


if __name__ == '__main__':
	
	make_line_dict()
	make_conversations()
	make_dictionary()
	make_train_data()


