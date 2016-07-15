from utils import *

class Chatbot(object):
	def __init__(self, model):
		self.model = model

	def utter(conva):
		conva_idxs = parse_input(conva)
		conva_mask = get_mask(conva_idxs)

		convb_idxs = model.utter(conva_idxs, conva_mask)
		convb_tokens = idxs2tokens(convb_idxs)

		convb = cut_end( " ".join( convb_tokens) )
		return convB