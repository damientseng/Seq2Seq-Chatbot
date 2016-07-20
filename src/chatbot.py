from utils import *

class Chatbot(object):
	def __init__(self, model):
		self.model = model

	def utter(self, conva):
		conva_idxs = parse_input(conva)
		conva_mask = get_mask(conva_idxs)

		convb_idxs = self.model.utter(conva_idxs, conva_mask)
		print convb_idxs
		convb_tokens = idxs2tokens(convb_idxs)

		convb = cut_end( " ".join( convb_tokens) )
		return convb
