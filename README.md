# A seq2seq Chatbot

A chatbot based on seq2seq model, implemented by Theano, based on the Google paper [A Neural Conversational Model](http://arxiv.org/abs/1506.05869).

The training dataset can be found here: [Cornell Movie--Dialogs Corpus](http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html)

========
##How to Use  
First, have theano installed.  
Download the dataset above, make a new folder called `data` for it.  
To prepare the data for training, run `make_convs.py`.  
Next run `build_model.py` to train:  
>$THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python build_model.py  

Note that the last few lines are commented out, which illustrate how to make a `Chatbot` instance with a `model` instance, and to talk to the chatbot instance by simply passing it a string.  

![](https://github.com/saltypaul/Seq2Seq-Chatbot/blob/master/pics/Training%20Phase.jpg)
![](https://github.com/saltypaul/Seq2Seq-Chatbot/blob/master/pics/Eval.jpg)
