# FunBot
Chatbot for Fun using Seq2Seq model. 


# what we need? 
Python 3.XX
Pytorch 

### This system can be run using Test.py

Step 1
Download data from [Cornell Movie dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 

Make sure that to put the data files in ../save/

Run Train.py or Test.py

Changes can be done to data_processing.py, if you need to modify the system. 


## References

[Neural Conversational Model](https://arxiv.org/abs/1506.05869)

[Attention Neural Machine](https://arxiv.org/abs/1508.04025)

[Seq2Seq Translation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

### Acknowledgements
we borrow some of the code from following sources
1. Sean Robertson’s practical-pytorch seq2seq-translation example: https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
2. FloydHub’s Cornell Movie Corpus preprocessing code: https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
