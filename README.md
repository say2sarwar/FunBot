# FunBot
Chatbot for Fun using Seq2Seq model. 


### Requirement
Python 3.xx

Pytorch 

### This system can be run using Test.py

Step 1
Download data from [Cornell Movie dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 

Make sure that to put the data files in ../save/

Training python3 train.py

testing python3 Test.py

data_processing.py can be modified for dir setting.  


## References

[Neural Conversational Model](https://arxiv.org/abs/1506.05869)

[Attention Neural Machine](https://arxiv.org/abs/1508.04025)

[Seq2Seq Translation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

[Neural Conversational Model](https://arxiv.org/abs/1506.05869)

[GANs](https://arxiv.org/abs/1406.2661)


[Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)

[Attention-based Neural Machine](https://arxiv.org/abs/1508.04025)


### Acknowledgements
we borrow code from following sources
1. Sean Robertson’s practical-pytorch seq2seq-translation example: https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
2. FloydHub’s Cornell Movie Corpus preprocessing code: https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
