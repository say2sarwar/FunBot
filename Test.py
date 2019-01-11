

from train import *

encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder,decoder)

evaluateInput(encoder,decoder, searcher,voc)