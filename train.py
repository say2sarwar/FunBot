teacher_forcing_ratio = 1.0
from data_processing import *

clip = 50.0

learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 50000
print_every = 1
save_every = 500 


encoder.train()
decoder.train()

print('Building optimizers ...')

encoder_optimizer = optim.Adam(encoder.parameters(),
	lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(),
	lr= learning_rate*decoder_learning_ratio)
if loadFilename:
	encoder_optimizer.load_state_dict(encoder_optimizer_sd)
	decoder_optimizer.load_state_dict(decoder_optimizer_sd)

print("Start Training!!!!!!")
trainIters(model_name, voc, pairs, encoder,decoder, encoder_optimizer, decoder_optimizer,
	embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
	print_every,save_every, clip, corpus_name, loadFilename)

