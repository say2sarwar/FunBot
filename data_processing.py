
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 1.0

corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
	with open(file, 'rb') as datafile:
		lines = datafile.readlines()
	for line in lines[:n]:
		print(line)

printLines(os.path.join(corpus, 'movie_lines.txt'))


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations



#path of the file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversations = []

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

print("\nProcessing Corpus...")
lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
print("\nLoading coversations...")
conversations = loadConversations(os.path.join(corpus, 'movie_conversations.txt'),
    lines, MOVIE_CONVERSATIONS_FIELDS)

print("\n Writing ne format file...")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

print("\nSample lines from file:")
printLines(datafile)


PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3
    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("\nkeep_words {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index
                )))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)



MAX_LENGTH = 15

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s 

def readVocs(datafile, corpus_name):
    print("Reading Lines...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[0])
    print("Counted words:", voc.num_words)
    return voc, pairs

save_dir = os.path.join("data",'save')
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

print("\npairs:")
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)

    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} fo total".format(
        len(pairs),len(keep_pairs), len(keep_pairs)/len(pairs)))
    return keep_pairs


pairs = trimRareWords(voc, pairs, MIN_COUNT)

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m =[]
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l,voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc,pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


small_batch_size = 5
batches = batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches


print("input_variable:", input_variable)
print("lengths:",lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:",max_target_len)


# Loss function

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len,
    encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
    batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Variable initializing
    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = target_variable[t].view(1,-1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss

            print_losses.append(mask_loss.item()*nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)


            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals += nTotal

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)


    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder,
    encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers,
    decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
    save_every, clip, corpus_name, loadFilename):

    training_batches = [batch2TrainData(voc, [random.choice(pairs)
        for _ in range(batch_size)]) for _ in range(n_iteration)]

    print("Initializing ...")
    start_iteration = 1
    print_loss =0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print("Training ...")
    for iteration in range(start_iteration, n_iteration +1):
        training_batch = training_batches[iteration - 1]

        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask,
            max_target_len, encoder, decoder, embedding, encoder_optimizer,
            decoder_optimizer, batch_size, clip)

        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration*100, print_loss_avg))
            print_loss = 0

        # Save Checkpoint
        if(iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}-{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()},
                os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

def evaluate(encoder, decoder, searcher, voc, sentence, max_length= MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)

    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)

    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words

class GreedySearchDecoder(nn.Module):
	"""docstring for GreedySearchDecoder"""
	def __init__(self, encoder, decoder):
		super(GreedySearchDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self,input_seq, input_length, max_length):
		encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
		decoder_hidden = encoder_hidden[:decoder.n_layers]

		decoder_input = torch.ones(1,1, device=device, dtype=torch.long)* SOS_token

		all_tokens = torch.zeros([0], device=device, dtype=torch.long)
		all_scores = torch.zeros([0], device=device)

		for _ in range(max_length):
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
			all_scores = torch.cat((all_scores, decoder_scores), dim=0)

			decoder_input = torch.unsqueeze(decoder_input, 0)
		return all_tokens, all_scores





def evaluateInput(encoder, decoder, searcher, voc):
	input_sentence = ''

	while(1):
		try:
			input_sentence = input(':>')
			if input_sentence == 'q' or input_sentence == 'quit':
				break

			input_sentence = normalizeString(input_sentence)

			output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

			output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

			print('MBot:',' '.join(output_words))

		except KeyError:
			print("Error: Unkown Word encountered.")

			# Need to use word2vec or fasttext to approximate the missing words


#Importing file
from encode import *

#Models configuration

model_name = 'chatbot_model'
#attn_model = 'dot'

attn_model = 'general'
hidden_size = 900
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to none if starting from scratch
loadFilename = None 
checkpoint_iter = 500

# file path if loading save model/checkpoints
loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}-{}'.format(
    encoder_n_layers,decoder_n_layers, hidden_size),'{}_checkpoint.tar'.format(checkpoint_iter))

if loadFilename:
	checkpoint = torch.load(loadFilename)
	encoder_sd = checkpoint['en']
	decoder_sd = checkpoint['de']
	encoder_optimizer_sd = checkpoint['en_opt']
	decoder_optimizer_sd = checkpoint['de_opt']
	embedding_sd = checkpoint['embedding']
	voc.__dict__ = checkpoint['voc_dict']

print("Building encoder and decoder ...")

embedding = nn.Embedding(voc.num_words, hidden_size)

if loadFilename:
	embedding.load_state_dict(embedding_sd)

encoder = RNNEncoder(hidden_size, embedding, encoder_n_layers, dropout)
decoder = RnnDecoder(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers,dropout)

if loadFilename:
	encoder.load_state_dict(encoder_sd)
	decoder.load_state_dict(decoder_sd)

encoder = encoder.to(device)
decoder = decoder.to(device)

print('Models has been built, enjoy using it!!!!')
