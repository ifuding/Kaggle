# import csv
import numpy as np
import random

class data_helper():
	def __init__(self, wv_model_path, load_wv_model, model_type, sequence_max_length=1024):
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
		self.char_dict = {}
		self.sequence_max_length = sequence_max_length
		print("alphabet size: {0}".format(len(self.alphabet)))
		for i,c in enumerate(self.alphabet):
			self.char_dict[c] = i+1
		self.wv_model_path = wv_model_path
		self.load_wv_model = load_wv_model
		self.model_type = model_type


	def char2vec(self, text):
		data = np.zeros(self.sequence_max_length)
		for i in range(0, len(text)):
			if i >= self.sequence_max_length:
				return data
			elif text[i] in self.char_dict:
				data[i] = self.char_dict[text[i]]
			else:
				# unknown character set to be 68
				data[i] = 68
		return data


	#This creates the character n-grams like it is described in fasttext
	def char_ngram_generator(text, n1=3, n2=3):
		# z = []
	#     There is a sentence in the paper where they mention they add a
	#     special character for the beginning and end of the word to
	#     distinguish prefixes and suffixes. This is what I understood.
	#     Feel free to send a pull request if this means something else
		text2 = '*'+text+'*'
		# for k in range(n1,n2):
		# 	z.append([text2[i:i+k] for i in range(len(text2)-k+1)])
		z = [text2[i:i+n] for i in range(0, len(line), n1)]
		# z.append(text)
		return z


	def load_wv_model(self):
    	wv_model = dict()
		if model_type == "word2vec":
			wv_model = gensim.models.KeyedVectors.load_word2vec_format(location, binary=True)
		elif model_type == "fast_text":	
			with open(self.wv_model_path, encoding="utf8") as emb_file:
				# with open("../../Data/normal_stem_wv_indata", "w+") as emb_file_indata:
				for line in emb_file:
					ls = line.strip().split(' ')
					word = ls[0]
						# if word in word_index:
						#     emb_file_indata.write(line)
					wv_model[word] = np.asarray(ls[1:], dtype='float32')
		return wv_model


	def text_to_triletter_sequence(self, location = 'wv_model_norm.gensim', tokenizer = None, nb_words = 10000, \
					embed_size = 300, model_type = "fast_text"):
		"""
		"""
		if not os.path.exists(location):
			print('Found {}'.format(location))
			return None
		print("-----Load Word2Vec Model-----")
		word_index = tokenizer.word_index
		if model_type == "word2vec":
			wv_model = gensim.models.KeyedVectors.load_word2vec_format(location, binary=True)
		elif model_type == "fast_text":
			wv_model = dict()
			with open(location, encoding="utf8") as emb_file:
				# with open("../../Data/normal_stem_wv_indata", "w+") as emb_file_indata:
				for line in emb_file:
					ls = line.strip().split(' ')
					word = ls[0]
						# if word in word_index:
						#     emb_file_indata.write(line)
					wv_model[word] = np.asarray(ls[1:], dtype='float32')
		print("word_index size: {0}".format(len(word_index)))
		embedding_matrix = np.zeros((nb_words, embed_size))
		word_in_corpus = 0
		for word, i in word_index.items():
			if i >= nb_words: continue
			if word in wv_model:
				embedding_matrix[i] = wv_model[word]
				word_in_corpus += 1
		print("{0} Words in corpus!".format(word_in_corpus))

		return embedding_matrix


	def text2sequence(self, text_array):
		"""
		"""
		all_data = []
		for text in text_array:
			text = text.lower()
			all_data.append(self.char2vec(text))
		return np.array(all_data)


	def load_csv_file(self, filename, num_classes):
		"""
		Load CSV file, generate one-hot labels and process text data as Paper did.
		"""
		all_data = []
		labels = []
		with open(filename) as f:
			reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
			for row in reader:
				# One-hot
				one_hot = np.zeros(num_classes)
				one_hot[int(row['class']) - 1] = 1
				labels.append(one_hot)
				# Char2vec
				data = np.ones(self.sequence_max_length)*68
				text = row['fields'][-1].lower()
				all_data.append(self.char2vec(text))
		f.close()
		return np.array(all_data), np.array(labels)


	def load_dataset(self, dataset_path):
		# Read Classes Info
		with open(dataset_path+"classes.txt") as f:
			classes = []
			for line in f:
				classes.append(line.strip())
		f.close()
		num_classes = len(classes)
		# Read CSV Info
		train_data, train_label = self.load_csv_file(dataset_path+'train.csv', num_classes)
		test_data, test_label = self.load_csv_file(dataset_path+'test.csv', num_classes)
		return train_data, train_label, test_data, test_label


	def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
			else:
				shuffled_data = data
			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				yield shuffled_data[start_index:end_index]