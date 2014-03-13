import sys
import numpy as np
from math import log

ny = 20
nx = 61188
nd = 11269
alpha = 1/float(nx)

cy = np.zeros(ny)
py = np.empty(ny)
doc_label = np.zeros(nx)
confusion = np.zeros((ny,ny), dtype = np.int32)

bag = np.zeros((ny, nx), dtype = np.float64)
bag[:] = alpha

test_label = []

def read_counts(filename):
	global py
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cy[label] += 1
		doc_label[document] = label
		document += 1

	py = cy / float(nd)

def get_ranked_words(filename):
	mylist = []
	words = ["" for i in range(nx)]

	for line in bag:
		mylist.extend(line)

	plain_bag = np.array(mylist)
	words_index = plain_bag.argsort()[:100] % nx

	index = 0
	for line in open(filename, "r"):
		words[index] = line.strip()
		index += 1

	file = open("rank.txt", "w")
	for i in words_index:
		file.write(str(words[i]) + "\n")
	file.close()

def read_validation(filename):
	current_id = 0

	sample = np.empty(nx) # [0 for i in range(nx)]
	sample[:] = alpha

	correct = 0

	for line in open(filename, "r"):
		params = line.split(" ");
		doc_id = int(params[0]) - 1
		word_id = int(params[1]) - 1
		word_count = int(params[2].strip())

		if (doc_id != current_id):	
			label = classify(sample)
			if label == test_label[current_id]:
				correct += 1

			current_id = doc_id

			confusion[test_label[current_id]][label] += 1

			sample = np.empty(nx)
			sample[:] = alpha
		
		sample[word_id] = word_count
	
	confusion[test_label[current_id]][label] += 1

	if classify(sample) == test_label[current_id]:
		correct += 1

	print("\nCorrectly classified documents: " + str(correct))
	print("Total number of tested documents: " + str(current_id + 1))

	print("Accuracy: " + str(correct / float(current_id + 1)))

def read_validation_label(filename):
	for line in open(filename, "r"):
		test_label.append(int(line.strip()) - 1)

def read_bag(filename):
	for line in open(filename, "r"):
		params = line.split(" ");
		doc_id = int(params[0]) - 1
		word_id = int(params[1]) - 1
		word_count = int(params[2].strip())

		label = doc_label[doc_id]

		bag[label][word_id] += word_count

def map_estimate():
	global bag

	total_words = bag.sum(1)
	
	regularization = total_words + alpha
	for i in range(len(bag[0])):
		bag[:, i] /= regularization

	map_probabilities = bag.sum(1);

	assert abs((map_probabilities).sum() - ny) < 0.01, "conditional probabilities do not sum to 1, actual probability: " + str(map_probabilities) + " , " + str((map_probabilities).sum())

	bag = np.log2(bag)

def classify(d):
	probabilities = (np.dot(bag, d) + py)
	return probabilities.argmax()

def print_confusion():
	file = open("confusion.txt", "w")
	for array in confusion:
		file.write(array)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# MAIN

if len(sys.argv) == 2:
	if is_number(sys.argv[1]):
		alpha = float(sys.argv[1])
	else:
		print "Alpha must be numeric"
		sys.exit()

print("Reading counts...")
read_counts("data/train.label")
print("Reading bag...")
read_bag("data/train.data")
print("Computing MAP parameters...")
map_estimate()
print("Getting top words...")
get_ranked_words("data/vocabulary.txt")
print("Reading labels...")
read_validation_label("data/test.label")
print("Classifying...")
read_validation("data/test.data")
print_confusion()