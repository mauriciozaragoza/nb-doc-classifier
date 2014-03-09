# from multiprocessing import Pool
import numpy as np
from math import log

ny = 20
nx = 61188
nd = 11269

cy = np.zeros(ny)
py =  np.zeros(ny)

alpha = 1 / float(nx)
doc_label = np.zeros(nx)
test_label = []

bag = np.zeros((ny, nx)) # [[0 for i in range(nx)] for j in range(ny)]

def read_counts(filename):
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cy[label] += 1
		doc_label[document] = label
		document += 1

	for i in range(ny):
		py[i] = cy[i]/float(nd)

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

			sample = np.empty(nx)
			sample[:] = alpha

			#print str(doc_id) + " classified: " + str(label) + " correct: " + str(test_label[current_id]) + " accuracy: " + str(correct / float(current_id))
		
		sample[word_id] = word_count
			
	if classify(sample) == test_label[current_id]:
		correct += 1

	print "classified: " + str(label) + " correct: " + str(test_label[current_id]) + " accuracy: " + str(correct / float(current_id + 1))

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
	for i in range(len(bag)):
		total_words = 0

		for j in range(len(bag[0])):
			total_words += bag[i][j]
			bag[i][j] += alpha
			
		for j in range(len(bag[0])):
			bag[i][j] /= (total_words + nx)

		for j in range(len(bag[0])):
			bag[i][j] = log(bag[i][j], 2)

def classify(d):
	posteriori = np.dot(bag, d) + py
	return posteriori.argmax()

# MAIN

print "reading counts"
read_counts("data/train.label")
print "reading bag"
read_bag("data/train.data")
print "computing posteriori"
map_estimate()
print "reading labels"
read_validation_label("data/test.label")
print "classifying"
read_validation("data/test.data")