import numpy as np
from math import log

ny = 20
nx = 61188
nd = 11269

cy = np.zeros(ny)
py = np.empty(ny)

alpha = 0.01
doc_label = np.zeros(nx)
test_label = []
confusion = np.zeros((ny,ny), dtype = np.int32)

bag = np.zeros((ny, nx), dtype = np.float64)
bag[:] = alpha

def read_counts(filename):
	global py
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cy[label] += 1
		doc_label[document] = label
		document += 1

	py = cy / float(nd)

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

			# print(str(doc_id) + " classified: " + str(label) + " correct: " + str(test_label[current_id]) + " accuracy: " + str(correct / float(current_id)))
		
		sample[word_id] = word_count
	
	confusion[test_label[current_id]][label] += 1

	if classify(sample) == test_label[current_id]:
		correct += 1

	print("correctly classified documents: " + str(correct))
	print("total number of tested documents: " + str(current_id + 1))

	print("accuracy: " + str(correct / float(current_id + 1)))

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
	# print(probabilities)
	return probabilities.argmax()

def print_confusion():
	for array in confusion:
		print(array)


# MAIN

print("reading counts")
read_counts("data/train.label")
print("reading bag")
read_bag("data/train.data")
print("computing MAP parameters")
map_estimate()
print("reading labels")
read_validation_label("data/test.label")
print("classifying")
read_validation("data/test.data")

#print_confusion()