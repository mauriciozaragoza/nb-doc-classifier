import matplotlib.pyplot as plt
import numpy as np
from math import log

ny = 20
nx = 61188
nd = 11269

cy = np.zeros(ny)
py = np.empty(ny)

alpha = 0.01
doc_label = np.zeros(nd)
test_label = []
confusion = np.zeros((ny,ny), dtype = np.int32)

bag = np.zeros((ny, nx), dtype = np.float64)

def read_counts(filename):
	global py
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cy[label] += 1
		doc_label[document] = label
		document += 1

	py = cy / float(nd)

def read_validation(filename, bag):
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
			label = classify(sample, bag)
			if label == test_label[current_id]:
				correct += 1

			current_id = doc_id

			confusion[test_label[current_id]][label] += 1

			sample = np.empty(nx)
			sample[:] = alpha

			# print(str(doc_id) + " classified: " + str(label) + " correct: " + str(test_label[current_id]) + " accuracy: " + str(correct / float(current_id)))
		
		sample[word_id] = word_count
	
	confusion[test_label[current_id]][label] += 1

	if classify(sample, bag) == test_label[current_id]:
		correct += 1

	print("correctly classified documents: " + str(correct))
	print("total number of tested documents: " + str(current_id + 1))

	print("accuracy: " + str(correct / float(current_id + 1)))
	return correct / float(current_id + 1)

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

def map_estimate(bag):
	posteriori = np.copy(bag)
	posteriori += alpha
	total_words = posteriori.sum(1)
	
	regularization = total_words + alpha
	for i in range(len(posteriori[0])):
		posteriori[:, i] /= regularization

	map_probabilities = posteriori.sum(1);

	assert abs((map_probabilities).sum() - ny) < 0.01, "conditional probabilities do not sum to 1, actual probability: " + str(map_probabilities) + " , " + str((map_probabilities).sum())

	return np.log2(posteriori)

def classify(d, bag):
	probabilities = (np.dot(bag, d) + py)
	# print(probabilities)
	return probabilities.argmax()

def print_confusion():
	for array in confusion:
		print(array)


# MAIN
n_samples = 20
# xaxis = sorted(np.logspace(log(0.00001, 10), log(1, 10), num = n_samples, base = 10))
xaxis = sorted(np.linspace(0.00005, 0.0005, num = 10))

print("testing on alphas: " + str(xaxis))
yaxis = []

plt.xlabel('Alpha')
plt.ylabel('Accuracy')
# plt.xscale('log')
plt.grid(True)

print("reading counts")
read_counts("data/train.label")
print("reading bag")
read_bag("data/train.data")
print("reading labels")
read_validation_label("data/test.label")

for a in xaxis:
	alpha = a
	print("computing MAP parameters")
	posteriori = map_estimate(bag)
	print("classifying")
	yaxis.append(read_validation("data/test.data", posteriori))

print ("results: " + str(yaxis))
plt.plot(xaxis, yaxis)
plt.show();
#print_confusion()