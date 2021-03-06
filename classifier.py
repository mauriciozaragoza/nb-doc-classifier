import numpy as np
import sys
from math import log

ny = 20			# number of classes
nx = 61188		# number of words in the vocabulary
nd = 11269		# number of documents
alpha = 1/float(nx)	# alpha value

cy = np.zeros(ny)			# counter for each class
py = np.empty(ny)			# probability for each class
doc_label = np.zeros(nd)	# labels of the documents
confusion = np.zeros((ny,ny), dtype = np.int32)	# confusion matrix

bag = np.zeros((ny, nx), dtype = np.float64)	# bag of words per class
doc_label = np.zeros(nd)

test_label = []	# class for each document on the test data

# This function receives the file path of the train.label file and fill the cy (counter for each class)
# and fill doc_label for each document
def read_counts(filename):
	global py
	document = 0

	for line in open(filename, "r"):	# for each line in the file
		label = int(line.strip()) - 1 	# get label
		cy[label] += 1 					# add 1 to each file of a particular label
		doc_label[document] = label 	# save label of the document 
		document += 1

	py = cy / float(nd)					# calculate probability por each class

# of the current word in their respective class
def read_bag(filename):	
	for line in open(filename, "r"):		# for eac line in the file
		params = line.split(" "); 		
		doc_id = int(params[0]) - 1 		# get the doc id
		word_id = int(params[1]) - 1 		# get the word
		word_count = int(params[2].strip())	# get the counter of that word

		label = doc_label[doc_id]			# get the label of the word is being read

		bag[label][word_id] += word_count	# add the counter

# This function receives the file path of the vocabulary and based on the MAP values calculates the
# top 100 words on which the classification relies on a lot and then those words are writen on a
# file called rank.txt
def get_ranked_words(filename, bag):
	mylist = []
	words = ["" for i in range(nx)]

	for line in bag:
		mylist.extend(line)			# make the bag 1 dimesion

	plain_bag = np.array(mylist)	# convert the 1 dimension bag on numpy vector
	words_index = plain_bag.argsort()[:100] % nx	# sort the np vector and get the top 100 word indexes

	index = 0
	for line in open(filename, "r"):	# read the vocabulary and save it into the words array
		words[index] = line.strip()
		index += 1

	file = open("rank.txt", "w")
	for i in words_index:				# write the file with the top 100 words
		file.write(str(words[i]) + "\n")
	file.close()

# This function receives the test.label file path and fills the test_label array with the classes of each document
def read_validation_label(filename):
	for line in open(filename, "r"):				# for each line in the file
		test_label.append(int(line.strip()) - 1)	# save the class of the current document

# This function receives the test.data file path decide to which class belongs to each document, also calculates
# the total accuracy on the test data set
def read_validation(filename, bag):
	current_id = 0

	sample = np.empty(nx)
	sample[:] = alpha

	correct = 0

	for line in open(filename, "r"):				# for each document in the test file
		params = line.split(" ");
		doc_id = int(params[0]) - 1 				# get the doc id of the current document
		word_id = int(params[1]) - 1 				# get the current word
		word_count = int(params[2].strip())			# get the word count of the current one

		if (doc_id != current_id):					# when start reading other document id
			label = classify(sample, bag)				# classify the document
			if label == test_label[current_id]:		# if the classification was correct correct
				correct += 1 						# add one to correct

			current_id = doc_id						# current id is the new document
			confusion[test_label[current_id]][label] += 1

			sample = np.empty(nx)					# erase the sample
			sample[:] = alpha 						# fill the  new sample with alpha

		sample[word_id] = word_count				# save the word count in the current word position of the sample
	
	confusion[test_label[current_id]][label] += 1 	# add one to the last calculated class in the confusion matrix

	if classify(sample, bag) == test_label[current_id]: # if the answer is correct add one to correct
		correct += 1

	print("\nAlpha: " + str(alpha))										# Print alpha
	print("Correctly classified documents: " + str(correct))			# Print correct answers
	print("Total number of tested documents: " + str(current_id + 1))	# Print total of documents classified

	print("Accuracy: " + str(correct / float(current_id + 1)))			# Print accuracy
	return correct / float(current_id + 1)

# This function apply the MAP calculation to the bag of words
def map_estimate(bag):
	posteriori = np.copy(bag)
	posteriori += alpha
	total_words = posteriori.sum(1)
	
	regularization = total_words + alpha
	for i in range(len(posteriori[0])):
		posteriori[:, i] /= regularization
	
	map_probabilities = posteriori.sum(1);

	# If propabilities do not sum 1, show an exception
	assert abs((map_probabilities).sum() - ny) < 0.01, "conditional probabilities do not sum to 1, actual probability: " + str(map_probabilities) + " , " + str((map_probabilities).sum())

	return np.log2(posteriori)	# Apply log2 to each element in the bag

# This function receives a sample text and returns to which class belogs to
def classify(sample, bag):
	probabilities = (np.dot(bag, sample) + py)	# calculete the probability of each class
	return probabilities.argmax()				# returns the index (class) with the largest probability

# This function writes on a file called confusion.txt the confusion matrix
def print_confusion():
	file = open("confusion.txt", "w")
	for array in confusion:
		for num in array:				# for each num in the confusion matrix
			file.write(str(num) + "\t")	# write on the file
		file.write("\n")

# This function receives a string and returns true if is numeric and false otherwise
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
		print("Alpha must be numeric")
		sys.exit()

print("Reading counts...")
read_counts("data/train.label")
print("Reading bag...")
read_bag("data/train.data")
print("Reading labels...")
read_validation_label("data/test.label")


print("Computing MAP parameters...")
posteriori = map_estimate(bag)
print("Classifying...")
read_validation("data/test.data", posteriori)

print_confusion()
get_ranked_words("data/vocabulary.txt", posteriori)
