ny = 20
nx = 61188

cy = [0 for i in range(ny)]
py = [0 for i in range(ny)]

doc_label = [0 for i in range(nx)]

bag = [[0 for i in range(nx)] for j in range(ny)]

def read_counts(filename):
	document = 0

	for line in open(filename, "r"):
		label = int(line.strip()) - 1
		cy[label] += 1
		doc_label[document] = label
		document += 1
		
	py = [cy[i]/nx for i in range(ny)]

def read_bag(filename):
	for line in open(filename, "r"):
		params = line.split(" ");
		doc_id = int(params[0])
		word_id = int(params[1]) 
		word_count = int(params[2].strip())

		label = doc_label[doc_id]

		bag[label][word_id] += word_count



# MAIN

read_counts("data/train.label")
read_bag("data/train.data")