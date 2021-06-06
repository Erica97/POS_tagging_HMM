# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np

def get_initial_probability_table(words, tags, unique_tags):
    initial_table = np.full(len(unique_tags), 0.0001, dtype='float')
    for i in range(len(words)-1):
        if tags[i] == 'PUN': #end of sentence
            initial_table[unique_tags[tags[i+1]]] += 1
    return initial_table/sum(initial_table)

def get_emission_matrix(tags, unique_tags, words, unique_words):
    emission_matrix = np.full((len(unique_tags), len(unique_words)), 0.0001, dtype='float')
    for w in range(len(words)):
        emission_matrix[unique_tags[tags[w]], unique_words[words[w]]] += 1

    row_sum = emission_matrix.sum(axis=1)
    norm_emission_matrix = emission_matrix/row_sum[:,np.newaxis]
    return norm_emission_matrix


#transition_matrix
def get_transition_matrix(tags, unique_tags):
    #creating t x t transition matrix of tags
    transition_matrix = np.full((len(unique_tags), len(unique_tags)), 0.001, dtype='float')
    for t in range(len(tags)-1):
        transition_matrix[unique_tags[tags[t]], unique_tags[tags[t+1]]] += 1
    row_sum = transition_matrix.sum(axis=1)
    norm_transition_matrix = transition_matrix/row_sum[:,np.newaxis]

    return norm_transition_matrix

# Viterbi Algorithm
def Viterbi(test_words, tags, words, unique_words, unique_tags):
    #transition matrix
    transition_matrix = get_transition_matrix(tags, unique_tags)
    emission_matrix = get_emission_matrix(tags, unique_tags, words, unique_words)
    #build initial probabilities table
    initial_probs = get_initial_probability_table(words, tags, unique_tags)
    most_frequent_word = np.argmax(initial_probs)
    prob_trellis = np.zeros((len(unique_tags), len(test_words)), dtype='float')
    path = {}
    for i in range(len(unique_tags)):
        path[i] = np.array(i)
    print("start Viterbi: ")
    #determine trellis values for x1, and normalize probe trellis column
    if test_words[0] in unique_words:
        prob_trellis[:,0] = initial_probs*emission_matrix[:, unique_words[test_words[0]]]/sum(initial_probs*emission_matrix[:,unique_words[test_words[0]]])
    else:
        prob_trellis[:,0] = initial_probs*emission_matrix[:, most_frequent_word]/sum(initial_probs*emission_matrix[:, most_frequent_word])


    #for x2 to xt find each state's most likely prior state x
    for o in range(1, len(test_words)):
        new_path = {}
        for s in range(len(unique_tags)):
            if test_words[o] not in unique_words:
                prior_tag = np.argmax(prob_trellis[:,o-1]*transition_matrix[:,s]*emission_matrix[s, most_frequent_word])
                prob_trellis[s, o] = prob_trellis[prior_tag, o-1]*transition_matrix[prior_tag, s]*emission_matrix[s, most_frequent_word]
            else:
                prior_tag = np.argmax(prob_trellis[:,o-1]*transition_matrix[:,s]*emission_matrix[s, unique_words[test_words[o]]])
                prob_trellis[s, o] = prob_trellis[prior_tag, o-1]*transition_matrix[prior_tag, s]*emission_matrix[s, unique_words[test_words[o]]]
            #update path
            new_path[s] = np.append(path[prior_tag], s)
        path = new_path
        #normalize columns
        prob_trellis[:, o] = prob_trellis[:, o]/sum(prob_trellis[:, o])

    #print("index of largest prob:", np.argmax(prob_trellis[:,-1]))
    solution  = []
    reversed_unique_tags = {value : key for (key, value) in unique_tags.items()}
    #print("path? ", path[np.argmax(prob_trellis[:,-1])])
    for stage in path[np.argmax(prob_trellis[:,-1])]:
        solution.append(reversed_unique_tags[stage])
    #print("solution:", len(test_words), len(solution), solution)
    return solution

def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #store all training tuples in the corpus
    train_tuples = []
    train_corpus = []
    #load in the training files
    for train in training_list:
        with open(train, 'r') as f:
            train_tuples += f.readlines()
    #get words and tags lists
    for tup in train_tuples:
        tup = tup.strip('\n').split(" : ")
        train_corpus.append(tup)
    #get unique tags dict
    tags = [pair[1] for pair in train_corpus]
    unique_tags = {}
    for index, tag in enumerate(set(tags)):
        unique_tags[tag] = index
    #get unique words dict
    words = [pair[0] for pair in train_corpus]
    unique_words = {}
    for index, word in enumerate(set(words)):
        unique_words[word] = index
    #load in the test file
    test_corpus = []
    with open(test_file, 'r') as t:
            test_words = t.readlines()
    for word in test_words:
        word = word.strip('\n')
        test_corpus.append(word)
    #get output
    predicted_tags = Viterbi(test_corpus, tags, words, unique_words, unique_tags)
    #output txt file
    with open(output_file, 'w') as o:
        for i in range(len(predicted_tags)):
            o.write(f"{test_corpus[i]} : {predicted_tags[i]}\n")
    return

if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
