import numpy as np
from hmm import HMM


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    for w_index, word in enumerate(unique_words.keys()):
        word2idx[word] = w_index

    for t_index, tag in enumerate(tags):
        tag2idx[tag] = t_index

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################

    total_count = len(train_data)
    first_tags = [line.tags[0] for line in train_data]
    first_tags_count = [first_tags.count(i) for i in tag2idx.keys()]
    for s in range(S):
        pi[s] = first_tags_count[s] / total_count

    for line in train_data:
        tag_ids = [tag2idx[i] for i in line.tags]
        word_ids = [word2idx[i] for i in line.words]
        for tid in range(len(tag_ids) - 1):
            A[tag_ids[tid], tag_ids[tid + 1]] += 1
            B[tag_ids[tid], word_ids[tid]] += 1
        B[tag_ids[-1], word_ids[-1]] += 1

    A = (A / np.sum(A, axis=1)[:, None])
    B = (B / np.sum(B, axis=1)[:, None])

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    current_words = model.obs_dict.keys()
    new_index = len(current_words)

    new_words = []
    for line in test_data:
        new_words.extend(line.words)
    to_add = list(set(new_words) - set(current_words))

    expand_value = len(to_add)

    if expand_value > 0:
        for i in to_add:
            model.obs_dict[i] = new_index
            new_index += 1

        new_matrix = np.zeros([len(tags), expand_value])
        new_matrix[:, :] = 1e-6

        model.B = np.append(model.B, new_matrix, axis=1)

    for line in test_data:
        tagging.append(model.viterbi(line.words))

    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):
    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
