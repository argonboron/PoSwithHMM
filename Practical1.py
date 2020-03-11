from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.corpus import alpino
import csv

sents = brown.tagged_sents(tagset='universal')
dutch_sents = alpino.tagged_sents()
d_len = len(dutch_sents)
print(d_len)
# Split testing and training data 
training_set = sents[0:10000]
testing_set = sents[10000:10500]
known_words = []
dutch_training = dutch_sents[0:d_len-500]
dutch_testing = sents[-500:]



# Turn all sentences into one long list, adding start and end tags
def tag_sentences_and_concatenate(data, use_unk, threshold, unk, dutch):
    all_sentences = []

    for sentence in data:
        all_sentences.append(('<start>','START'))
        all_sentences.extend(sentence)
        all_sentences.append(('<end>','END'))
    known_words = count_words(all_sentences, threshold)
    if use_unk:
        all_sentences = deal_with_unknown_words(all_sentences, known_words, unk, dutch)
    return all_sentences, known_words

# Get Witten Bell smoothed probabilities for emissions
def get_emissions(data, use_unk, threshold, unk, dutch):
    smoothed = {}
    tagged_data, known_words = tag_sentences_and_concatenate(data, use_unk, threshold, unk, dutch)

    tags = set([t for (_,t) in tagged_data])

    for tag in tags:
        words = [w for (w,t) in tagged_data if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    return smoothed, known_words

#replace words that fit critera and are not 'known' with UNK tags
def deal_with_unknown_words(data, known_words, unk, dutch):
    for i in range (0, len(data)):
        word = data[i][0]
        if word not in known_words:
            if dutch:
                if word[-5:] == 'ische':
                    new_tuple = ('UNK-ische', data[i][1])
                    data[i] = new_tuple
                elif word[-4:] == 'thie':
                    new_tuple = ('UNK-thie', data[i][1])
                    data[i] = new_tuple
                elif word[-6:] == 'achtig':
                    new_tuple = ('UNK-achtig', data[i][1])
                    data[i] = new_tuple
            else:
                if i is not 0 and data[i-1][1] is not 'START':
                    if len(word) > 1 and word[0].isupper() and word[1:].islower():
                        new_tuple = ('Unk', data[i][1])
                        data[i] = new_tuple
                elif unk > 1 and word[-3:] == 'ing':
                    new_tuple = ('UNK-ing', data[i][1])
                    data[i] = new_tuple
                elif unk > 2 and word[-2:] == 'ly':
                    new_tuple = ('UNK-ly', data[i][1])
                    data[i] = new_tuple
                elif unk > 3 and word[-3:] == 'ion':
                    new_tuple = ('UNK-ion', data[i][1])
                    data[i] = new_tuple
                elif unk > 4 and word[-2:] == 'al':
                    new_tuple = ('UNK-al', data[i][1])
                    data[i] = new_tuple
    return data

#Find out if words are significant by counting occurences, and return a list of 'known' words
def count_words(data, threshold):
    count = {}
    known = []
    words = [w for (w,_) in data]
    for word in words:
        if word is not '':
            if word in count:
                count[word] = count[word] + 1
            else:
                count[word] = 1    
            if count[word] > threshold and word not in known:
                known.append(word)
    return known

#Generate tuple pairs for each tag followign the previous - including start and end tags.
def get_tagged_tuples(data):
    all_pairs = []

    for sentence in data:
        last_tag = 'START'

        for pair in sentence:
            all_pairs.append((last_tag, pair[1]))
            last_tag = pair[1]

        all_pairs.append((last_tag, 'END'))

    return all_pairs


# Get Witten Bell smoothed probabilites for transitions
def get_transitions(data):
    smoothed = {}
    tag_pairs = get_tagged_tuples(data)

    tags = set([t for (t,_) in tag_pairs])

    for tag in tags:
        next_tags = [next_tag for (cur_tag,next_tag) in tag_pairs if cur_tag == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(next_tags), bins=1e5)

    return smoothed

#Get the maximum probability given the previous layer and the current tag,
def get_max_prob(transitions, emissions, current_tag, word, prev_probabilites):
    max_prob = 0.0
    max_prev_tag = ''

    for prev_tag in transitions:
        if transitions[prev_tag].prob(current_tag)*prev_probabilites[prev_tag]['probability'] > max_prob:
            max_prob = transitions[prev_tag].prob(current_tag)*prev_probabilites[prev_tag]['probability']
            max_prev_tag = prev_tag
            
    max_prob = max_prob * emissions[current_tag].prob(word)
    return {'parent': max_prev_tag, 'probability': max_prob}

# Backtrack through probability table, to generate prediction
def backtrack(probability_table):
    predicted = []
    current_tag = None
    current_max = 0.0

    # Get initial maximum probability from the last row
    for prob in probability_table[-1]:
        if (float(probability_table[-1][prob]['probability']) > current_max):
            current_max = float(probability_table[-1][prob]['probability'])
            current_tag = prob
    
    # Loop back through the parents
    current_parent = current_tag

    for i in range(len(probability_table)-1, -1, -1):
        predicted.insert(0, current_parent)
        if current_parent is None:
            break
        current_parent = probability_table[i][current_parent]['parent']

    return predicted

# Viterbi algorithm
def viterbi(sentence, emissions, transitions):
    probability_table = [{}]

    # Calculate first layer based on start tag
    for current_tag in transitions:
        if current_tag is not 'END':
            prob = transitions['START'].prob(current_tag)
            prob = prob * emissions[current_tag].prob(sentence[0])
            probability_table[0][current_tag] = {'parent': 'START', 'probability': prob}

    # Calculate rest of the layers iteratively
    for i in range(1, len(sentence)):
        probability_table.append({})

        for current_tag in transitions:
            probability_table[i][current_tag] = get_max_prob(transitions, emissions, current_tag, sentence[i], probability_table[i-1])

    #Backtrack to get prediction
    predicted_tags = backtrack(probability_table)
    return predicted_tags

# Calculate accuracy
def get_prediction_accuracy(testing_set, known_words, use_unk, unk, emissions, transitions, dutch):
    total_num = 0
    correct = 0
    for sentence in testing_set:

        if use_unk:
            sentence = deal_with_unknown_words(sentence, known_words, unk, dutch)

        words = [w for (w,_) in sentence]
        tags = [t for (_,t) in sentence]
        total_num = total_num + len(tags)

        prediction = viterbi(words, emissions, transitions)

        for i in range (0, len(prediction)):
            if prediction[i] == tags[i]:
                correct = correct+1

    accuracy = correct/total_num
    print(correct, total_num, accuracy)
    return accuracy


def test_experiment_Unk():
    header = ["Threshold", "No Unk", "Unk", "UNK-ing", "UNK-ly", "UNK-ion", "UNK-al"]
    row_list = [header]
    use_unk = False
    emissions, known_words = get_emissions(training_set, use_unk, 0, 0, False)
    transitions = get_transitions(training_set)
    use_unk = True
    for i in range (1,6):
        row = [str(i)]
        for u in range (0,6):
            print('Training...', i, u)
            emissions, known_words = get_emissions(training_set, use_unk, i, u, False)
            transitions = get_transitions(training_set)
            row.append(get_prediction_accuracy(testing_set, known_words, use_unk, u, emissions, transitions, False))
        row_list.append(row)
    with open('unk_experiments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


#test_experiment_Unk()

def test_experiment():
    use_unk = False
    emissions, known_words = get_emissions(training_set, use_unk, 0, 0, False)
    transitions = get_transitions(training_set)
    training_num = 1000
    header = [""]
    for i in range (0, 12):
        header.append(training_num)
        training_num = training_num + 500
    row_list = [header]
    for i in range (0, 2):
        training_num = 1000
        use_unk = not use_unk
        if use_unk:
            row = ["With Unk"]
        else:
            row = ["No Unk"]
        for u in range (0,12):
            print('Training...', use_unk, training_num)
            emissions, known_words = get_emissions(sents[0:training_num], use_unk, 1, 5, False)
            transitions = get_transitions(sents[0:training_num])
            row.append(get_prediction_accuracy(sents[-500:], known_words, use_unk, 5, emissions, transitions, False))
            training_num = training_num + 500
        row_list.append(row)
    with open('experiments_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


test_experiment()

training_set = dutch_training
testing_set = dutch_testing
sents = dutch_sents

def test_dutch():
    use_unk = False
    emissions, known_words = get_emissions(training_set, use_unk, 0, 0, True)
    transitions = get_transitions(training_set)
    training_num = 1000
    header = [""]
    for i in range (0, 12):
        header.append(training_num)
        training_num = training_num + 500
    row_list = [header]
    for i in range (0, 2):
        training_num = 1000
        use_unk = not use_unk
        if use_unk:
            row = ["With Unk"]
        else:
            row = ["No Unk"]
        for u in range (0,12):
            print('Training...', use_unk, training_num)
            emissions, known_words = get_emissions(sents[0:training_num], use_unk, 1, 5, True)
            transitions = get_transitions(sents[0:training_num])
            row.append(get_prediction_accuracy(sents[-500:], known_words, use_unk, 5, emissions, transitions, True))
            training_num = training_num + 500
        row_list.append(row)
    with open('experiments_dutch.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

#test_dutch()