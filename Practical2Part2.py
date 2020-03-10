from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown

sents = brown.tagged_sents(tagset='universal')
training_set = sents[0:10000]
testing_set = sents[10000:10500]

def tag_sentences_and_concatenate(data):
    all_sentences = []
    for sentence in data:
        all_sentences.append(('','START'))
        all_sentences.extend(sentence)
        all_sentences.append(('','END'))
    
    return all_sentences

def get_emissions(data):
    smoothed = {}
    tagged_data = tag_sentences_and_concatenate(data)
    tags = set([t for (_,t) in tagged_data])
    for tag in tags:
        words = [w for (w,t) in tagged_data if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    return smoothed
    # print('probability of Det -> The is', smoothed['DET'].prob('The'))

def get_transitions(data):
    smoothed = {}
    tag_pairs = get_tagged_tuples(data)
    tags = set([t for (t,_) in tag_pairs])
    for tag in tags:
        next_tags = [next_tag for (cur_tag,next_tag) in tag_pairs if cur_tag == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(next_tags), bins=1e5)
    # print('probability of START -> DET is', smoothed['NOUN'].prob('NOUN'))
    return smoothed

def get_tagged_tuples(data):
    all_pairs = []
    for sentence in data:
        last_tag = 'START'
        for pair in sentence:
            all_pairs.append((last_tag, pair[1]))
            last_tag = pair[1]
        all_pairs.append((last_tag, 'END'))
    return all_pairs

emissions = get_emissions(training_set)
transitions = get_transitions(training_set)

def get_max_prob(transitions, emissions, current_tag, word, prev_probabilites):
    max_prob = 0.0
    max_prev_tag = ''
    #print(transitions)
    for prev_tag in transitions:
        if transitions[prev_tag].prob(current_tag)*prev_probabilites[prev_tag]['probability'] > max_prob:
            max_prob = transitions[prev_tag].prob(current_tag)*prev_probabilites[prev_tag]['probability']
            max_prev_tag = prev_tag
    #print(max_prev_tag)
    max_prob = max_prob * emissions[current_tag].prob(word)
    return {'parent': max_prev_tag, 'probability': max_prob}

def backtrack(probability_table):
    predicted = []
    current_tag = None
    current_max = 0.0

    for prob in probability_table[-1]:
        if (float(probability_table[-1][prob]['probability']) > current_max):
            current_max = float(probability_table[-1][prob]['probability'])
            current_tag = prob
    
    current_parent = probability_table[-1][prob]['parent']
    predicted.append(current_tag)

    for i in range(len(probability_table)-1, -1, -1):
        predicted.insert(0, current_parent)
        current_parent = probability_table[i][current_parent]['parent']
    print(predicted)
    return predicted

def print_table(probability_table):
    for i in range (0, len(probability_table)):
        #print('line')
        print(probability_table[i])

def viterbi(sentence, emissions, transitions):
    probability_table = [{}]
    for current_tag in transitions:
        if current_tag is not 'END':
            max_prob = transitions['START'].prob(current_tag)
            max_prob = max_prob * emissions[current_tag].prob(sentence[0])
            probability_table[0][current_tag] = {'parent': 'START', 'probability': max_prob}

    for i in range(1, len(sentence)):
        probability_table.append({})

        for current_tag in transitions:
            probability_table[i][current_tag] = get_max_prob(transitions, emissions, current_tag, sentence[i], probability_table[i-1])

    predicted_tags = backtrack(probability_table)
    return predicted_tags

def get_prediction_accuracy(testing_set):
    total_num = 0
    correct = 0

    for sentence in testing_set:
        words = [w for (w,_) in sentence]  #need to split into words and tags
        tags = [t for (_,t) in sentence]
        total_num = total_num+len(tags)
        prediction = viterbi(words, emissions, transitions)
        for i in range (0, len(tags)):
            if (prediction[i] == tags[i]):
                correct=correct+1
    accuracy = correct/total_num
    return accuracy

print(get_prediction_accuracy(testing_set))