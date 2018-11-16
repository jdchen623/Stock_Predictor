import collections

import numpy as np
import pandas as pd
import util
import svm

TRAIN_SPLIT = .6
VALIDATION_SPLIT = .2
TEST_SPLIT = .2

def hello():
    print("Hello!")

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # HELLO!!!

    # *** START CODE HERE ***
    words = np.core.defchararray.lower(message.split(" "))
    return words
    # *** END CODE HERE ***

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counts = {}
    for message in messages:
        words = set(get_words(message))
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    dict = {}
    uuid = 0
    for word in counts:
        if counts[word] >= 5:
            dict[word] = uuid
            uuid += 1
    return dict
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    text = np.zeros([len(messages), len(word_dictionary)], dtype=int)
    row = 0
    for message in messages:
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                text[row, word_dictionary[word]] += 1
        row += 1
    return text
    # *** END CODE HERE ***

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    print("*******************training*******************")
    H, W = matrix.shape
    k = W
    phi_0 = np.zeros([k, ])
    denom_0 = 0
    phi_1 = np.zeros([k, ])
    denom_1 = 0
    phi_y = 0

    for i in range(H):
        print("training "+ str(i) + " of " + str(H))

        n_i = 0
        y_i = labels[i]
        x_i = matrix[i]

        for j in range(W):
            x_i_j = x_i[j]
            n_i += x_i_j

            if y_i == 0:
                phi_0[j] += x_i_j
            else:
                phi_1[j] += x_i_j

        if y_i == 0:
            denom_0 += n_i
        else:
            phi_y += 1
            denom_1 += n_i

    phi_0 = (phi_0 + 1) / (denom_0 + k)
    phi_1 = (phi_1 + 1) / (denom_1 + k)
    phi_y = phi_y / H

    return (phi_0, phi_1, phi_y)
    # *** END CODE HERE ***

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """
    # *** START CODE HERE ***

    print("*******************testing*******************")

    phi_0, phi_1, phi_y = model
    H, W = matrix.shape
    labels = np.zeros([H, ])
    for i in range(H):
        print("testing "+ str(i) + " of " + str(H))
        x = matrix[i]
        # p_y_0 = 1
        # p_y_1 = 1
        p_y_0 = 0
        p_y_1 = 0
        for j in range(W):
            if x[j] > 0:
                # p_y_0 *= x[j] * phi_0[j]
                # p_y_1 *= x[j] * phi_1[j]
                p_y_0 += x[j] + np.log(phi_0[j])
                p_y_1 += x[j] + np.log(phi_1[j])

        if p_y_0 == 0 and p_y_1 == 0:
            labels[i] = 0
            continue

        p_y_0_final = p_y_0 + np.log(1 - phi_y)
        p_y_1_final = p_y_1 + np.log(phi_y)
        # p_y_0_final = p_y_0 * (1 - phi_y)/(p_y_0 * (1 - phi_y) + p_y_1 * phi_y)
        # p_y_1_final = p_y_1 * phi_y/(p_y_0 * (1 - phi_y) + p_y_1 * phi_y)

        if p_y_0_final > p_y_1_final:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels

    # *** END CODE HERE ***

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    words = []
    weights = []
    phi_0, phi_1, phi_y = model

    for word in dictionary:
        index = dictionary[word]
        words.append(word)
        weights.append(np.log(phi_1[index]/phi_0[index]))

    top_5_words = []
    for i in range(5):
        top = np.argmax(weights)
        top_5_words.append(words[top])
        del words[top]
        del weights[top]
    return top_5_words

    # *** END CODE HERE ***

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    max_accuracy = 0
    max_radius = None
    for radius in radius_to_consider:
        labels = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(labels == val_labels)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_radius = radius
    return max_radius
    # *** END CODE HERE ***

def load_dataset(csv_path):

    print("*******************loading dataset*******************")

    data = pd.read_csv(csv_path, encoding = "ISO-8859-1");
    num_data_points = len(data.index);

    train_tweets = data['Tweet content'].values[0:int(num_data_points * TRAIN_SPLIT)]
    val_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    test_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]

    train_labels = data['increase'].values[0:int(num_data_points * TRAIN_SPLIT)]
    val_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    test_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]

    return train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels
def main():
    train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels = load_dataset("final_data/compiled_data.csv")
    dictionary = create_dictionary(train_tweets)
    util.write_json('./output/dictionary', dictionary)
    train_matrix = transform_text(train_tweets, dictionary)
    val_matrix = transform_text(val_tweets, dictionary)
    test_matrix = transform_text(test_tweets, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)
    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])
    util.write_json('./output/p06_optimal_radius', optimal_radius)
    print('The optimal SVM radius was {}'.format(optimal_radius))
    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)
    svm_accuracy = np.mean(svm_predictions == test_labels)
    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


    # train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    # val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    # test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    #
    # dictionary = create_dictionary(train_messages)
    #
    # util.write_json('./output/p06_dictionary', dictionary)
    #
    # train_matrix = transform_text(train_messages, dictionary)
    #
    # np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])
    #
    # val_matrix = transform_text(val_messages, dictionary)
    # test_matrix = transform_text(test_messages, dictionary)
    #
    # naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    #
    # naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    #
    # np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)
    #
    # naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
    #
    # print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    #
    # top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)
    #
    # print('The top 5 indicative words for Naive Bayes are: ', top_5_words)
    #
    # util.write_json('./output/p06_top_indicative_words', top_5_words)
    #
    # optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])
    #
    # util.write_json('./output/p06_optimal_radius', optimal_radius)
    #
    # print('The optimal SVM radius was {}'.format(optimal_radius))
    #
    # svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)
    #
    # svm_accuracy = np.mean(svm_predictions == test_labels)
    #
    # print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()
