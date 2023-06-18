# Import Modules
import os
import string
from tqdm import tqdm
import numpy as np
import math
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
# Part B
from sklearn.naive_bayes import BernoulliNB
from fpdf import FPDF

matplotlib.use('tkagg')
plt.style.use('ggplot')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BernoulliNaiveBayes:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.prob_cn = None
        self.prob_conditional_c = None
        self.word_num = None
        return

    def fit(self, x, y):
        train_examples = len(y)
        self.word_num = len(x[0])
        y_counts = np.unique(y, return_counts=True)[1]
        self.prob_cn = (y_counts[1] + self.alpha) / (train_examples + 2 * self.alpha)

        # calculate P(x|y), the probability of word ti given it is class c
        prob_ti_given_c = np.zeros([2, self.word_num])

        # create counters for the appearance of each word for negative and positive reviews
        sum_exists_neg = 0
        sum_exists_pos = 0
        for each_word in range(self.word_num):
            for each_review in range(train_examples):
                # look in the negative reviews
                if each_review < y_counts[0]:
                    sum_exists_neg += x[each_review][each_word]
                # look in the positive reviews
                else:
                    sum_exists_pos += x[each_review][each_word]

            # calculate probabilities for p(ti | cn) and p(ti | cp) for each word
            prob_ti_given_c[0][each_word] = (sum_exists_neg + self.alpha) / (y_counts[1] + 2 * self.alpha)
            prob_ti_given_c[1][each_word] = (sum_exists_pos + self.alpha) / (y_counts[0] + 2 * self.alpha)

            # reset the counters
            sum_exists_neg = 0
            sum_exists_pos = 0

        self.prob_conditional_c = prob_ti_given_c

    def predict(self, x):
        num_of_examples = len(x)
        result = np.zeros([num_of_examples])

        for example in range(num_of_examples):
            # calculate p(x | cn) and p(x | cp)
            prob_x_given_cn = 0.0
            prob_x_given_cp = 0.0
            for each_word in range(self.word_num):
                # probability x given cn
                prob_neg_pt1 = math.pow(self.prob_conditional_c[0][each_word], x[example][each_word])
                prob_neg_pt2 = math.pow((1 - self.prob_conditional_c[0][each_word]), (1 - x[example][each_word]))
                prob_neg_mul = prob_neg_pt1 * prob_neg_pt2
                prob_x_given_cn = prob_x_given_cn + math.log2(prob_neg_mul)

                # probability x given cp
                prob_pos_pt1 = math.pow(self.prob_conditional_c[1][each_word], x[example][each_word])
                prob_pos_pt2 = math.pow((1 - self.prob_conditional_c[1][each_word]), (1 - x[example][each_word]))
                prob_pos_mul = prob_pos_pt1 * prob_pos_pt2
                prob_x_given_cp = prob_x_given_cp + math.log2(prob_pos_mul)

            # probability that the review is negative
            prob_review_neg_cn = self.prob_cn * prob_x_given_cn
            prob_review_neg_cp = (1 - self.prob_cn) * prob_x_given_cp

            prob_review_neg = prob_review_neg_cn / (prob_review_neg_cn + prob_review_neg_cp)

            # if probability that the review is negative is bigger than 0.5 then the review is negative
            # or else it is positive
            if prob_review_neg > 0.5:
                result[example] = 1
            else:
                result[example] = 0

        return result


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ********************* Functions **************************
# Read text File
def read_text_file(file_path_string):
    with open(str(file_path_string), 'r', encoding="utf8") as current_file:
        content = current_file.read()
        edited_content = edit_text(content).replace("  ", " ").lower()
        return edited_content


# Edit file's contents and remove punctuation and special characters
def edit_text(file_text):
    file_text = file_text.replace("<br>", ' ').replace("<br />", ' ').replace("\t", ' ')
    file_text = file_text.replace("\x85", ' ').replace("\x97", ' ').replace("\x96", ' ').replace("\x91", ' ')
    final_text = file_text.translate(str.maketrans(' ', ' ', string.punctuation))
    return final_text


# Load data, edit them and create the required arrays
def dataInitialization(m=10000, n=36, num_train=25000):
    # ********************* Initialization **************************
    # m = Number of most frequent words
    # n = Number of words to be omitted
    # num_train = Number of training examples

    # Number of negative training examples (equals to number of positive training examples)
    neg_num_train = num_train / 2

    # Folder Path
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, 'aclImdb')

    path_list = ['neg', 'pos']

    # Create lists of reviews
    train_reviews = []
    test_reviews = []

    # Create list of words
    words = []

    # Create an empty dictionary
    d = dict()

    # Create y_train and y_test
    y_train = []
    y_test = []
    for i in range(int(num_train)):
        if i < neg_num_train:
            y_train.append(1)
            y_test.append(1)
        else:
            y_train.append(0)
            y_test.append(0)

    # *********************** Main **************************
    # ~~~~~~~~~~~~~~~~~~~~~ Fetch data ~~~~~~~~~~~~~~~~~~~~~
    # Use each path for train data
    individual_path = os.path.join(path, 'train')
    for new_path in tqdm(path_list, desc="Fetching train data"):
        files_path = os.path.join(individual_path, new_path)
        # Change the directory
        os.chdir(files_path)
        counter = 0
        # iterate through all files
        for file in os.listdir():
            # Check whether file is in text format and whether the desirable number of training examples is reached
            if file.endswith(".txt") and counter < neg_num_train:
                file_path = os.path.join(files_path, file)
                text = read_text_file(str(file_path))
                # Split the line into words
                split_words = text.split()
                # Update the lists
                train_reviews.append(split_words)
                words.extend(split_words)
            counter += 1

    # Use each path for test data
    individual_path = os.path.join(path, 'test')
    for new_path in tqdm(path_list, desc="Fetching test data"):
        files_path = os.path.join(individual_path, new_path)
        # Change the directory
        os.chdir(files_path)
        # iterate through all files
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                file_path = os.path.join(files_path, file)
                text = read_text_file(str(file_path))
                # Split the line into words
                split_words = text.split()
                # Update the list
                test_reviews.append(split_words)

    # ~~~~~~~~~~~~~~~~~~~~~ Create the vocabulary ~~~~~~~~~~~~~~~~~~~~~
    pbar = tqdm(total=100, desc="Creating the vocabulary")
    # Iterate over each word in line
    for word in words:
        if word.isdigit():
            continue
        # Check if the word is already in dictionary
        if word in d:
            # Increment count of word by 1
            d[word] += 1
        else:
            # Add the word to dictionary with count 1
            d[word] = 1

    pbar.update(33)

    # Sort dictionary
    sorted_by_frequency = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

    pbar.update(33)

    # Find top words
    vocabulary = []
    c = 0
    for key in list(sorted_by_frequency.keys()):
        if n <= c < (m + n):
            vocabulary.append(key)
        c += 1

    pbar.update(34)

    pbar.close()

    # ~~~~~~~~~~~~~~~~~~~~~ Create binary vectors ~~~~~~~~~~~~~~~~~~~~~
    x_train_binary = list()
    x_test_binary = list()

    for text in tqdm(train_reviews, desc="Creating binary vectors for train data"):
        binary_vector = list()
        for vocab_token in vocabulary:
            if vocab_token in text:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
        x_train_binary.append(binary_vector)

    for text in tqdm(test_reviews, desc="Creating binary vectors for test data"):
        binary_vector = list()
        for vocab_token in vocabulary:
            if vocab_token in text:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
        x_test_binary.append(binary_vector)

    x_test_binary = np.array(x_test_binary)

    return y_train, y_test, x_train_binary, x_test_binary


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Statistics ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main function to handle the statistics and send them to be printed
def main():
    accuracy_a, precision_a, recall_a, f1_a, accuracy_b, precision_b, recall_b, f1_b = statisticsTables()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~ Bernoulli Naive Bayes implementation stats ~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Accuracy between Training Data and Test Data: ")
    print("Table: ")
    print_table(accuracy_a, "Accuracy", 1)
    print("Learning curve: ")
    learningCurve(accuracy_a, "Accuracy", 1)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Precision between Training Data and Test Data: ")
    print("Table: ")
    print_table(precision_a, "Precision", 2)
    print("Learning curve: ")
    learningCurve(precision_a, "Precision", 2)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Recall between Training Data and Test Data: ")
    print("Table: ")
    print_table(recall_a, "Recall", 3)
    print("Learning curve: ")
    learningCurve(recall_a, "Recall", 3)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F1 between Training Data and Test Data: ")
    print("Table: ")
    print_table(f1_a, "F1", 4)
    print("Learning curve: ")
    learningCurve(f1_a, "F1", 4)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~ Bernoulli Naive Bayes scikit-learn stats ~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Accuracy between Training Data and Test Data: ")
    print("Table: ")
    print_table(accuracy_b, "Accuracy", 5)
    print("Learning curve: ")
    learningCurve(accuracy_b, "Accuracy", 5)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Precision between Training Data and Test Data: ")
    print("Table: ")
    print_table(precision_b, "Precision", 6)
    print("Learning curve: ")
    learningCurve(precision_b, "Precision", 6)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Recall between Training Data and Test Data: ")
    print("Table: ")
    print_table(recall_b, "Recall", 7)
    print("Learning curve: ")
    learningCurve(recall_b, "Recall", 7)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("F1 between Training Data and Test Data: ")
    print("Table: ")
    print_table(f1_b, "F1", 8)
    print("Learning curve: ")
    learningCurve(f1_b, "F1", 8)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    generate_pdf()
    delete_trash_files()


# Create the arrays for the statistics, call both the implemented and the pre-made by scikit-learn Bernoulli
# Naive Bayes. Fill the arrays with the calculated data and return them.
def statisticsTables():
    print("Running accuracyTable()..")
    max_num_train = 25000

    accuracies_a = np.zeros([2, 10])
    precisions_a = np.zeros([2, 10])
    recalls_a = np.zeros([2, 10])
    f1s_a = np.zeros([2, 10])
    accuracies_b = np.zeros([2, 10])
    precisions_b = np.zeros([2, 10])
    recalls_b = np.zeros([2, 10])
    f1s_b = np.zeros([2, 10])

    for i in range(0, 10):
        print("Round " + str(i + 1) + ":")
        accuracies_a[0][i] = (i + 1) * 10
        precisions_a[0][i] = (i + 1) * 10
        recalls_a[0][i] = (i + 1) * 10
        f1s_a[0][i] = (i + 1) * 10
        accuracies_b[0][i] = (i + 1) * 10
        precisions_b[0][i] = (i + 1) * 10
        recalls_b[0][i] = (i + 1) * 10
        f1s_b[0][i] = (i + 1) * 10

        current_train = max_num_train * (accuracies_a[0][i]) / 100
        y_train_data, y_test_data, x_train_binary_data, x_test_binary_data = dataInitialization(num_train=current_train)

        print("Initialization..")
        nb_part_a = BernoulliNaiveBayes()
        nb_part_b = BernoulliNB()
        print("Initialized")

        print("Fitting..")
        nb_part_a.fit(x_train_binary_data, y_train_data)
        nb_part_b.fit(x_train_binary_data, y_train_data)
        print("Fit")

        print("Predicting..")
        prediction_part_a = nb_part_a.predict(x_test_binary_data)
        prediction_part_b = nb_part_b.predict(x_test_binary_data)
        print("Predicted")

        accuracies_a[1][i] = accuracy(y_test_data, prediction_part_a)
        precisions_a[1][i] = precision(y_test_data, prediction_part_a)
        recalls_a[1][i] = recall(y_test_data, prediction_part_a)
        f1s_a[1][i] = f1(precisions_a[1][i], recalls_a[1][i])
        accuracies_b[1][i] = accuracy(y_test_data, prediction_part_b)
        precisions_b[1][i] = precision(y_test_data, prediction_part_b)
        recalls_b[1][i] = recall(y_test_data, prediction_part_b)
        f1s_b[1][i] = f1(precisions_b[1][i], recalls_b[1][i])
    print("Completed accuracyTable()")
    return accuracies_a, precisions_a, recalls_a, f1s_a, accuracies_b, precisions_b, recalls_b, f1s_b


# Calculate the accuracy between the training data and the testing data
def accuracy(y_test_data, prediction):
    print("Running accuracy()..")
    num_test = len(y_test_data)
    matches_counter = 0
    for i in range(0, num_test):
        if y_test_data[i] == prediction[i]:
            matches_counter += 1
    print("Completed accuracy()")
    return matches_counter / num_test


# Calculate the precision between the training data and the testing data (positive)
def precision(y_test_data, prediction):
    print("Running precision()..")
    num_test = len(y_test_data)
    all_positives = np.unique(prediction, return_counts=True)[1][0]
    true_positives_counter = 0
    for i in range(0, num_test):
        if prediction[i] == 0 and prediction[i] == y_test_data[i]:
            true_positives_counter += 1
    print("Completed precision()")
    return true_positives_counter / all_positives


# Calculate the recall between the training data and the testing data (positive)
def recall(y_test_data, prediction):
    print("Running recall()..")
    num_test = len(y_test_data)
    true_positives_counter = 0
    false_negatives_counter = 0
    for i in range(0, num_test):
        if prediction[i] == 0 and prediction[i] == y_test_data[i]:
            true_positives_counter += 1
        if prediction[i] == 1 and prediction[i] != y_test_data[i]:
            false_negatives_counter += 1
    print("Completed recall()")
    return true_positives_counter / (true_positives_counter + false_negatives_counter)


# Calculate the F1 between the training data and the testing data (positive)
def f1(precision_i, recall_i):
    print("Running f1()..")
    print("Completed f1()")
    return (2 * precision_i * recall_i) / (precision_i + recall_i)


# Print the given table of statistics and save it in a txt file
def print_table(table, stat_name, position):
    print("Running print_accuracy_table()..")
    transposed_array = table.transpose()
    print(tabulate(transposed_array, headers=["Percentage of training data", stat_name], tablefmt='fancy_grid'))

    # create a new txt file
    text_file = open('table' + str(position) + '.txt', 'w', encoding="utf-8")
    # write string to file
    text_file.write(tabulate(transposed_array, headers=["Percentage of training data", stat_name], tablefmt='github'))
    # close file
    text_file.close()

    print("Completed print_accuracy_table()")


# Print the learning curve based on the given table of statistics
def learningCurve(table, stat_name, position):
    print("Running learningCurve()..")

    plt.plot(table[0], table[1])

    # Set the title
    if position < 5:
        plt.title('Bernoulli NB implemented')
    else:
        plt.title('Bernoulli NB scikit-learn')

    # Add labels to the axes
    plt.xlabel("Percentage of Training Data")
    plt.ylabel(stat_name)

    # Set the limits of the axis
    plt.ylim([0, 1])
    plt.xlim([0, 100])

    # Save plot as a png image
    plt.savefig("learning-curve" + str(position) + ".png")

    # Display the plot
    plt.show()

    print("Completed learningCurve()")


# Generate a pdf file containing all the results (tables and learning curves)
def generate_pdf():
    print("Running generate_pdf()..")

    # Find all generated png images
    image_list = []
    for file in os.listdir():
        if file.endswith('.png') and file.startswith("learning-curve"):
            image_list.append(file)

    # Find all generated txt files
    txt_list = []
    for file in os.listdir():
        if file.endswith('.txt') and file.startswith("table"):
            txt_list.append(file)

    # save FPDF() class into a variable pdf
    pdf = FPDF()
    # set style and size of font
    pdf.set_font("Arial", size=8)

    for i in range(8):
        # Add a page
        pdf.add_page()
        if i == 0:
            pdf.cell(w=200, h=10, txt='Bernoulli Naive Bayes implementation', border=1, ln=1, align='C', fill=False)
            pdf.cell(w=200, h=5, txt="\n", ln=1, align='C')
        elif i == 4:
            pdf.cell(w=200, h=10, txt='Bernoulli Naive Bayes scikit-learn', border=1, ln=1, align='C', fill=False)
            pdf.cell(w=200, h=5, txt="\n", ln=1, align='C')

        # Find the correct files
        current_txt = None
        current_img = None
        for text in txt_list:
            if str(i + 1) in text:
                current_txt = text
        for img in image_list:
            if str(i + 1) in img:
                current_img = img

        # Add the table from the txt file to the pdf
        pdf.cell(w=200, h=5, txt="Table:\n", ln=1, align='C')
        # open the text file in read mode
        f = open(current_txt, "r")
        # insert the texts in pdf
        for x in f:
            pdf.cell(w=200, h=5, txt=x, ln=1, align='C')

        pdf.cell(w=200, h=5, txt="", ln=1, align='C')
        pdf.cell(w=200, h=5, txt="", ln=1, align='C')

        # Add the image to the pdf
        pdf.cell(w=200, h=5, txt="\nLearning curve:\n", ln=1, align='C')
        pdf.image(current_img, None, None, 210)

    # save the pdf with name results.pdf
    pdf.output("results.pdf", "F")

    print("Completed generate_pdf()")


# Delete all the unnecessary files that were created earlier
def delete_trash_files():
    print("Running delete_trash_files()..")

    for file in os.listdir():
        if (file.endswith('.txt') and file.startswith("table")) or (file.endswith('.png') and file.startswith("learning-curve")):
            os.remove(file)

    print("Completed delete_trash_files()")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main()
