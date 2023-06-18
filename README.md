# Bernoulli_Naive_Bayes_Classifier

AI machine learning, Bernoulli NB implimentation. 

Using the 'Large Movie Review Dataset', also known as "IMDB dataset" (https://ai.stanford.edu/~amaas/data/sentiment/). 

----

Includes a pdf file showcasing the results of the experiments performed with the dataset:

• learning curves and corresponding tables showing the accuracy between the training data (training data, what has been used each time) and test data based on the number of training examples that are used in each repetition of the experiment,

• learning curves and tables with precision, recall, F1, based on the number of training examples.

It also includes the comparison of the performance of the implementation with the performance of Scikit-learn, construncting the same learning curves and tables as above

----

Note: 

• Before running the file you need to have installed the following libraries: tqdm, numpy, tabulate, matplotlib, sklearn, fpdf

• In addition, the IMDB dataset folder needs to be downloaded, unzipped and the folder "aclImdb" has to be place in the same directory as the .py file.
