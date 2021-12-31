# Machine learning example using iris dataset
# Classification problem.
# Uses a variety of different algorithms to predict class based on sepal/petal lengths and widths

# Python version 3.6
# Source: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Step 1: Check Python versions
# Step 2: Load libraries
# Step 3: Load dataset
# Step 4: Summarise data
# Step 5: Visualise data
# Step 6: Evaluate algorithms
# Step 7: Make predictions


#######################
########  Step 1 ########
#######################

# Check versions of the libraries

# Python
import sys
print("Python: {}".format(sys.version))
# SciPy
import scipy
print("scipy: {}".format(scipy.__version__))
# NumPy
import numpy
print('numpy: {}'.format(numpy.__version__))
# MatplotLib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# Pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# Scikit-Learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))      


#######################
########  Step 2 ########
#######################

# Load libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#######################
########  Step 3 ########
#######################

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


#######################
########  Step 4 ########
#######################

# Summarise dataset
print("\n", dataset.shape) # Number of rows and columns
print("\n", dataset.head(15)) # Select first 15 rows to take a peek at the data
print("\n", dataset.describe()) # Summary statistics of data
print("\n", dataset.groupby('class').size()) # Get rows in each class


#######################
########  Step 5 ########
#######################

# Visualise the data
# box plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histogram
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()


#######################
########  Step 6 ########
#######################

# Evaluate algorithms
# Get test and validation set
array = dataset.values
X = array[:,0:4] # Select Sepal and Petal lengths and widths
Y = array[:,4] # Select class
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size = validation_size, random_state = seed)

# Test Harness
seed = 7
scoring = 'accuracy'

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
kFoldsplits = 10
print("")

for name, model in models:
    kfold = model_selection.KFold(n_splits = kFoldsplits, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print("{}:     {}:     ({})".format(name, format(cv_results.mean(), '.3f'), format(cv_results.std(), '.3f')))

# Compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#######################
########  Step 7 ########
#######################

# Making predictions
SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
acc_score = accuracy_score(Y_validation, predictions)

print("\n", "Accuracy of SVM model: {}".format(format(acc_score,'.3f')))
print("\n", "Confusion matrix: \n", confusion_matrix(Y_validation, predictions))
print("\n", "Classification report: \n", classification_report(Y_validation, predictions))
