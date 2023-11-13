import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score




def process_columns(x):
    '''
    pandas.DataFrame.loc[] is a property that is used to access a group of rows and columns by label(s) or a boolean array.
    '''
    # for each row of x, if column name = "Você tem reserva de emergência? and value = Sim, replace to 1, of value = Nao, replace by 0
    x.loc[x['Você tem reserva de emergência?'] == 'Sim', 'Você tem reserva de emergência?'] = 1
    x.loc[x['Você tem reserva de emergência?'] == 'Não', 'Você tem reserva de emergência?'] = 0
    
    x.loc[x['Com que frequência você gasta mais do que ganha?'] == 'Nunca', 'Com que frequência você gasta mais do que ganha?'] = 3
    x.loc[x['Com que frequência você gasta mais do que ganha?'] == 'Às vezes', 'Com que frequência você gasta mais do que ganha?'] = 2
    x.loc[x['Com que frequência você gasta mais do que ganha?'] == 'Bem frequentemente', 'Com que frequência você gasta mais do que ganha?'] = 1
    x.loc[x['Com que frequência você gasta mais do que ganha?'] == 'Sempre', 'Com que frequência você gasta mais do que ganha?'] = 0

    x.loc[x['Com que frequência você consegue guardar o seu dinheiro?'] == 'Nunca', 'Com que frequência você consegue guardar o seu dinheiro?'] = 3
    x.loc[x['Com que frequência você consegue guardar o seu dinheiro?'] == 'Às vezes', 'Com que frequência você consegue guardar o seu dinheiro?'] = 2
    x.loc[x['Com que frequência você consegue guardar o seu dinheiro?'] == 'Bem frequentemente', 'Com que frequência você consegue guardar o seu dinheiro?'] = 1
    x.loc[x['Com que frequência você consegue guardar o seu dinheiro?'] == 'Sempre', 'Com que frequência você consegue guardar o seu dinheiro?'] = 0
    
    x.loc[x['Você já teve o nome no Serasa?'] == 'Sim', 'Você já teve o nome no Serasa?'] = 1
    x.loc[x['Você já teve o nome no Serasa?'] == 'Não', 'Você já teve o nome no Serasa?'] = 0
    
    '''
    print(x['Com que frequência você gasta mais do que ganha?'])

    print(x['Você tem reserva de emergência?'])
    print(x['Com que frequência você consegue guardar o seu dinheiro?'] )
    print(x['Você já teve o nome no Serasa?'])
    '''

    return x
#open document

df = pd.read_csv('data.csv')

#print column names
print(df.columns)

#print number of rowns and columns
print(df.shape)
#print first 5 rows
print(df.head())

#remove first and last columns
x = df.iloc[:, 1:-1]

print(x.columns)

# get only the last column from df - the class
y = df.iloc[:, -1]

# process x - text to numbers 
x = process_columns(x)


# split test and train subsets

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.3)

# Standardize features by removing the mean and scaling to unit variance.

scaler = StandardScaler()

# Compute the mean and std to be used for later scaling.
mean_and_std = scaler.fit(trainX)

# Perform standardization by centering and scaling.
trainX_std = mean_and_std.transform(trainX)
testX_std = mean_and_std.transform(testX)

# Create the MLP

mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 500,activation = 'relu', # identity, logistic, tanh, relu
                        solver = 'adam') # also test: lbfgs

mlp_clf.fit(trainX_std, trainY)

y_pred = mlp_clf.predict(testX_std)


# feautures

print(f"score: {mlp_clf.score(testX_std,testY)}\nloss:{mlp_clf.loss_}\nnumber of training samples:{mlp_clf.t_}\nnumber of features seen during fit:{mlp_clf.n_features_in_}\nnumber of iterations: {mlp_clf.n_iter_}\nnumber of layers: {mlp_clf.n_layers_}")

# The ith element in the list represents the loss at the ith iteration.

print(mlp_clf.loss_curve_)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

loss_curve = mlp_clf.loss_curve_
# draw the curve from loss_curve using matplotlib
plt.plot(loss_curve)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()


# code to plot the confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(testY, y_pred, labels=mlp_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mlp_clf.classes_)
disp.plot()
plt.show()

"""

resources

- neural network models: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- great summary: https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141
- mlp classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.score
- train test split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform
- accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

- more hands on:
https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron
https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch

- great example: https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier/#loading-the-data
- https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier/#data-pre-processing-1

"""