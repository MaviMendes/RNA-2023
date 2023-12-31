import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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

def plot_loss_curve(mlp):

    loss_curve = mlp.loss_curve_
    # draw the curve from loss_curve using matplotlib
    plt.plot(loss_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

def plot_confusion_matrix(testY,y_pred,mlp):

    # code to plot the confusion matrix
    cm = confusion_matrix(testY, y_pred, labels=mlp.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mlp.classes_)
    disp.plot()
    plt.show()


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



# 1 - Teste generico 

# split test and train subsets

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.3)

print(f'before scaler, trainx:{trainX.head(5)}')
# Standardize features by removing the mean and scaling to unit variance.

scaler = StandardScaler()

# Compute the mean and std to be used for later scaling.
mean_and_std = scaler.fit(trainX)

# Perform standardization by centering and scaling.
trainX_std = mean_and_std.transform(trainX)
print(f'after scaler, traninx:{trainX_std}')

testX_std = mean_and_std.transform(testX)


print(f'testx std:{testX_std}')

#min max scaler: Transform features by scaling each feature to a given range.

#This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
print('now using minmax scaler')

scaler = MinMaxScaler(feature_range=(-1,1))

# Compute the mean and std to be used for later scaling.
min_max = scaler.fit(trainX)
print(min_max.feature_names_in_)
# Perform standardization by centering and scaling.
trainX_std = min_max.transform(trainX)
print(f'after scaler, traninx:{trainX_std}')

testX_std = min_max.transform(testX)


print(f'testx std:{testX_std}')
"""

# Create the MLP

mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 500,activation = 'relu', # identity, logistic, tanh, relu
                        solver = 'adam') # also test: lbfgs

mlp_clf.fit(trainX_std, trainY)

y_pred = mlp_clf.predict(testX_std)


# features

print(f"score: {mlp_clf.score(testX_std,testY)}\nloss:{mlp_clf.loss_}\nnumber of training samples:{mlp_clf.t_}\nnumber of features seen during fit:{mlp_clf.n_features_in_}\nnumber of iterations: {mlp_clf.n_iter_}\nnumber of layers: {mlp_clf.n_layers_}")



print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

plot_loss_curve(mlp_clf)
plot_confusion_matrix(testY,y_pred,mlp_clf)

"""

"""
Dataset partitioning 

hold-out: split up your dataset into a ‘train’ and ‘test’ set

cross-validation: Cross-validation or ‘k-fold cross-validation’ is when the dataset is randomly split up into ‘k’ groups.
One of the groups is used as the test set and the rest are used as the training set. 
The model is trained on the training set and scored on the test set. 
Then the process is repeated until each unique group as been used as the test set.

https://medium.com/@jaz1/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f

https://scikit-learn.org/stable/modules/cross_validation.html

"""

#Holdout 


#cross-validation

"""

Topologies 

neurons, layers, learning rate, etc
MLPClassifier  parameters better explored


"""

"""
Error analysis- loss function

"""

# The ith element in the list represents the loss at the ith iteration.

# Erro médio absoluto ou MAE (do inglês Mean AbsoluteError)

# Erro quadrático médio ou MSE (MeanSquared Error)

# Raiz quadrada do erro quadrático médio ou RMSE (Root Mean Squared Error)

# Coeficiente de determinação ou R2 (coefficient of determination)

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