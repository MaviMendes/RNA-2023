import data_processing
import plotting 
import avaliation

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

def holdout():

    x,y = data_processing.process_dataset('data.csv')

    # split test and train subsets. Split arrays or matrices into random train and test subsets. Quick utility that wraps input validation, next(ShuffleSplit().split(X, y))

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.3)

    print(f"trainX: {trainX.shape}, trainY: {trainY.shape},testX: {testX.shape}, testY: {testY.shape}")

    # Scale (normalization)

    trainX_scaled, testX_scaled = data_processing.minmax_scaller(trainX, testX)

    #trainX_scaled, testX_scaled = data_processing.standard_scaller(trainX, testX)
    
    scores = avaliation.solvers_activations(trainX_scaled, trainY, testX_scaled, testY)
    
    print('Holdout scores')
    for score in scores:
        print(score)

    #for loss in bl:
        #print(loss)

    #for validation in vs:
        #print(validation)

holdout()