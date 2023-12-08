from sklearn.neural_network import MLPClassifier
import plotting 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict
from statistics import mean
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def cross_validate_mlp(mlp, X, y, cv=10):
    """
    Perform K-Fold cross-validation on the given MLPClassifier.

    Parameters:
    mlp (MLPClassifier): The MLPClassifier to validate.
    X (array-like): The input data.
    y (array-like): The target data.
    cv (int, optional): The number of folds in K-Fold CV. Defaults to 5.

    Returns:
    array of float: The cross-validation scores.
    """
    scores = cross_val_score(mlp, X, y, cv=cv)
    
    #print(f'predict: {predict}')
    #print(f'labels: {y}')

    return scores

def solvers_activations(trainX_scaled=[], trainY=[], testX_scaled=[], testY=[],cross_validation=False,x=[],y=[]):

    solvers = ['adam', 'lbfgs']
    activation_functions = ['identity','relu','logistic', 'tanh']

    print(f'trainy: {trainY}\ntesty: {testY}\n\n')

    scores = []
    
    #best_loss = []
    #validation_scores = []
    for solver in solvers:
        for activation in activation_functions:
            # Create the MLP

            mlp = MLPClassifier(hidden_layer_sizes=(200,150,100),
                                    max_iter = 2000,activation = activation, # identity (for 200,150,100 layer, close to zero at 20th it, relu at 40), logistic, tanh, relu
                                    solver = solver) # adam,lbfgs
            
            if cross_validation:
                title = f'{solver} | {activation}'
                cross_val_score = cross_validate_mlp(mlp,x, y, cv=10 )
                score = f"{title} | Score: {cross_val_score} | AVG: {mean(cross_val_score)}"
                scores.append(score)
                #plotting.plot_cross_validation_scores(scores, title)
                pred = cross_val_predict(mlp, x, y, cv=10)
                print(title)
                cm = confusion_matrix(y, pred, labels=['Responsável','Irresponsável','Intermediário'])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Responsável','Irresponsável','Intermediário'])
    
                disp.plot()
                plt.show()
                
            else:
                # Fit the model to data matrix X and target(s) y. Returns a trained MLP model.
                mlp.fit(trainX_scaled, trainY)
                
                # Predict using the multi-layer perceptron classifier.
                pred = mlp.predict(testX_scaled)
                print(f'pred: {pred}')
                title = f'{solver} | {activation}'
                score = f"{title} | Score: {mlp.score(testX_scaled,testY)}"
                #best_loss_item = f"{solver} | {activation} | best_loss: {mlp.best_loss_}"
                #validation_scores_item = f"{solver} | {activation} | validation_scores: {mlp.validation_scores_}"
                scores.append(score)
                #best_loss.append(best_loss_item)
                #validation_scores.append(validation_scores_item)
             

                """
                try:
                    plotting.plot_loss_curve(mlp, title)
                except:
                    pass
                

                try:
                    plotting.plot_confusion_matrix(testY, pred, mlp, title)
                except:
                    pass
                """

            
    return scores

# add errors

