from sklearn.neural_network import MLPClassifier
import plotting 

def solvers_activations(trainX_scaled, trainY, testX_scaled, testY):

    solvers = ['adam', 'lbfgs']
    activation_functions = ['identity','relu','logistic', 'tanh']

    scores = []
    #best_loss = []
    #validation_scores = []
    for solver in solvers:
        for activation in activation_functions:
            # Create the MLP

            mlp = MLPClassifier(verbose=True,hidden_layer_sizes=(200,150,100),
                                    max_iter = 2000,activation = activation, # identity (for 200,150,100 layer, close to zero at 20th it, relu at 40), logistic, tanh, relu
                                    solver = solver) # also test: lbfgs
            
            # Fit the model to data matrix X and target(s) y. Returns a trained MLP model.
            mlp.fit(trainX_scaled, trainY)
            
            # Predict using the multi-layer perceptron classifier.
            pred = mlp.predict(testX_scaled)
            
            score = f"{solver} | {activation} | Score: {mlp.score(testX_scaled,testY)}"
            #best_loss_item = f"{solver} | {activation} | best_loss: {mlp.best_loss_}"
            #validation_scores_item = f"{solver} | {activation} | validation_scores: {mlp.validation_scores_}"
            #scores.append(score)
            #best_loss.append(best_loss_item)
            #validation_scores.append(validation_scores_item)
            
            title = f'{solver} | {activation}'
            
            try:
                plotting.plot_loss_curve(mlp, title)
            except:
                pass
            

            try:
                plotting.plot_confusion_matrix(testY, pred, mlp, title)
            except:
                pass

            
    return scores#, best_loss

# add errors