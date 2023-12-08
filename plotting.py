from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_loss_curve(mlp, title):

    print('*** PLOT LOSS CURVE ***')
    loss_curve = mlp.loss_curve_
    # draw the curve from loss_curve using matplotlib

    plt.plot(loss_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {title}')
    plt.show()

def plot_confusion_matrix(testY,y_pred,mlp, title):

    print(f'*** PLOT CONFUSION MATRIX - {title}***')
    
    # code to plot the confusion matrix
    cm = confusion_matrix(testY, y_pred, labels=mlp.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mlp.classes_)
    # set title to confusion matrix 
    
    disp.plot()
    plt.show()


def plot_cross_validation_scores(scores, title):

    print('* PLOT CROSS VALIDATION SCORES *')
    # code to plot the cross validation scores
    plt.plot(scores)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f'Cross Validation Scores - {title}')
    plt.show()