import avaliation
from sklearn.model_selection import train_test_split
import data_processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
#https://scikit-learn.org/stable/modules/grid_search.html
#https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/

def main():

    x,y = data_processing.process_dataset('data.csv')
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(-1,1))
    mean_and_std = scaler.fit(x)
    x_std = mean_and_std.transform(x)

    scores = avaliation.solvers_activations(cross_validation=True,x=x_std,y=y)
    print('Cross validation scores')

    for score in scores:
        print(score)
    

main()