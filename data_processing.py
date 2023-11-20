import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_columns(x):

    print('*** PROCESS COLUMNS ***')
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

def process_dataset(dataset_path):

    print('*** PROCESS DATASET ***')
    #open document

    df = pd.read_csv(dataset_path)

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

    # process x - text to numbers . Similar to OrdinalEncoder() and OneHotEncoder() from sklearn preprocessing
    x = process_columns(x)

    

    return x,y

def standard_scaller(train,test):

    # Standardize features by removing the mean and scaling to unit variance.

    scaler = StandardScaler()

    # Compute the mean and std to be used for later scaling.
    mean_and_std = scaler.fit(train)

    # Perform standardization by centering and scaling.
    train_std = mean_and_std.transform(train)
    #print(f'trainX_std:{train_std}')
    test_std = mean_and_std.transform(test)

    return train_std,test_std

#min max scaler: Transform features by scaling each feature to a given range.

#This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

def minmax_scaller(train,test):

    scaler = MinMaxScaler(feature_range=(-1,1))

    # Compute the min max in the range (-1,1)
    min_max = scaler.fit(train)

    #Names of features seen during fit. Defined only when X has feature names that are all strings.
    #print(min_max.feature_names_in_)
    # Perform standardization by centering and scaling.
    train_std = min_max.transform(train)
    #print(f'after scaler, trainx:{train_std}')

    testX_std = min_max.transform(test)


    #print(f'testx std:{testX_std}')

    return train_std,testX_std