# Preprocessing script
#def preprocess():
    #print('Data preprocessing')
import utils

def find_outliers_IQR(column):
   q1 = column.quantile(0.25)
   q3 = column.quantile(0.75)
   IQR = q3-q1
   outliers = column[((column<(q1-1.5*IQR)) | (column>(q3+1.5*IQR)))]
   not_outliers = column[~((column<(q1-1.5*IQR)) | (column>(q3+1.5*IQR)))]
   return outliers, not_outliers

def find_outliers(dataframe):
    dataframe['Age'].plot.box()
    #utils.matplotlib.pyplot.savefig("squares.png")

    outliers = find_outliers_IQR(dataframe['Age'])[0]
    print('number of outliers: '+ str(len(outliers))) 

    dataframe['Education'].value_counts().plot.bar()

    dataframe['Education'].value_counts()

    dataframe['Employment'].value_counts().plot.bar()

    dataframe['Employment'].value_counts()

    dataframe['Employment'] = dataframe['Employment'].str.strip()
    dataframe['Employment']

    dataframe['Employment'].value_counts().plot.bar()

    dataframe['Employment'].value_counts()

    dataframe['Income'].plot.box()

    outliers = find_outliers_IQR(dataframe['Income'])[0]
    print('number of outliers: '+ str(len(outliers)))

    not_outliers = find_outliers_IQR(dataframe['Income'])[1]
    not_outliers

    #dataframe['Income'] = dataframe['Income'].drop(outliers.index)
    #dataframe.dropna().reset_index()

    dataframe['Marital status'].value_counts().plot.bar()

    dataframe['Marital status'].value_counts()

    dataframe['Violence'].value_counts().plot.bar()

    dataframe['Violence'].value_counts()

    return dataframe

def find_dataType(dataframe):
    dataframe.info()

    #dataframe['Age'].dtype

    return dataframe

def data_encoding(dataframe):
    education_dict = {
    'primary': 1,
    'secondary': 2,
    'none': 0,
    'tertiary': 3
    }

    employment_dict = {
        'unemployed': 0,
        'semi employed': 2,
        'employed': 1
    }

    marital_status_dict = {
        'married': 1,
        'unmarred': 0
    }

    violence_dict = {
        'no': 0,
        'yes': 1
    }

    dataframe_map = dataframe['Education'].map(education_dict)
    dataframe.loc[:, 'Education'] = dataframe_map
    dataframe

    dataframe_map = dataframe['Employment'].map(employment_dict)
    dataframe.loc[:, 'Employment'] = dataframe_map
    dataframe

    dataframe_map = dataframe['Marital status'].map(marital_status_dict)
    dataframe.loc[:, 'Marital status'] = dataframe_map
    dataframe

    dataframe_map = dataframe['Violence'].map(violence_dict)
    dataframe.loc[:, 'Violence'] = dataframe_map
    dataframe

    dataframe = dataframe.astype({'Education': 'int64', 'Employment': 'int64', 'Marital status': 'int64', 'Violence': 'int64'})
    dataframe.info()

    return dataframe

def preprocess(dataframe):

    print(dataframe.isnull().sum())

    dataframe = find_outliers(dataframe)

    dataframe.duplicated().sum()

    dataframe = find_dataType(dataframe)

    dataframe = data_encoding(dataframe)
    
    return dataframe
