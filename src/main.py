# Main script
#print('Main script running')
import analysis
import preprocess
import utils

def main():
    projectStartTime = utils.time.time()

    datasetFileName = 'Domestic violence'
    url = 'https://www.kaggle.com/api/v1/datasets/download/fahmidachowdhury/domestic-violence-against-women'
    data_raw_path = 'data/raw/'

    dataframe = utils.dataset_csv_load2DataFrame(datasetFileName, data_raw_path, url)

    print(dataframe.columns)
    dataframe.columns = dataframe.columns.str.strip()
    print(dataframe.columns)

    dataframe = preprocess.preprocess(dataframe)

    processedFileName = 'Cleaned_Domestic_Violence'
    data_processed_path = 'data/processed/'

    utils.dataframe_export_csv(dataframe, processedFileName, data_processed_path)

    dataframe = analysis.analysis(dataframe)

    projectEndTime = utils.time.time()
    print('Project runtime: {0} minutes {1} second.'.format(int(divmod(projectEndTime - projectStartTime, 60)[0]), divmod(projectEndTime - projectStartTime, 60)[1]))

if __name__ == '__main__':
    main()