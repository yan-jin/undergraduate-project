from preprocess_dataset import process_path
from preprocess_dataset import combine_files

datasetRaw_path = '../datasetRaw'
datasetProcessed_path = '../datasetProcessed'

if __name__ == '__main__':
    process_path(datasetProcessed_path, datasetRaw_path)
    combine_files(datasetProcessed_path, datasetProcessed_path)
