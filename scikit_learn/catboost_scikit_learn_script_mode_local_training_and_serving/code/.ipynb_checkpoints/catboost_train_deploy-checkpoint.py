from __future__ import print_function

import argparse
import os

from io import StringIO
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

model_file_name = 'catboost-regressor-model.dump'

if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    
    args = parser.parse_args()
    print("Got Args: {}".format(args))

    # Take the set of files and read them all into a single pandas dataframe
    train_input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(train_input_files) == 0:
        raise ValueError(
            (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in train_input_files]
    train_df = pd.concat(raw_data)

    validation_input_files = [os.path.join(args.validation, file) for file in os.listdir(args.validation)]
    if len(validation_input_files) == 0:
        raise ValueError(
            (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in validation_input_files]
    validation_df = pd.concat(raw_data)

    # Assumption is that the label is the last column
    print('building training and validation datasets')
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1:].values
    X_validation = validation_df.iloc[:, :-1].values
    y_validation = validation_df.iloc[:, -1:].values

    # define and train model
    model = CatBoostRegressor()

    model.fit(X_train, y_train, eval_set=(X_validation, y_validation), logging_level='Silent')

    # print abs error
    print('validating model')
    abs_err = np.abs(model.predict(X_validation) - y_validation)

    # print couple perf metrics
    for q in [10, 50, 90]:
        print('AE-at-' + str(q) + 'th-percentile: ' + str(np.percentile(a=abs_err, q=q)))

    # persist model
    path = os.path.join(args.model_dir, model_file_name)
    print('saving model file to {}'.format(path))
    model.save_model(path)


    print("Training Completed")


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(os.path.join(model_dir, model_file_name))
    return catboost_model


def predict_fn(input_data, model):
    print('Invoked with {} records'.format(input_data.shape[0]))

    predictions = model.predict(input_data)
    return predictions
