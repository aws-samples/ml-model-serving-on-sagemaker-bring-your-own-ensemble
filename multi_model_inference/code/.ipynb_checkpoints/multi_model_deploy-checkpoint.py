#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os

from io import StringIO
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

import joblib
import json
import logging
import sys

import pickle
from my_custom_library import cross_validation

import csv
import pickle
from my_custom_library import cross_validation
from sagemaker_containers import _content_types
import xgboost as xgb


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

model_file_name = 'catboost-regressor-model.dump'



if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--num_round", type=int, default=os.environ.get("SM_HP_num_round"))
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--k_fold", type=int, default=5)
    
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
    
    
    
    """
    Train the XGBoost model
    """

    K = args.k_fold

    hyperparameters = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "num_round": args.num_round,
    }


    rmse_list, model = cross_validation(train_df, K, hyperparameters)
    k_fold_avg = sum(rmse_list) / len(rmse_list)
    print(f"RMSE average across folds: {k_fold_avg}")

    model_location = args.model_dir + "/xgboost-model"
    pickle.dump(model, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


    print("Training Completed")



def input_fn(input_data, content_type):
    dtype=None
    payload = StringIO(input_data)
    
    return np.genfromtxt(payload, dtype=dtype, delimiter=",")


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(os.path.join(model_dir, model_file_name))
    
    model_file = "xgboost-model"
    model = pickle.load(open(os.path.join(model_dir, model_file), "rb"))
    
    all_model = [catboost_model, model]
    return all_model


def predict_fn(input_data, model):

    print('Invoked with {} records'.format(input_data.shape[0]))

#     predictions_catb = model[0].predict(input_data)
    
#     decoded_payload = input_data.strip().decode("utf-8")
#     dtest = encoder.csv_to_dmatrix(decoded_payload, dtype=np.float)
#     predictions_xgb = model[1].predict(dtest,
#                                           ntree_limit=getattr(model, "best_ntree_limit", 0),
#                                           validate_features=False)
#     return [predictions_catb, predictions_xgb]

#     print('Invoked with {} records'.format(input_data.shape[0]))
#     np_array = decoder.decode(input_data, content_type)
#     reader = csv.reader(input_data, delimiter=',')
    predictions_catb = model[0].predict(input_data)
    print("catboost results:")
    print(predictions_catb)

    
#     csv_string = input_data.decode() if isinstance(input_data, bytes) else input_data
#     sniff_delimiter = csv.Sniffer().sniff(csv_string.split('\n')[0][:512]).delimiter
#     delimiter = ',' if sniff_delimiter.isalnum() else sniff_delimiter
#     logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))

#     np_payload = np.array(list(map(lambda x: _clean_csv_string(x, delimiter),     
#                                    csv_string.split('\n')))).astype(dtype)
    dtest = xgb.DMatrix(input_data)
    predictions_xgb = model[1].predict(dtest,
                                          ntree_limit=getattr(model, "best_ntree_limit", 0),
                                          validate_features=False)
    print("xgboost results:")
    print(predictions_xgb)
    
    
    return np.mean(np.array([predictions_catb, predictions_xgb]), axis=0)


