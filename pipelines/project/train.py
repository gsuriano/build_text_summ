from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Thext.SentenceRankerPlus import *
from Thext import RedundancyManager
from Thext import Highlighter
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    feature_columns_names = ["sentence","abstract"]
    rouge_label = 'r2f'
    
    logger = logging.getLogger(__name__)
    base_dir = "/opt/ml/processing"
    fn_train = f"{base_dir}/train/train.csv"
    fn_val = f"{base_dir}/validation/validation.csv"
    
    logger.debug("Reading downloaded data.")
    train = pd.read_csv(
        fn_train,
        header=None,
        names=feature_columns_names + [rouge_label],
    )

    val = pd.read_csv(
        fn_val,
        header=None,
        names=feature_columns_names + [rouge_label],
    )
    
    base_model_name = "morenolq/thext-cs-scibert"
    model_name_or_path = "morenolq/thext-cs-scibert"
    sr = SentenceRankerPlus(base_model_name=base_model_name, model_name_or_path=model_name_or_path)

    sr.set_text(train['sentence'].values,True)
    sr.set_text(val['sentence'].values,False)
    sr.set_abstract(train['abstract'].values,True)
    sr.set_abstract(val['abstract'].values,False)
    sr.set_labels(train[rouge_label].values,True)
    sr.set_labels(val[rouge_label].values,False)
    
    sr.prepare_for_training()
    
    sr.fit(args.model_dir)
