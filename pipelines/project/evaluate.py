"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
from Thext import SentenceRankerPlus
from Thext import RedundancyManager
from Thext import Highlighter
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def compute_rogue_single_doc(sentences, highlights):

  predicted_highlights_concat = ' '.join(map(str, sentences))
  real_highlights_concat =  highlights

  r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)
  score = r_computer.get_scores(predicted_highlights_concat,real_highlights_concat)

  return score['rouge-1']['f'],score['rouge-2']['f'], score['rouge-l']['f']

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/input/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    test_path = "/opt/ml/data/test/test.csv"
    feature_columns_names = ["sentence","abstract"]
    rouge_label = 'r2f'
    
    dataset = pd.read_csv(
        test_path,
        header=None,
        names=feature_columns_names + [rouge_label],
    )
    

    base_model_name = "morenolq/thext-cs-scibert"
    model_name_or_path = "/opt/ml/processing/input/model/"
    sr = SentenceRankerPlus()
    sr.base_model_name = base_model_name
    sr.load_model (base_model_name=base_model_name, model_name_or_path = model_name_or_path)
    
    h = Highlighter(sr)
    
    num_highlights = 3
    
    rougue1_f = np.array([])
    rougue2_f = np.array([])
    rouguel_f = np.array([])
    
    
    for row in dataset:
    
        sentences = sent_tokenize(row['article'])
        highlights = row['highlights']
        
        max_length = max([ len(word_tokenize(s)) for s in sentences])
        
        size = 0
        max_size = 384 -2 - max_length
        
        abs = ''
        for s in sentences :
            to_add = len(word_tokenize(s))
            if size + to_add < max_size :
              abs += s
              size += to_add
            else :
              break
        
        sentences = h.get_highlights_simple(sentences, abs,
                      rel_w=1.0,
                      pos_w=0.0,
                      red_w=0.0,
        
                      prefilter=False,
                      NH = num_highlights)
        
        
        
        r1f,r2f,rlf = compute_rogue_single_doc(sentences, highlights)
        
        rougue1_f = np.append(rougue1_f,r1f)
        rougue2_f = np.append(rougue2_f,r2f)
        rouguel_f = np.append(rouguel_f,rlf)
    
    report_dict = {
        "rouge_metrics": {
            "r1f": {
                "value": np.average(rougue1_f)
            },
            "r2f": {
                "value": np.average(rougue2_f)
            },
            "rlf": {
                "value": np.average(rouguel_f)
            },
        },
    }

    output_dir = "/opt/ml/processing/output/metrics"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
