# To-do
# Code for testing causal reasoning ability
# 1. argparse (in main.py): hyperparameters(k-shot, ...), model, category(datasets), metric, prompt type, ...
# 2. (1) model: in model/model.py (gpt-neo, gpt-j, flan-t5, flan-t5-XXL, bart, gpt2) (2) category(datasets): in data/data.py (or by dataname.py)
#    (3) metric: in util/metric.py  (4) type of prompts: in util/prompt.py
#    (5) eval.py: Given specific dataset, prompt, and metric, return scores and generated answers (using data.py, metric.py)
#    (5) main.py: Given the type of model, eval category (set of datasets), metric, prompt type through argparse,
#        run eval.py given the arguments.