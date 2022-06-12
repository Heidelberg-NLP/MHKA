# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """
'''
Much of this code is taken from HuggingFace's repo:
https://github.com/huggingface/transformers/tree/master/examples
'''
from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
TOKEN_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}
_, _, tokenizer_bert = TOKEN_CLASSES['openai-gpt']
tokenizer_bert_base = tokenizer_bert.from_pretrained('openai-gpt', do_lower_case=True,cache_dir=None,)

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question,  contexts, middle, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (qustion).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.middle = middle
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features, 
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'knowledge_ids': knowledge_ids,
                'knowledge_mask': knowledge_mask,
                }
            for input_ids, input_mask, segment_ids, knowledge_ids, knowledge_mask in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class ANLIProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
        
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f,delimiter='\t', quoting=csv.QUOTE_NONE)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        
        examples = [
            InputExample(
                example_id=int(line[0]),
                
                question = [line[5]+line[7]+line[6], line[5]+line[8]+line[6]],
                contexts = [line[1], line[1]],
                middle = [line[3], line[4]],
                endings = [line[2], line[2]],
                label=str(int(line[-1])-1)
            ) for line in lines[0:]  # we skip the line with the column names
        ]
        
        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    know_length: int,
    tokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
)-> List[InputFeatures]:
    """    
    Loads a data file into a list of `InputFeatures`
    """
    know_length = 1024 
    label_map = {label : i for i, label in enumerate(label_list)}
    length = []
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        choices_features = []
        tokens = []
        label_ids = []

        relation_label = []
        for ending_idx, (question, context, middle, ending) in enumerate(zip(example.question, example.contexts, example.middle, example.endings)):
            
            tokens_a = tokenizer.tokenize(context)
            tokens_m = tokenizer.tokenize(middle)
            tokens_b = tokenizer.tokenize(ending)
             
            knowledge = tokenizer.encode_plus(question, add_special_tokens=False, max_length=know_length,)
            knowledge_ids = knowledge["input_ids"]
            length.append(len(knowledge_ids))
            knowledge_attention_mask = [1] * len(knowledge_ids)
            padding_know = [0] * (know_length - len(knowledge_ids))
            knowledge_ids += padding_know
            knowledge_attention_mask += padding_know 
            
            assert len(knowledge_ids) == know_length
            assert len(knowledge_attention_mask) == know_length
            
            tokens_mb = tokens_m + tokens_b
            _truncate_seq_pair(tokens_a, tokens_mb, max_length - 3)
            
            tokens = ["[CLS]"] + tokens_a + tokens_m + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + len(tokens_m) + 2) + [1] * (len(tokens_b)+ 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            #print(len(input_ids), max_length)
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            attention_mask += padding
            segment_ids += padding
            
            #print(len(knowledge_ids), know_length) 
            
            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(segment_ids) == max_length
            choices_features.append((input_ids, attention_mask, segment_ids, knowledge_ids, knowledge_attention_mask))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids, knowledge_ids, knowledge_attention_mask) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("knowledge_ids: {}".format(' '.join(map(str, knowledge_ids))))  
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))
    #print(max(length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            logger.info('Attention! you are removing from token_b (anli task is ok). '
                        'If you are training ARC and RACE (you are poping question + options), '
                        'you need to try to use a bigger max seq length!')
            tokens_b.pop()


processors = {
    "anli": ANLIProcessor
}


GLUE_TASKS_NUM_LABELS = {
    "race", 4,
    "swag", 2,
    "arc", 4,
    "anli", 2
}
