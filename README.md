# CLaP
Code for our NAACL-2024 paper "Contextual Label Projection for Cross-Lingual Structured Prediction"
--

## Setup

We use `Python=3.8` for the setup. You can use the following command to setup the corresponding conda environment
```
conda env create -f env.yml
```

## Data and Preprocessing

For EAE, we utilize the ACE dataset and refer to the pre-processing from [XGear](https://github.com/PlusLabNLP/X-Gear/tree/main/preprocessing). You can save the data in `data/processed_data/eae` or change the path location accordingly in the config files. A sample example describing the data format is shown below
```
"doc_id": "CNN_CF_20030303.1900.02",
"wnd_id": "CNN_CF_20030303.1900.02_8",
"entity_mentions": [
    {
        "id": "CNN_CF_20030303.1900.02_8-E0",
        "start": 4, 
        "end": 5, 
        "entity_type": "PER", 
        "mention_type": "UNK", 
        "text": "director"
    }, 
    ...
],
"event_mentions": [
    {
        "event_type": "Personnel:End-Position", 
        "id": "CNN_CF_20030303.1900.02_8-EV0", 
        "trigger": {
            "start": 11, "end": 12, "text": "former"
        }, 
        "arguments": [
            {
                "entity_id": "CNN_CF_20030303.1900.02_8-E3", "text": "Bob Dornan", "role": "Person"
            }, 
            ...
        ]
    }
],
"tokens": ["He", "'s", "now", "national", "director", ...],
"sentence": "He 's now national director of Win Without War , ..."
```

For NER, we utilize the WikiAnn and MasakhaNER datasets. We use the original scripts from [XTREME](https://github.com/google-research/xtreme?tab=readme-ov-file#download-the-data) and [MasakhaNER](https://huggingface.co/datasets/masakhane/masakhaner2) to download the data. You can save the data in `data/processed_data/ner` or change the path location accordingly in the config files. A sample example describing the data format is shown below
```
Karl	B-PER
Ove	I-PER
Knausgård	I-PER
(	O
born	O
1968	O
)	O
```

## Running CLaP

CLaP comprises two major steps:

**STEP 1:** The first step is the basic translation of the input sentence to the target language. We provide code to use `many-to-many-large-50` for the translation (although you can use any other HuggingFace model as well). You can run this step as
```
CUDA_VISIBLE_DEVICES=0 python src/translate.py -c [translate_config_file]
```

For example, for EAE in Chinese, you can use
```
CUDA_VISIBLE_DEVICES=0 python src/translate.py -c eae_config/translate/m2m50-large-en2zh.json
```

**STEP 2:** The second step involves contextual machine translation step to contextually translate the labels from the souce to the target language. You can run this step as
```
CUDA_VISIBLE_DEVICES=0,1 python src/contextual_translate.py -c [contextual_config_file]
```

For example, for NER in Arabic, you can use
```
CUDA_VISIBLE_DEVICES=0,1 python src/contextual_translate.py -c ner_config/contextual/llama2_2shot-en2ar.json
```

We provide sample config files for both steps for each task in the `eae_config` and `ner_config` folders respectively. All arguments for the training can be altered from the config file. Many languages for NER are not compatible with many-to-many-large-50 translation model, but we include it for reference to use with other translation models.

**Note**: If you plan to use an external MT/LLM system for step 1/2, make sure to adhere to the output file formats for future processing.

## Downstream Evaluation

### Event Argument Extraction (EAE)

**Training:** To train the model, you can run
```
CUDA_VISIBLE_DEVICES=0 python src/xgear/train.py -c [xgear_train_config_file]
```

For example, for Chinese,
```
CUDA_VISIBLE_DEVICES=0 python src/xgear/train.py -c eae_config/xgear/config_XGear_EAE_ace05_en_mBART50_en-zh.jsonnet
```

**Evaluation:** For cross-lingual evaluation, you'll need to create config file for the corresponding language. Once created, you can run
```
CUDA_VISIBLE_DEVICES=0 python src/xgear/evaluate_crosslingual.py --test_config [xgear_test_config_file] --eae_model [saved_eae_model]
```

For example, for Chinese,
```
CUDA_VISIBLE_DEVICES=0 python src/xgear/evaluate_crosslingual.py --test_config eae_config/xgear/config_XGear_EAE_ace05_zh_mBART50.jsonnet --eae_model ./outputs/XGear_EAE_ace05_en_mBART50/s0/20240404_174913553/best_model_dev0.state
```

### Named Entity Recognition (NER)

**Pre-processing:** For XTREME, files need to be pre-processed. You can preprocess the CLaP generated files as follows
```
python src/xtreme/third_party/preprocess_panx.py --input_file [input_file] --output_file [output_file] --model_name_or_path [model_name] --max_len [max_length]
```

For example, for Arabic,
```
python src/xtreme/third_party/preprocess_panx.py --input_file ./data/translated_processed_data_ner/llama2_2shot_contextual-ar-translate/trans_ar_train.txt --output_file ./data/translated_processed_data_ner/llama2_2shot_contextual-ar-translate/trans_ar_train_processed.txt --model_name_or_path xlm-roberta-large --max_len 128
```

**Training:** To train the downstream model, you can use
```
bash src/xtreme/scripts/train_panx.sh [model_name] [gpu] [tgt_lang] [translate_train_file] ...
```

For example, for Arabic,
```
bash src/xtreme/scripts/train_panx.sh xlm-roberta-large 0 ar ./data/translated_processed_data_ner/llama2_2shot_contextual-ar-translate/trans_ar_train_processed.txt
```

**Evaluation:** For evaluation, you can use
```
python src/xtreme/third_party/evaluate_panx.py -g [gold_file] -p [pred_file]
```

For example, for Arabic,
```
python src/xtreme/third_party/evaluate_panx.py -g ./data/ner/panx/panx_processed_maxlen128/ar/test.xlm-roberta-large -p ./outputs_ner/panx/xlm-roberta-large-LR2e-5-epoch5-MaxLen128-TT_lang\[ar\]/test_ar_predictions.txt
```

## Citation

If you use this model or find our work useful in your research, please cite our paper.
```
@inproceedings{parekh2024contextual,
    title={Contextual Label Projection for Cross-Lingual Structured Prediction},
    author={Tanmay Parekh and I-Hung Hsu and Kuan-Hao Huang and Kai-Wei Chang and Nanyun Peng},
    booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    year={2024},
}
```

If you experiment using XGear on EAE, please also cite
```
@inproceedings{acl2022xgear,
    author    = {Kuan-Hao Huang and I-Hung Hsu and Premkumar Natarajan and Kai-Wei Chang and Nanyun Peng},
    title     = {Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year      = {2022},
}
```

If you experiment using XTREME on NER, please also cite
```
@article{hu2020xtreme,
      author    = {Junjie Hu and Sebastian Ruder and Aditya Siddhant and Graham Neubig and Orhan Firat and Melvin Johnson},
      title     = {XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}
```

## Contact

If you have any issues, please contact Tanmay Parekh (tparekh@g.ucla.edu)
