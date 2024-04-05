local task = "EAE";
local dataset = "ace05";
local model_type = "XGear";
local train_lang = "chinese";
local pretrained_model_name = "facebook/mbart-large-50";
local pretrained_model_alias = {
    "facebook/mbart-large-50": "mBART50"
};
local langauge_alias = {
    "english": "en",
    "chinese": "zh",
    "arabic": "ar",
    "spanish": "es"
};

{
    // general config
    "task": task, 
    "dataset": dataset,
    "model_type": model_type, 
    "lang": train_lang, 
    "gpu_device": 0, 
    "seed": 0, 
    "cache_dir": "./cache", 
    "output_dir": "./outputs/%s_%s_%s_%s_%s" % [model_type, task, dataset, langauge_alias[train_lang], pretrained_model_alias[pretrained_model_name]], 
    "train_file": "./data/processed_data/%s_%s_mT5/train.json" % [dataset, langauge_alias[train_lang]],
    "dev_file": "./data/processed_data/%s_%s_mT5/dev.json" % [dataset, langauge_alias[train_lang]],
    "test_file": "./data/processed_data/%s_%s_mT5/test.json" % [dataset, langauge_alias[train_lang]],

    // auxiliary data
    "postag_type_stoi_file": "./src/models/XGear/auxiliary_files/%s/postag2id.json" % [dataset],
    "extra_info_file": "./src/models/XGear/auxiliary_files/%s/auxiliary_info.json" % [dataset],

    // model config
    "pretrained_model_name": pretrained_model_name,
    "input_style": ["triggerword", "template"],   
    "output_style": ["argument:roletype"], 
    "max_length": 350,
    "max_output_length": 100,
    
    // train config
    "max_epoch": 60,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-05,
    "weight_decay": 1e-05,
    "grad_clipping": 5.0,
    "beam_size": 4,
}

