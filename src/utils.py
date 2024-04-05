import random, json, torch
import numpy as np
from argparse import Namespace

from eae_utils import *
from ner_utils import *
from llm_utils import *

MMT50_LANG_CONVERT = {
    "ar": "ar_AR",
    "arabic": "ar_AR",
    "zh": "zh_CN",
    "chinese": "zh_CN",
    "en": "en_XX",
    "english": "en_XX",
    "hi": "hi_IN",
    "hindi": "hi_IN",
    "es": "es_XX",
    "spanish": "es_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    "ru": "ru_RU",
    "nl": "nl_XX",
    "it": "it_IT",
    "ja": "ja_XX",
    "pt": "pt_XX",
    "vi": "vi_VN",
    "ko": "ko_KR",
    "id": "id_ID",
    "fi": "fi_FI",
    "af": "af_ZA",
    "hi": "hi_IN",
    "bn": "bn_IN",
    "et": "et_EE",
    "fa": "fa_IR",
    "he": "he_IL",
    "ka": "ka_GE",
    "kk": "kk_KZ",
    "ml": "ml_IN",
    "mr": "mr_IN",
    "my": "my_MM",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "tr": "tr_TR",
    "ur": "ur_PK"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def load_config(args):
    config = json.load(open(args.config, 'r'))
    config.update(vars(args))
    config = Namespace(**config)
    if hasattr(config, "xgear_config"):
        xgear_config = load_xgear_config(config.xgear_config)
        return config, xgear_config
    return config, None

def load_data(config):
    if config.task.lower() == "eae":
        train_data = load_EAE_data(config.train_file, config)
    elif config.task.lower() == "ner":
        train_data = load_ner_data(config.train_file, config.src_lang, config.max_input_length)

    return train_data

def clean_text(text):
    cleaned_tokens = sep_punc(re.sub(' +', ' ', text))
    return " ".join(cleaned_tokens)

def sep_punc(text):
    new_text = "-%s-" % text
    separators = string.punctuation + string.whitespace
    separators_re = "|".join(re.escape(x) for x in separators)
    tokens = zip(re.split(separators_re, new_text), re.findall(separators_re, new_text))
    flattened = itertools.chain.from_iterable(tokens)
    cleaned_text = [x for x in flattened if x and not x.isspace()]
    return cleaned_text[1:-1]
