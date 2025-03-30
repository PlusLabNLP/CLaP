from argparse import ArgumentParser
from tqdm import tqdm
import os, json, torch
from collections import namedtuple

from utils import *

class M2M_MODEL():

    def __init__(self, config):
        if config.model_name == "facebook/mbart-large-50-many-to-many-mmt":
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            self.tokenizer = MBart50TokenizerFast.from_pretrained(config.model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(config.model_name)
            self.mmt_lang_convert = MMT50_LANG_CONVERT

        self.max_length = config.max_length
        self.tokenizer.src_lang = self.mmt_lang_convert[config.src_lang]
        self.src_lang = self.tokenizer.src_lang
        self.tgt_lang = self.mmt_lang_convert[config.tgt_lang]
        self.model.cuda()

        _dummy = self.tokenizer(self.tokenizer.pad_token)['input_ids']
        self.SRC_TOKEN = _dummy[0]
        self.EOS_TOKEN = _dummy[2]
        self.PAD_TOKEN = _dummy[1]
    
    def change_src_lang(self, new_src_lang):
        self.tokenizer.src_lang = self.mmt_lang_convert[new_src_lang]
        self.src_lang = self.tokenizer.src_lang
    
    def change_tgt_lang(self, new_tgt_lang):
        self.tgt_lang = self.mmt_lang_convert[new_tgt_lang]

    def translate(self, dataloader):

        translated_data = []
        for batch_idx, batch in tqdm(enumerate(dataloader)):

            if self.src_lang == self.tgt_lang:
                translated_data.extend(batch.batch_text)
                continue
            
            encoded_src = self.tokenizer(batch.batch_text, return_tensors="pt", padding=True)
            encoded_src = { enc: encoded_src[enc].cuda() for enc in encoded_src }

            generated_tokens = self.model.generate(
                **encoded_src,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                max_new_tokens=self.max_length
            )
            translated_batch = self.tokenizer.batch_decode(generated_tokens.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translated_data.extend(translated_batch)

        return translated_data

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    config, xgear_config = load_config(args)

    # Set seed
    set_seed(config.seed)

    if config.task == "eae":
        
        # load data - adapt according to data format
        src_data = load_data(xgear_config)
        src_lang = config.src_lang

        # Create target folder
        if not os.path.exists(config.tgt_folder):
            os.makedirs(config.tgt_folder)

        print ("%s data..." % config.split)
        processed_src, dataloaders = process_data_translate(src_data, src_lang, xgear_config, config.batch_size, config.src_lang)

        # Save source data
        if config.save_source:
            if not os.path.exists(config.translations_folder):
                os.makedirs(config.translations_folder)
            with open("%s/source_%s.json" % (config.translations_folder, config.split), 'w') as f:
                json.dump(processed_src, f)
            with open("%s/source_%s_sentences.json" % (config.translations_folder, config.split), 'w') as f:
                json.dump([ dt["text"] for dt in processed_src[0] ], f)

        translated_data = None
        if config.load_orig_translations and os.path.exists("%s/translated_%s.json" % (config.translations_folder, config.split)):
            with open("%s/translated_%s.json" % (config.translations_folder, config.split), 'r') as f:
                translated_data = json.load(f)
        else:
            # Load model
            model = M2M_MODEL(config)

            # Independently translate text, trigger and arguments
            translated_data = []
            translated_data.append(model.translate(dataloaders[0]))         # Translate text
            translated_data.append(model.translate(dataloaders[1]))         # Translate trigger + arguments
            translated_data.append(processed_src[2])
            translated_data.append(processed_src[3])

            # Save original translations
            if config.save_orig_translations:
                if not os.path.exists(config.translations_folder):
                    os.makedirs(config.translations_folder)
                with open("%s/translated_%s.json" % (config.translations_folder, config.split), 'w') as f:
                    json.dump(translated_data, f, ensure_ascii=False)

        # Save data - adapt according to data format
        create_data_and_save_eae(processed_src, translated_data, config)
    
    elif config.task == "ner":
        
        # load data - adapt according to data format
        src_train_data = load_data(config)
        dataloader_text, dataloader_entity = process_data_translate_ner(src_train_data, config.batch_size)

        # Save source data
        if config.save_source:
            if not os.path.exists(config.translations_folder):
                os.makedirs(config.translations_folder)
            with open("%s/source_%s.json" % (config.translations_folder, config.split), 'w') as f:
                json.dump(src_train_data, f)
        
        # Create target folder
        if not os.path.exists(config.tgt_folder):
            os.makedirs(config.tgt_folder)

        translated_data = None
        if config.load_orig_translations and os.path.exists("%s/translated_%s.json" % (config.translations_folder, split)):
            with open("%s/translated_%s.json" % (config.translations_folder, config.split), 'r') as f:
                translated_data = json.load(f)
        else:
            # Load model
            if config.model_name.startswith("facebook"):
                model = M2M_MODEL(config)

            # Independently translate text, trigger and arguments
            translated_data = []
            translated_data.append(model.translate(dataloader_text))         # Translate text
            translated_data.append(model.translate(dataloader_entity))       # Translate entity
            translated_data.append(src_train_data[2])
            translated_data.append(src_train_data[3])

            # Save original translations
            if config.save_orig_translations:
                if not os.path.exists(config.translations_folder):
                    os.makedirs(config.translations_folder)
                with open("%s/translated_%s.json" % (config.translations_folder, config.split), 'w') as f:
                    json.dump(translated_data, f)

        # Save data - adapt according to data format
        create_data_and_save_ner(translated_data, config, config.split, all_entity_included=config.labels_in_sent)

if __name__ == "__main__":
    main()
