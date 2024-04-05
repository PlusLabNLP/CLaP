# python preprocess_panx_singlefile.py --input_file ../XL_AMPERE/Data/translated_processed_data_ner/llama2_2shot_contextual-fr-translate-entity-v2/trans_fr_train-en.txt --output_file ../XL_AMPERE/Data/translated_processed_data_ner/llama2_2shot_contextual-fr-translate-entity-v2/trans_fr_train-en_processed.txt --model_name_or_path xlm-roberta-large --max_len 128
import argparse
import string, re, itertools
import os
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer

TOKENIZERS = {
  'bert': BertTokenizer,
  'xlm': XLMTokenizer,
  'xlmr': XLMRobertaTokenizer,
}

def _preprocess_one_file(infile, outfile, tokenizer, max_len):
    if not os.path.exists(infile):
        print(f'{infile} not exists')
        return 0
    special_tokens_count = 3 if isinstance(tokenizer, XLMRobertaTokenizer) else 2
    max_seq_len = max_len - special_tokens_count
    subword_len_counter = 0
    with open(infile, "rt") as fin, open(outfile, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                subword_len_counter = 0
                continue

            items = line.split()
            token = items[0].strip()
            if len(items) == 2:
                label = items[1].strip()
            else:
                label = 'O'
            current_subwords_len = len(tokenizer.tokenize(token))

            if (current_subwords_len == 0 or current_subwords_len > max_seq_len) and len(token) != 0:
                token = tokenizer.unk_token
                current_subwords_len = 1

            if (subword_len_counter + current_subwords_len) > max_seq_len:
                fout.write(f"\n{token}\t{label}\n")
                subword_len_counter = current_subwords_len
            else:
                fout.write(f"{token}\t{label}\n")
                subword_len_counter += current_subwords_len
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--max_len", default=128, type=int,
                        help="the maximum length of sentences")
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                        help="The pre-trained model")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="whether to do lower case")
    args = parser.parse_args()
    
    if args.model_name_or_path == "xlm-roberta-large" or args.model_name_or_path == "xlm-roberta-base":
        model_type = "xlmr"
    tokenizer = TOKENIZERS[model_type].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    infile = args.input_file
    outfile = args.output_file
    code = _preprocess_one_file(infile, outfile, tokenizer, args.max_len)
    if code > 0:
        print(f'finish preprocessing {outfile}')
