import json, _jsonnet, re
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import DataLoader

text_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_text']
TextBatch = namedtuple('TextBatch', field_names=text_fields, defaults=[None] * len(text_fields))

def text_collate_fn(batch):
    return TextBatch(
        batch_text=[instance["text"] for instance in batch]
    )

entity_fields = ['batch_text', 'batch_source', 'batch_trans']
EntityContextualBatch = namedtuple('EntityContextualBatch', field_names=entity_fields, defaults=[None] * len(entity_fields))

def entitytext_contextual_collate_fn(batch):
    return EntityContextualBatch(
        batch_text=[instance["entitytext"] for instance in batch],
        batch_source=[instance["text"] for instance in batch],
        batch_trans=[instance["trans_text"] for instance in batch]
    )

def load_xgear_config(config_fn):
    config = json.loads(_jsonnet.evaluate_file(config_fn))
    config = Namespace(**config)
    return config

def load_EAE_data(file, config):
    
    data = []
    try:
        with open(file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
    except:
        with open(file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            data = [json.loads(line) for line in lines]
    
    dt_instances = []
    num_mentions = 0
    for dt in data:
        
        entities = dt['entity_mentions']
        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])
        entity_map = {entity['id']: entity for entity in entities}
        instances = []

        for i, event_mention in enumerate(event_mentions):
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))

            instance = {"id": event_mention["id"],
                        "trigger": trigger, 
                        "arguments": arguments}
            instances.append(instance)
            num_mentions += 1
        
        dt_instances.append({
            "doc_id": dt["doc_id"], 
            "wnd_id": dt["wnd_id"], 
            "tokens": dt["tokens"], 
            "text": dt["sentence"],
            "event_mentions": instances
        })
            
    print('Loaded {} EAE instances from {}'.format(num_mentions, file))
    return dt_instances

def get_span_idx(text, span, trigger_span=None, char_lang=False):
    """
    This function is how we map the generated prediction back to span prediction.
    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    candidates = []
    delimiter = " " if not char_lang else ""
    tokens = text.split() if not char_lang else text
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            c_string = delimiter.join(tokens[i:j+1])
            if c_string == span:
                candidates.append((i, j+1))
                break
            elif not span.startswith(c_string):
                break
                
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]

def process_data_translate(data, lang, config, batch_size, src_lang):
    text_data, entity_data, role_data, num_entities = [], [], [], []
    for i, dt in enumerate(data):
        
        text_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "text": dt["text"],
                    "event_mentions": dt["event_mentions"]
                    }
        text_data.append(text_dt)

        num_entity_per_mention = []

        for em in dt["event_mentions"]:
            entity_data.append({"text": em["trigger"][-1]})
            role_data.append("Trigger")

            for arg in em["arguments"]:
                entity_data.append({"text": arg[-1]})
                role_data.append(arg[2])
            
            num_entity_per_mention.append(len(em["arguments"]) + 1)
        
        num_entities.append(num_entity_per_mention)
            
    print ("Generate %d XGear Trigger + Argument instances from %d EAE instances" % (len(entity_data), len(data)))

    dataloader_text = DataLoader(text_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=text_collate_fn)
    dataloader_entity = DataLoader(entity_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=text_collate_fn)
    
    return (text_data, entity_data, role_data, num_entities), (dataloader_text, dataloader_entity)

def process_contextual_data_contextual(data, translated_text, lang, config, batch_size, src_lang):
    
    assert len(data) == len(translated_text), (len(data), len(translated_text))

    text_data, entity_data, role_data, num_entities = [], [], [], []
    for dt, trans_text in zip(data, translated_text):

        text_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "text": dt["text"],
                    "event_mentions": dt["event_mentions"]
                    }
        text_data.append(text_dt)

        num_entity_per_mention = []
        for em in dt["event_mentions"]:

            trigger_dt = {"entitytext": em["trigger"][-1],
                        "text": dt["text"],
                        "trans_text": trans_text
                        }
            entity_data.append(trigger_dt)
            role_data.append("Trigger")

            for arg in em["arguments"]:
                arg_dt = {"text": dt["text"],
                        "entitytext": arg[-1],
                        "trans_text": trans_text
                        }
                entity_data.append(arg_dt)
                role_data.append(arg[2])

            num_entity_per_mention.append(len(em["arguments"]) + 1)

        num_entities.append(num_entity_per_mention)
            
    print ("Generate %d XGear Trigger + Argument instances from %d EAE instances" % (len(entity_data), len(data)))

    dataloader_entity = DataLoader(entity_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=entitytext_contextual_collate_fn)
    return (text_data, entity_data, role_data, num_entities), dataloader_entity

def clean_translations(text):
    text = text.strip()
    for p in [",", "。", "”", "“", ":", "." ]:
        while text.startswith(p):
            text = text[1:]
        while text.endswith(p):
            text = text[:-1]
    return text

def sep_punc(text):
    text = text.strip()
    for p in [",", ".", "-", "+", "’", "'", "/", "\x92", '"', "(", ")", "%", "=", "*"]:
        text = re.sub("%s+" % re.escape(p), " %s " % p, text)
    text = re.sub(" +", " ", text)
    return text.strip()

def complete_phrase(sentence, phrase):
    if phrase not in sentence:
        return phrase, 0

    all_possible_completions = []
    for m in re.finditer(re.escape(phrase), sentence):
        i = m.start()
        while i != 0 and sentence[i-1] != " ":
            i -= 1
        while sentence[i] in ["'", '"']:
            i += 1
        
        j = m.end()
        try:
            while j != len(sentence) and sentence[j] != " ":
                j += 1
        except:
            print (re.escape(phrase), re.escape(sentence))
            print (j, sentence, len(sentence), m, phrase)
            assert False

        # while sentence[j-1] in ["'", '"', ",", "\\"]:
            # j -= 1
        # if j == len(sentence) and sentence[j-1] in ["."]:
            # while sentence[j-1] in ["."]:
                # j -= 1
        
        all_possible_completions.append(sentence[i:j])

    assert len(all_possible_completions) > 0, (phrase, re_sentence)
    all_possible_completions = sorted(all_possible_completions, key=lambda x: len(x))
    return all_possible_completions[0], int(all_possible_completions[0] == phrase)

def create_data_and_save_eae(src_data, trans_data, config):
    src_data, src_entities, src_roles, src_num_entities = src_data[0], src_data[1], src_data[2], src_data[3]
    trans_text, trans_entities = trans_data[0], trans_data[1]
    assert len(src_data) == len(trans_text)
    assert len(trans_entities) == len(src_entities) == len(src_roles)

    tgt_data, entity_idx = [], 0
    char_lang = True if config.tgt_lang in ["zh", "chinese", "ja", "japanese"] else False
    num_ems_tgt, num_ems_src = 0, 0
    for i, dt in enumerate(src_data):

        if char_lang:
            trans_text[i] = trans_text[i].replace(" ", "")

        tgt_dt = {
            "doc_id": dt["doc_id"] + "_%s" % config.tgt_lang,
            "wnd_id": dt["wnd_id"] + "_%s" % config.tgt_lang,
            "text": trans_text[i],
            "tokens": trans_text[i].split() if not char_lang else list(trans_text[i]),
            "entity_mentions": [],
            "event_mentions": []
        }
        args_in_text = 1
        text2entity = {}

        assert len(dt["event_mentions"]) == len(src_num_entities[i]), (i, len(dt["event_mentions"]), len(src_num_entities[i]))
        for j, em in enumerate(dt["event_mentions"]):
            num_ems_src += 1

            tgt_em = {
                "event_type": em["trigger"][2],
                "id": em["id"],
                "trigger": {},
                "arguments": []
            }
            
            assert src_roles[entity_idx] == "Trigger"
            entity_trans = trans_entities[entity_idx]
            if not char_lang and trans_entities[entity_idx] in trans_text[i]:
                entity_trans, _ = complete_phrase(trans_text[i], trans_entities[entity_idx])
            t_s, t_e = get_span_idx(trans_text[i], entity_trans, char_lang=char_lang)

            if t_s != -1:
                tgt_em["trigger"]["text"] = entity_trans
                tgt_em["trigger"]["start"] = t_s
                tgt_em["trigger"]["end"] = t_e
                entity_idx += 1
            elif trans_entities[entity_idx] in trans_text[i]:
                import ipdb; ipdb.set_trace()
            else:
                entity_idx += src_num_entities[i][j]
                continue
            
            if config.only_ed:
                tgt_dt["event_mentions"].append(tgt_em)
                entity_idx += src_num_entities[i][j] - 1
                continue

            for arg in em["arguments"]:
                entity_trans = trans_entities[entity_idx]
                if not char_lang and trans_entities[entity_idx] in trans_text[i]:
                    entity_trans, _ = complete_phrase(trans_text[i], trans_entities[entity_idx])

                arg = {
                    "entity_id": None,
                    "text": entity_trans,
                    "role": src_roles[entity_idx]
                }

                a_s, a_e = get_span_idx(trans_text[i], entity_trans, trigger_span=(t_s, t_e), char_lang=char_lang)
                if a_s != -1:
                    if entity_trans in text2entity:
                        arg["entity_id"] = text2entity[entity_trans]
                    else:
                        entity = {
                            "id": dt["doc_id"] + "_%s_%s" % (config.tgt_lang, "EN%d" % len(tgt_dt["entity_mentions"])),
                            "start": a_s,
                            "end": a_e,
                            "text": entity_trans
                        }
                        text2entity[entity_trans] = entity["id"]
                        tgt_dt["entity_mentions"].append(entity)
                        arg["entity_id"] = entity["id"]

                    tgt_em["arguments"].append(arg)
                else:
                    args_in_text = 0
                
                entity_idx += 1
            
            if args_in_text:
                tgt_dt["event_mentions"].append(tgt_em)
            args_in_text = 1
        
        if len(tgt_dt["event_mentions"]) > 0:
            num_ems_tgt += len(tgt_dt["event_mentions"])
            tgt_data.append(tgt_dt)
    
    print ("Final target event mentions: %d/%d = %.2f" % (num_ems_tgt, num_ems_src, float(num_ems_tgt/num_ems_src)))
    print ("Final target sentences: %d/%d = %.2f" % (len(tgt_data), len(src_data), float(len(tgt_data)/len(src_data))))
    if config.only_ed:
        filename = config.tgt_folder + "/trans_%s_%s_ed.txt" % (config.tgt_lang, config.split)
    else:
        filename = config.tgt_folder + "/trans_%s_%s_eae.txt" % (config.tgt_lang, config.split)
    with open(filename, 'w') as f:
        for dt in tgt_data:
            f.write(json.dumps(dt, ensure_ascii=False) + "\n")

    return