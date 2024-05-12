import os, logging, json, _jsonnet, random, datetime, pprint
import numpy as np
import torch
from argparse import Namespace
from models import *
import ipdb
import re

logger = logging.getLogger(__name__)

VALID_TASKS = ["E2E", "ED", "EAE"]

TRAINER_MAP = {
    ("XGear", "EAE"): XGearEAETrainer,
    ("XGear+Copy", "EAE"): XGearCopyEAETrainer,
}

def load_config(config_fn):
    config = json.loads(_jsonnet.evaluate_file(config_fn))
    config = Namespace(**config)
    assert config.task in VALID_TASKS, f"Task must be in {VALID_TASKS}"
    
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def set_gpu(gpu_device):
    if gpu_device >= 0:
        torch.cuda.set_device(gpu_device)
        
def set_logger(config):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    output_dir = os.path.join(config.output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                        handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
        json.dump(vars(config), fp, indent=4)
        
    config.output_dir = output_dir
    config.log_path = log_path
    
    return config

def load_data(task, file, add_extra_info_fn, config):
    if task == "E2E":
        data, type_set = load_E2E_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "ED":
        data, type_set = load_ED_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif task == "EAE":
        data, type_set = load_EAE_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif task == "EARL":
        data, type_set = load_EARL_data(file, add_extra_info_fn, config)
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return data, type_set

def load_all_data(config, add_extra_info_fn, use_aux_info=False):
    if config.task == "E2E":
        train_data, train_type_set = load_E2E_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_E2E_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_E2E_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "ED":
        train_data, train_type_set = load_ED_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_ED_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_ED_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"]}
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif config.task == "EAE":
        train_data, train_type_set = load_EAE_data(config.train_file, add_extra_info_fn, config, use_aux_info=use_aux_info)
        dev_data, dev_type_set = load_EAE_data(config.dev_file, add_extra_info_fn, config, use_aux_info=use_aux_info)
        test_data, test_type_set = load_EAE_data(config.test_file, add_extra_info_fn, config, use_aux_info=use_aux_info)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    elif config.task == "EARL":
        train_data, train_type_set = load_EARL_data(config.train_file, add_extra_info_fn, config)
        dev_data, dev_type_set = load_EARL_data(config.dev_file, add_extra_info_fn, config)
        test_data, test_type_set = load_EARL_data(config.test_file, add_extra_info_fn, config)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return train_data, dev_data, test_data, type_set

def load_E2E_data(file, add_extra_info_fn, config):
    if type(file) == list:
        data = []
        type_set = dict()
        for f in file:
            d, t = load_E2E_data_(f, add_extra_info_fn, config)
            data.append(d)
            for key, value in t.items():
                if key not in type_set:
                    type_set[key] = set()
                type_set[key] |= value
        return data, type_set
    else:
        return load_E2E_data_(file, add_extra_info_fn, config)

def load_E2E_data_(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
    instances = []
    for dt in data:

        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        events = []
        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            events.append({"trigger": trigger, "arguments": arguments})

        events.sort(key=lambda x: (x['trigger'][0], x['trigger'][1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["sentence"], 
                    "events": events, 
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for event in instance['events']:
            trigger_type_set.add(event['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for event in instance['events']:
            for argument in event["arguments"]:
                role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    logger.info('Loaded {} E2E instances ({} trigger types and {} role types) from {}'.format(
        len(new_instances), len(trigger_type_set), len(role_type_set), file))
    
    return new_instances, type_set

def load_ED_data(file, add_extra_info_fn, config):
    if type(file) == list:
        data = []
        type_set = dict()
        for f in file:
            d, t = load_ED_data_(f, add_extra_info_fn, config)
            data.append(d)
            for key, value in t.items():
                if key not in type_set:
                    type_set[key] = set()
                type_set[key] |= value
        return data, type_set
    else:
        return load_ED_data_(file, add_extra_info_fn, config)

def load_ED_data_(file, add_extra_info_fn, config):

    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    data = [json.loads(line) for line in lines]
    
    instances = []
    for dt in data:

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        triggers = []
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            triggers.append(trigger)

        triggers.sort(key=lambda x: (x[0], x[1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["sentence"], 
                    "triggers": triggers,
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for trigger in instance['triggers']:
            trigger_type_set.add(trigger[2])

    type_set = {"trigger": trigger_type_set}
    
    # approach-specific preprocessing
    new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
        len(new_instances), len(trigger_type_set), file))
    
    return new_instances, type_set

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    entities.sort(key=lambda x: x['end']-x['start'], reverse=True)
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def load_EAE_data(file, add_extra_info_fn, config, use_aux_info=False):
    if type(file) == list:
        data = []
        type_set = dict()
        for f in file:
            d, t = load_EAE_data_(f, add_extra_info_fn, config)
            data.append(d)
            for key, value in t.items():
                if key not in type_set:
                    type_set[key] = set()
                type_set[key] |= value
        return data, type_set
    else:
        return load_EAE_data_(file, add_extra_info_fn, config, use_aux_info=use_aux_info)

def load_EAE_data_(file, add_extra_info_fn, config, use_aux_info=False):

    data = []
    try:
        with open(file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
    except:
        with open(file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            data = [json.loads(line) for line in lines]
    
    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']
        entities, entity_merge_id_map = remove_overlap_entities(entities)

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            visited = set()
            for arg in event_mention['arguments']:
                if arg['entity_id'] in entity_merge_id_map:
                    mapped_entity = entity_map[entity_merge_id_map[arg['entity_id']]]
                else:
                    mapped_entity = entity_map[arg['entity_id']]
                
                if (mapped_entity['start'], mapped_entity['end']) not in visited:
                    # argument = (start index, end index, role type, text span)
                    argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                    arguments.append(argument)
                    visited.add((mapped_entity['start'], mapped_entity['end']))

            arguments.sort(key=lambda x: (x[0], x[1]))

            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["sentence"] if "sentence" in dt else dt["text"], 
                        "language": dt["language"] if "language" in dt else "en",
                        "trigger": trigger, 
                        "arguments": arguments}

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    # approach-specific preprocessing
    if not use_aux_info:
        new_instances = instances
    else:
        new_instances = add_extra_info_fn(instances, data, config)
    assert len(new_instances) == len(instances)
    
    logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
        len(new_instances), len(trigger_type_set), len(role_type_set), file))
    
    return new_instances, type_set

def convert_ED_to_EAE(data, gold):
    instances = []
    for dt, gd in zip(data, gold):
        for trigger in dt["triggers"]:
            trigger_ = (trigger[0], trigger[1], trigger[2], " ".join(gd["tokens"][trigger[0]:trigger[1]]))
            instance = {"doc_id": gd["doc_id"], 
                        "wnd_id": gd["wnd_id"], 
                        "tokens": gd["tokens"], 
                        "text": gd["text"], 
                        "trigger": trigger_, 
                        "arguments": [], 
                        "extra_info": gd["extra_info"]
                       }
            instances.append(instance)
    
    return instances

def combine_ED_and_EAE_to_E2E(ed_predicitons, eae_predictions):
    e2e_predictions = []
    idx = 0
    for ed_prediciton in ed_predicitons:
        events = []
        for trigger in ed_prediciton["triggers"]:
            eae_prediction = eae_predictions[idx]
            assert ed_prediciton["doc_id"] == eae_prediction["doc_id"]
            assert ed_prediciton["wnd_id"] == eae_prediction["wnd_id"]
            assert trigger[0] == eae_prediction["trigger"][0]
            assert trigger[1] == eae_prediction["trigger"][1]
            assert trigger[2] == eae_prediction["trigger"][2]
            events.append({"trigger": trigger, "arguments": eae_prediction["arguments"]})
            idx += 1
        
        ed_prediciton["events"] = events
        e2e_predictions.append(ed_prediciton)

    return e2e_predictions

def load_text_file(file):
    file_data = []
    with open(file, 'r') as f:
        for line in f:
            file_data.append(line.strip())
    return file_data

def load_trans_data(config):
    assert hasattr(config, "trans_data"), "No meta information of translated data in config"

    trans_langs = list(config.trans_data.keys())
    trans_data = {l:{} for l in trans_langs}

    for lang in config.trans_data:
        trans_data[lang], _ = load_EAE_data(config.trans_data[lang], None, config, use_aux_info=False)
        trans_data[lang] = [ dt for d in trans_data[lang] for dt in d ]
    
    return trans_data

def extract_arguments_from_seq(seq_texts):
    args = []
    for prediction in seq_texts:
        args_pred = []
        tag_s = re.search('<--[^/>][^>]*-->', prediction)
        while tag_s:
            prediction = prediction[tag_s.end():]
            r_type = tag_s.group()[3:-3]
            tag_e = re.search(f'</--{r_type}-->', prediction)
            if tag_e:
                arg = prediction[:tag_e.start()].strip()
                for a in arg.split(' [and] '):
                    a = a.strip()
                    if a != '' and a != "[None]":
                        args_pred.append((a, r_type))
                prediction = prediction[tag_e.end():]
            tag_s = re.search('<--[^/>][^>]*-->', prediction)
        args.append(args_pred)
    return args
