import json, _jsonnet
from xgear.models.XGear.template_generate import event_template, eve_template_generator, IN_SEP, AND
from argparse import Namespace
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple
from torch.utils.data import DataLoader

text_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_text']
TextBatch = namedtuple('TextBatch', field_names=text_fields, defaults=[None] * len(text_fields))

trigger_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_text']
TriggerTranslateBatch = namedtuple('TriggerTranslateBatch', field_names=trigger_fields, defaults=[None] * len(trigger_fields))

arg_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_role', 'batch_text']
ArgumentTranslateBatch = namedtuple('ArgumentTranslateBatch', field_names=arg_fields, defaults=[None] * len(arg_fields))

def text_collate_fn(batch):
    return TextBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_text=[instance["text"] for instance in batch]
    )

def triggertext_translate_collate_fn(batch):
    return TriggerTranslateBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_text=[instance["triggertext"] for instance in batch]
    )

def argumenttext_translate_collate_fn(batch):
    return ArgumentTranslateBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_role=[instance["argumentrole"] for instance in batch],
        batch_text=[instance["argumenttext"] for instance in batch]
    )

trigger_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_text', 'batch_source', 'batch_trans']
TriggerContextualBatch = namedtuple('TriggerContextualBatch', field_names=trigger_fields, defaults=[None] * len(trigger_fields))

arg_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_role', 'batch_text', 'batch_source', 'batch_trans']
ArgumentContextualBatch = namedtuple('ArgumentContextualBatch', field_names=arg_fields, defaults=[None] * len(arg_fields))

def triggertext_contextual_collate_fn(batch):
    return TriggerContextualBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_text=[instance["triggertext"] for instance in batch],
        batch_source=[instance["text"] for instance in batch],
        batch_trans=[instance["trans_text"] for instance in batch]
    )

def argumenttext_contextual_collate_fn(batch):
    return ArgumentContextualBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_role=[instance["argumentrole"] for instance in batch],
        batch_text=[instance["argumenttext"] for instance in batch],
        batch_source=[instance["text"] for instance in batch],
        batch_trans=[instance["trans_text"] for instance in batch]
    )

def load_xgear_config(config_fn):
    config = json.loads(_jsonnet.evaluate_file(config_fn))
    config = Namespace(**config)
    return config

def load_EAE_data(file, config):
    
    data = []
    with open(file, 'r', encoding='utf-8') as fp:
        try:
            data = json.load(fp)
        except:
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
                        "text": dt["sentence"], 
                        "language": dt["language"],
                        "trigger": trigger, 
                        "arguments": arguments}

            instances.append(instance)
            
    print('Loaded {} EAE instances from {}'.format(len(instances), file))
    return instances

def process_data_translate(data, lang, config, batch_size, src_lang):
    text_data, trigger_data, argument_data = [], [], []
    n_total = 0
    for i, dt in enumerate(data):
        
        n_total += 1
        
        _trigger = (dt["trigger"][0], dt["trigger"][1], dt["trigger"][2])
        _arguments = [(_trigger, (r[0], r[1], r[2])) for r in dt["arguments"]]
        event_template = eve_template_generator(config.dataset, dt["tokens"], [_trigger], _arguments,
                        config.input_style, config.output_style, lang)
        event_training_data = event_template.get_training_data()
        assert len(event_training_data) == 1
        data_ = event_training_data[0]

        text_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "text": dt["text"],
                    "input": data_["seq_in"],
                    "target": data_["seq_out"],
                    "info": data_,
                    "idx": i,
                    "lang": src_lang
                    }
        text_data.append(text_dt)

        trigger_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "triggertext": dt["trigger"][-1],
                    "idx": i,
                    # "input": data_[0],
                    # "target": data_[1],
                    # "info": data_[2],
                    }
        trigger_data.append(trigger_dt)

        for arg in dt["arguments"]:
            # Only allow valid roles. Check if role in target string
            if arg[2] not in data_["seq_out"]:
                print("Omitting: ", arg, "not in target")
                continue

            arg_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "argumentrole": arg[2],
                    "argumenttext": arg[-1],
                    "idx": i,
                    # "input": data_[0],
                    # "target": data_[1],
                    # "info": data_[2],
                    }
            argument_data.append(arg_dt)
            
    print ("Generate %d XGear Trigger instances from %d EAE instances" % (len(trigger_data), n_total))
    print ("Generate %d XGear Argument instances from %d EAE instances" % (len(argument_data), n_total))

    dataloader_text = DataLoader(text_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=text_collate_fn)
    dataloader_trigger = DataLoader(trigger_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=triggertext_translate_collate_fn)
    dataloader_argument = DataLoader(argument_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=argumenttext_translate_collate_fn)
    
    return (text_data, trigger_data, argument_data), (dataloader_text, dataloader_trigger, dataloader_argument)

def process_contextual_data_contextual(data, translated_text, lang, config, batch_size, src_lang, multiple_context=False):
    
    assert len(data) == len(translated_text), (len(data), len(translated_text))

    text_data, trigger_data, argument_data = [], [], []
    n_total = 0
    for i, (dt, trans_text) in enumerate(zip(data, translated_text)):
        
        n_total += 1
        
        _trigger = (dt["trigger"][0], dt["trigger"][1], dt["trigger"][2])
        _arguments = [(_trigger, (r[0], r[1], r[2])) for r in dt["arguments"]]
        event_template = eve_template_generator(config.dataset, dt["tokens"], [_trigger], _arguments,
                        config.input_style, config.output_style, lang)
        event_training_data = event_template.get_training_data()
        assert len(event_training_data) == 1
        data_ = event_training_data[0]

        text_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "text": dt["text"],
                    "input": data_["seq_in"],
                    "target": data_["seq_out"],
                    "info": data_,
                    "idx": i,
                    "lang": src_lang
                    }
        text_data.append(text_dt)
        
        trigger_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "triggertext": dt["trigger"][-1],
                    "text": dt["text"],
                    "idx": i,
                    "trans_text": trans_text
                    }
        trigger_data.append(trigger_dt)

        for arg in dt["arguments"]:
            # Only allow valid roles. Check if role in target string
            if arg[2] not in data_["seq_out"]:
                print("Omitting: ", arg, "not in target")
                continue

            arg_dt = {"doc_id": dt["doc_id"],
                    "wnd_id": dt["wnd_id"],
                    "text": dt["text"],
                    "argumentrole": arg[2],
                    "argumenttext": arg[-1],
                    "idx": i,
                    "trans_text": trans_text
                    }
            argument_data.append(arg_dt)
            
    print ("Generate %d XGear Trigger instances from %d EAE instances" % (len(trigger_data), n_total))
    print ("Generate %d XGear Argument instances from %d EAE instances" % (len(argument_data), n_total))

    dataloader_trigger = DataLoader(trigger_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=triggertext_contextual_collate_fn)
    dataloader_argument = DataLoader(argument_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=argumenttext_contextual_collate_fn)
    
    return (text_data, trigger_data, argument_data), (dataloader_trigger, dataloader_argument)

def remove_overlap_entities(entities):

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

        while sentence[j-1] in ["'", '"', ",", "\\"]:
            j -= 1
        if j == len(sentence) and sentence[j-1] in ["."]:
            while sentence[j-1] in ["."]:
                j -= 1
        
        all_possible_completions.append(sentence[i:j])

    assert len(all_possible_completions) > 0, (phrase, re_sentence)
    all_possible_completions = sorted(all_possible_completions, key=lambda x: len(x))
    return all_possible_completions[0], int(all_possible_completions[0] == phrase)

def create_input_contextual_eae(src_text, trans_text, src_trigger, trans_trigger, tgt_lang):
    def get_template(input_str):
        template_start_idx = input_str.find(IN_SEP["template"])
        return input_str[template_start_idx:]
    
    input_gen_data = []
    trigger_in_text_idx = []
    t_improved = 0
    for s_text, t_text, s_trigger, t_trigger in zip(src_text, trans_text, src_trigger, trans_trigger):
        assert s_text["doc_id"] == s_trigger["doc_id"]
        assert s_text["wnd_id"] == s_trigger["wnd_id"]

        in_template = get_template(s_text["input"])
        t_trigger = clean_translations(t_trigger)

        if tgt_lang not in ["zh", "chinese"] and t_trigger in t_text:
            t_trigger, improved = complete_phrase(t_text, t_trigger)
            t_improved += improved

        input_gen_str = t_text + " " + IN_SEP["triggerword"] + " " + t_trigger + " " + in_template
        input_gen_data.append(input_gen_str)

        if t_trigger.lower() in t_text.lower() and t_trigger != "":
            trigger_in_text_idx.append(len(input_gen_data)-1)

    print ("Triggers improved: %d/%d = %.2f" % (t_improved, len(trigger_in_text_idx), t_improved/len(trigger_in_text_idx)))
    print ("Number of times trigger in input text: %d/%d = %.2f" % (len(trigger_in_text_idx), len(trans_trigger), float(len(trigger_in_text_idx)/len(trans_trigger))))
    return input_gen_data, trigger_in_text_idx

def create_output_contextual_eae(src_text, tgt_text, src_arg, tgt_arg, tgt_lang, text_key="argumenttext", sep_src=False):
    output_gen_data = []
    src_txt_idx = 0
    arg_idx = 0
    args_in_text_idx = []
    arg_in_text_idx = 1
    num_arg_in_text = 0
    a_improved = 0

    for src_txt_idx, s_text in tqdm(enumerate(src_text)):
        base_template = deepcopy(s_text["target"])

        while arg_idx < len(src_arg) and src_arg[arg_idx]["doc_id"] == s_text["doc_id"] and \
            src_arg[arg_idx]["wnd_id"] == s_text["wnd_id"] and src_arg[arg_idx]["idx"] == src_txt_idx:

            s_arg = src_arg[arg_idx]
            try:
                t_arg = tgt_arg[arg_idx]
            except:
                import ipdb; ipdb.set_trace()
                print (t_arg, arg_idx)
                assert False
            # t_arg = clean_translations(t_arg)         # TODO: Fix this. Currently trailing . in U.S. are removed
            if s_text["lang"] in ["zh", "chinese"]:
                s_arg[text_key] = s_arg[text_key].replace(" ", "")
            s_arg[text_key] = sep_punc(s_arg[text_key]) if sep_src else s_arg[text_key]

            if s_arg[text_key] not in base_template:
                print ("ERROR: source arg %s not in base template" % s_arg[text_key], base_template)
                old_s_arg = s_arg[text_key]
                s_arg[text_key] = base_template.split("<--%s-->" % s_arg["argumentrole"])[1].split("</--%s-->" % s_arg["argumentrole"])[0].strip()
                print ("REPLACING %s with %s" % (old_s_arg, s_arg[text_key]))

            if tgt_lang not in ["zh", "chinese"]:
                t_arg, improved = complete_phrase(tgt_text[src_txt_idx].lower(), t_arg)
                a_improved += improved

            arg_template_src_text = "<--" + s_arg["argumentrole"] + "--> " + s_arg[text_key] + " </--" + s_arg["argumentrole"] + "-->"
            arg_template_tgt_text = "<--" + s_arg["argumentrole"] + "--> " + t_arg + " </--" + s_arg["argumentrole"] + "-->" if t_arg != "" else \
                                    "<--" + s_arg["argumentrole"] + "--> " + "[None]" + " </--" + s_arg["argumentrole"] + "-->"
            if arg_template_src_text in base_template:
                base_template = base_template.replace(arg_template_src_text, arg_template_tgt_text)
            
            # Exempt the case of AND
            elif AND in base_template:
                base_template = base_template.replace(" %s " % s_arg[text_key], " %s " % t_arg, 1)
            
            # Maybe it's the next sentence
            elif src_text[src_txt_idx+1]["doc_id"] == src_arg[arg_idx]["doc_id"] and \
                src_text[src_txt_idx+1]["wnd_id"] == src_arg[arg_idx]["wnd_id"]:
                break
            
            # Erroneous cases. Do your best and print cases to keep an eye.
            else:
                print ("Error:\nTemplate: %s\nReplaced arg:" % (base_template), s_arg)
                base_template = base_template.replace(s_arg[text_key], t_arg)

            if t_arg.lower() in tgt_text[src_txt_idx].lower() and t_arg != "":
                num_arg_in_text += 1
                arg_in_text_idx &= arg_in_text_idx
            else:
                arg_in_text_idx = 0
            arg_idx += 1
        
        output_gen_data.append(base_template)
        if arg_in_text_idx:
            args_in_text_idx.append(len(output_gen_data)-1)
        arg_in_text_idx = 1
    
    print ("Arguments improved: %d/%d = %.2f" % (a_improved, len(args_in_text_idx), a_improved/len(args_in_text_idx)))
    print ("Number of times arguments in input text: %d/%d = %.2f" % (num_arg_in_text, len(tgt_arg), float(num_arg_in_text/len(tgt_arg))))
    print ("Number of times all arguments in input text: %d/%d = %.2f" % (len(args_in_text_idx), len(src_text), float(len(args_in_text_idx)/len(src_text))))
    return output_gen_data, args_in_text_idx

def create_data_and_save_contextual_eae(src_data, trans_data, config, split="train", print_statistics=False, arg_text_key="argumenttext"):
    src_text, src_trigger, src_arg = src_data[0], src_data[1], src_data[2]
    trans_text, trans_trigger, trans_arg = trans_data[0], trans_data[1], trans_data[2]

    if config.src_lang != config.tgt_lang:
        print ("Creating Input Text...")
        final_input, trigger_in_text = create_input_contextual_eae(src_text, trans_text, src_trigger, trans_trigger, config.tgt_lang)
        print ("Creating Output Text...")
        sep_src = config.sep_src if hasattr(config, "sep_src") else False
        final_output, args_in_text = create_output_contextual_eae(src_text, trans_text, src_arg, trans_arg, config.tgt_lang, text_key=arg_text_key, sep_src=sep_src)

        with open(config.tgt_folder + "/trans_%s_%s_input.txt" % (config.tgt_lang, split), 'w') as f:
            filter_list = range(len(final_input))
            if config.labels_in_sent:
                filter_list = [ d for d in trigger_in_text if d in args_in_text ]
            final_input = [ final_input[d] for d in range(len(final_input)) if d in filter_list ]
            f.write("\n".join(final_input))

        with open(config.tgt_folder + "/trans_%s_%s_output.txt" % (config.tgt_lang, split), 'w') as f:
            filter_list = range(len(final_input))
            if config.labels_in_sent:
                filter_list = [ d for d in trigger_in_text if d in args_in_text ]
            final_output = [ final_output[d] for d in range(len(final_output)) if d in filter_list ]
            f.write("\n".join(final_output))
    
    if config.save_source:
        with open(config.tgt_folder + "/source_%s_%s_input.txt" % (config.src_lang, split), 'w') as f:
            source_input = [ s["input"] for s in src_text ]
            if config.labels_in_sent:
                source_input = [ source_input[d] for d in range(len(source_input)) if d in trigger_in_text and d in args_in_text ]
            f.write("\n".join(source_input))
        
        with open(config.tgt_folder + "/source_%s_%s_output.txt" % (config.src_lang, split), 'w') as f:
            source_output = [ s["target"] for s in src_text ]
            if config.labels_in_sent:
                source_output = [ source_output[d] for d in range(len(source_output)) if d in trigger_in_text and d in args_in_text ]
            f.write("\n".join(source_output))

    return

def create_input_translate_eae(src_text, trans_text, src_trigger, trans_trigger):
    def get_template(input_str):
        template_start_idx = input_str.find(IN_SEP["template"])
        return input_str[template_start_idx:]
    
    input_gen_data = []
    trigger_in_text_idx = []
    for s_text, t_text, s_trigger, t_trigger in zip(src_text, trans_text, src_trigger, trans_trigger):
        assert s_text["doc_id"] == s_trigger["doc_id"]
        assert s_text["wnd_id"] == s_trigger["wnd_id"]

        in_template = get_template(s_text["input"])
        t_trigger = clean_translations(t_trigger)
        input_gen_str = t_text + " " + IN_SEP["triggerword"] + " " + t_trigger + " " + in_template
        input_gen_data.append(input_gen_str)

        if t_trigger in t_text and t_trigger != "":
            trigger_in_text_idx.append(len(input_gen_data)-1)

    print ("Number of times trigger in input text: %d/%d = %.2f" % (len(trigger_in_text_idx), len(trans_trigger), float(len(trigger_in_text_idx)/len(trans_trigger))))
    return input_gen_data, trigger_in_text_idx

def create_output_translate_eae(src_text, tgt_text, src_arg, tgt_arg, text_key="argumenttext", sep_src=False):
    output_gen_data = []
    src_txt_idx = 0
    arg_idx = 0
    args_in_text_idx = []
    arg_in_text_idx = 1
    num_arg_in_text = 0

    for src_txt_idx, s_text in tqdm(enumerate(src_text)):
        base_template = deepcopy(s_text["target"])

        while arg_idx < len(src_arg) and src_arg[arg_idx]["doc_id"] == s_text["doc_id"] and \
            src_arg[arg_idx]["wnd_id"] == s_text["wnd_id"] and src_arg[arg_idx]["idx"] == src_txt_idx:

            s_arg = src_arg[arg_idx]
            t_arg = tgt_arg[arg_idx]
            if s_text["lang"] in ["zh", "chinese"]:
                s_arg[text_key] = s_arg[text_key].replace(" ", "")
            s_arg[text_key] = sep_punc(s_arg[text_key]) if sep_src else s_arg[text_key]

            if s_arg[text_key] not in base_template:
                print ("ERROR: source arg %s not in base template" % s_arg[text_key], base_template)
                old_s_arg = s_arg[text_key]
                s_arg[text_key] = base_template.split("<--%s-->" % s_arg["argumentrole"])[1].split("</--%s-->" % s_arg["argumentrole"])[0].strip()
                print ("REPLACING %s with %s" % (old_s_arg, s_arg[text_key]))

            arg_template_src_text = "<--" + s_arg["argumentrole"] + "--> " + s_arg[text_key] + " </--" + s_arg["argumentrole"] + "-->"
            arg_template_tgt_text = "<--" + s_arg["argumentrole"] + "--> " + t_arg + " </--" + s_arg["argumentrole"] + "-->" if t_arg != "" else \
                                    "<--" + s_arg["argumentrole"] + "--> " + "[None]" + " </--" + s_arg["argumentrole"] + "-->"
            if arg_template_src_text in base_template:
                base_template = base_template.replace(arg_template_src_text, arg_template_tgt_text)
            
            # Exempt the case of AND
            elif AND in base_template:
                base_template = base_template.replace(" %s " % s_arg[text_key], " %s " % t_arg, 1)
            
            # Maybe it's the next sentence
            elif src_text[src_txt_idx+1]["doc_id"] == src_arg[arg_idx]["doc_id"] and \
                src_text[src_txt_idx+1]["wnd_id"] == src_arg[arg_idx]["wnd_id"]:
                break
            
            # Erroneous cases. Do your best and print cases to keep an eye.
            else:
                print ("Error:\nTemplate: %s\nReplaced arg:" % (base_template), s_arg)
                base_template = base_template.replace(s_arg[text_key], t_arg)

            if t_arg.lower() in tgt_text[src_txt_idx].lower() and t_arg != "":
                num_arg_in_text += 1
                arg_in_text_idx &= arg_in_text_idx
            else:
                arg_in_text_idx = 0
            arg_idx += 1
        
        output_gen_data.append(base_template)
        if arg_in_text_idx:
            args_in_text_idx.append(len(output_gen_data)-1)
        arg_in_text_idx = 1
    
    print ("Number of times arguments in input text: %d/%d = %.2f" % (num_arg_in_text, len(tgt_arg), float(num_arg_in_text/len(tgt_arg))))
    print ("Number of times all arguments in input text: %d/%d = %.2f" % (len(args_in_text_idx), len(src_text), float(len(args_in_text_idx)/len(src_text))))
    return output_gen_data, args_in_text_idx

def create_data_and_save_translate_eae(src_data, trans_data, config, split="train", print_statistics=False, arg_text_key="argumenttext"):
    src_text, src_trigger, src_arg = src_data[0], src_data[1], src_data[2]
    trans_text, trans_trigger, trans_arg = trans_data[0], trans_data[1], trans_data[2]

    if config.src_lang != config.tgt_lang:
        print ("Creating Input Text...")
        final_input, trigger_in_text = create_input_translate_eae(src_text, trans_text, src_trigger, trans_trigger)
        print ("Creating Output Text...")
        sep_src = config.sep_src if hasattr(config, "sep_src") else False
        final_output, args_in_text = create_output_translate_eae(src_text, trans_text, src_arg, trans_arg, text_key=arg_text_key, sep_src=sep_src)

        with open(config.tgt_folder + "/trans_%s_%s_input.txt" % (config.tgt_lang, split), 'w') as f:
            filter_list = range(len(final_input))
            if config.labels_in_sent:
                filter_list = [ d for d in trigger_in_text if d in args_in_text ]
            final_input = [ final_input[d] for d in range(len(final_input)) if d in filter_list ]
            f.write("\n".join(final_input))

        with open(config.tgt_folder + "/trans_%s_%s_output.txt" % (config.tgt_lang, split), 'w') as f:
            filter_list = range(len(final_input))
            if config.labels_in_sent:
                filter_list = [ d for d in trigger_in_text if d in args_in_text ]
            final_output = [ final_output[d] for d in range(len(final_output)) if d in filter_list ]
            f.write("\n".join(final_output))
    
    if config.save_source:
        with open(config.tgt_folder + "/source_%s_%s_input.txt" % (config.src_lang, split), 'w') as f:
            source_input = [ s["input"] for s in src_text ]
            if config.labels_in_sent:
                source_input = [ source_input[d] for d in range(len(source_input)) if d in trigger_in_text and d in args_in_text ]
            f.write("\n".join(source_input))
        
        with open(config.tgt_folder + "/source_%s_%s_output.txt" % (config.src_lang, split), 'w') as f:
            source_output = [ s["target"] for s in src_text ]
            if config.labels_in_sent:
                source_output = [ source_output[d] for d in range(len(source_output)) if d in trigger_in_text and d in args_in_text ]
            f.write("\n".join(source_output))

    return