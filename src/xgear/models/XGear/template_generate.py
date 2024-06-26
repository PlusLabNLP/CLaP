from .pattern import patterns
import re
import ipdb

INPUT_STYLE_SET = ['triggerword', 'template', 'trigger_tagger']
OUTPUT_STYLE_SET = ['argument:roletype']
NO_SPACE_LANGS = {"chinese", "zh"}
IN_SEP = {
    'triggerword': '<--triggerword-->',
    'template': '<--template-->',
}
TAGGER = {
    'b-trigger': '<--trigger-->',
    'e-triggger': '</--trigger-->'
}
NO_ROLE = "[None]"
AND = "[and]"
SEP = "\n"

ROLE_LIST = ['Prosecutor', 'Beneficiary', 'Seller', 'Buyer', 'Plaintiff', 'Recipient', 'Giver', 'Vehicle', 'Org', 'Adjudicator', 'Origin', 'Defendant', 'Instrument', 'Agent', 'Target', 'Person', 'Victim', 'Attacker', 'Destination', 'Artifact', 'Entity', 'Place']

class eve_template_generator():
    def __init__(self, dataset, passage, triggers, roles, input_style, output_style, lang):
        """
        generate strctured information for events
        
        args:
            dataset(str): which dataset is used
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            output_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
            lang(str): what language is the passage written in
        """
        self.raw_passage = passage
        self.lang = lang
        self.no_space = lang in NO_SPACE_LANGS
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style

        self.event_templates = []
        for event in self.events:
            self.event_templates.append(
                event_template(event['event type'], patterns[dataset][event['event type']], 
                self.input_style, self.output_style, event['tokens'], lang, event)
            )
        self.data = [x.generate_pair((x.trigger_text, x.trigger_span)) for x in self.event_templates]
        self.data = [x for x in self.data if x]

    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = ''.join(passage[trigger[0]:trigger[1]]) if self.no_space else ' '.join(passage[trigger[0]:trigger[1]])
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': ''.join(passage[argument[1][0]:argument[1][1]]) if self.no_space else ' '.join(passage[argument[1][0]:argument[1][1]]),
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ''.join(passage) if self.no_space else ' '.join(passage),
                'tokens': passage
            })
        return event_structures

def format_output_template(input_role_list):
    return " ".join(["<--{}--> {{VALUE_{}}} </--{}-->".format(r, r, r) for r in sorted(input_role_list, reverse=True)])
    # return " ".join(["<--{}--> {{VALUE_{}}} </--{}-->".format(r, r, r) for r in ROLE_LIST if r in input_role_list])
    # return " ".join(["<--{}--> {{VALUE_{}}} </--{}-->".format(r, r, r) for r in input_role_list])

class event_template():
    def __init__(self, event_type, info_dict, input_style, output_style, passage, lang, gold_event=None):
        self.info_dict = info_dict
        self.event_type = event_type
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.lang = lang
        self.no_space = lang in NO_SPACE_LANGS
        self.passage = ''.join(passage) if self.no_space else ' '.join(passage)
        self.tokens = passage

        if gold_event is not None:
            self.gold_event = gold_event
            self.trigger_text = gold_event['trigger text']
            self.trigger_span = [gold_event['trigger span']]
            self.arguments = [gold_event['arguments']]         
        else:
            self.gold_event = None

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    value_dict = {"VALUE_"+str(r): "[None]" for r in self.info_dict['valid roles']}
                    output_template += ' {} {}'.format(SEP, (format_output_template(self.info_dict['valid roles'])).format(**value_dict))
        return (f'{SEP}'.join(output_template.split(f'{SEP}')[1:])).strip()

    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair

        query_trigger: a tuple of (trigger_text, trigger_span)
        """
        input_str, passage_, supplementary = self.generate_input_str(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        # return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)
        return {"seq_in": input_str, "seq_out": output_str, "gold_event": self.gold_event, "gold_sample": gold_sample, "event_type": self.event_type, "tokens": self.tokens, "added_sequence": supplementary, "passage": passage_}

    def generate_input_str(self, query_trigger):
        if "trigger_tagger" in self.input_style:
            ipdb.set_trace()
            aug_tokens = self.tokens
            aug_tokens.insert(query_trigger[1][1], TAGGER['e-trigger']) # add end of trigger
            aug_tokens.insert(query_trigger[1][0], TAGGER['b-trigger']) # add begin of trigger
            input_str = ''.join(aug_tokens) if self.no_space else ' '.join(aug_tokens)
        else:
            input_str = self.passage
        supplementary = ''
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'triggerword':
                    supplementary += ' {} {}'.format(IN_SEP['triggerword'], query_trigger[0])
                if i_style == 'template':
                    supplementary += ' {} {}'.format(IN_SEP['template'], self.output_template)
        return input_str+supplementary, input_str, supplementary

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = dict()
                        roles = self.info_dict['valid roles']
                        for role_type in roles:
                            filler['{}{}'.format("VALUE_", role_type)] = f" {AND} ".join([ a['argument text'] for a in argu[role_type]]) if role_type in argu.keys() else NO_ROLE
                        output_texts.append((format_output_template(self.info_dict['valid roles'])).format(**filler))
                        gold_sample = True
                    output_str += ' {} {}'.format(SEP, ' <sep> '.join(output_texts)) # In our case, this <sep> seems to be useless
        output_str = (f'{SEP}'.join(output_str.split(f'{SEP}')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split(f'{SEP}')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'argument:roletype':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    tag_s = re.search('<--[^/>][^>]*-->', prediction)
                                    while tag_s:
                                        prediction = prediction[tag_s.end():]
                                        r_type = tag_s.group()[3:-3]
                                        
                                        if r_type in self.info_dict['valid roles']:
                                            tag_e = re.search(f'</--{r_type}-->', prediction)
                                            if tag_e:
                                                arg = prediction[:tag_e.start()].strip()
                                                for a in arg.split(f' {AND} '):
                                                    a = a.strip()
                                                    if a != '' and a != NO_ROLE:
                                                        output.append((a, r_type, {'cor tri cnt': a_cnt}))
                                                prediction = prediction[tag_e.end():]
                                        
                                        tag_s = re.search('<--[^/>][^>]*-->', prediction)
                            except:
                                pass
                        used_o_cnt += 1
        return output

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        # categorize prediction
        pred_trigger = []
        pred_argument = []
        for pred in predict_output:
            if pred[1] == self.event_type:
                pred_trigger.append(pred)
            else:
                pred_argument.append(pred)
        
        # get trigger id map
        pred_trigger_map = {}
        for p_tri in pred_trigger:
            # assert p_tri[2]['tri counter'] not in pred_trigger_map.keys()
            pred_trigger_map[p_tri[2]['tri counter']] = p_tri

        # trigger score
        gold_tri_num = len(self.trigger_span)
        pred_tris = []
        for pred in pred_trigger:
            pred_span = self.predstr2span(pred[0])
            if pred_span[0] > -1:
                pred_tris.append((pred_span[0], pred_span[1], pred[1]))
        pred_tri_num = len(pred_tris)
        match_tri = 0
        for pred in pred_tris:
            id_flag = False
            for gold_span in self.trigger_span:
                if gold_span[0] == pred[0] and gold_span[1] == pred[1]:
                    id_flag = True
            match_tri += int(id_flag)

        # argument score
        converted_gold = self.get_converted_gold()
        gold_arg_num = len(converted_gold)
        pred_arg = []
        for pred in pred_argument:
            # find corresponding trigger
            pred_span = self.predstr2span(pred[0], self.trigger_span[0][0])
            if (pred_span is not None) and (pred_span[0] > -1):
                pred_arg.append((pred_span[0], pred_span[1], pred[1]))
        pred_arg = list(set(pred_arg))
        pred_arg_num = len(pred_arg)
        
        target = converted_gold
        match_id = 0
        match_type = 0
        for pred in pred_arg:
            id_flag = False
            id_type = False
            for gold in target:
                if gold[0]==pred[0] and gold[1]==pred[1]:
                    id_flag = True
                    if gold[2] == pred[2]:
                        id_type = True
                        break
            match_id += int(id_flag)
            match_type += int(id_type)
        return {
            'gold_tri_num': gold_tri_num, 
            'pred_tri_num': pred_tri_num,
            'match_tri_num': match_tri,
            'gold_arg_num': gold_arg_num,
            'pred_arg_num': pred_arg_num,
            'match_arg_id': match_id,
            'match_arg_cls': match_type
        }
    
    def get_converted_gold(self):
        converted_gold = []
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold.append((arg['argument span'][0], arg['argument span'][1], arg_type))
        return list(set(converted_gold))
    
    def predstr2span(self, pred_str, trigger_idx=None):
        sub_words = [_.strip() for _ in pred_str.strip().lower().split()]
        candidates=[]
        for i in range(len(self.tokens)):
            j = 0
            while j < len(sub_words) and i+j < len(self.tokens):
                if self.tokens[i+j].lower() == sub_words[j]:
                    j += 1
                else:
                    break
            if j == len(sub_words):
                candidates.append((i, i+len(sub_words)))
        if len(candidates) < 1:
            return -1, -1
        else:
            if trigger_idx is not None:
                return sorted(candidates, key=lambda x: abs(trigger_idx-x[0]))[0]
            else:
                return candidates[0]
