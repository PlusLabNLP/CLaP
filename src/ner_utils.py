import re, string, itertools
from torch.utils.data import DataLoader
from collections import namedtuple

TextBatch = namedtuple('TextBatch', field_names=["batch_text"], defaults=[None])
def text_collate_fn(batch):
    return TextBatch(
        batch_text=[instance for instance in batch]
    )

EntityBatch = namedtuple('EntityBatch', field_names=["batch_text", "batch_trans"], defaults=[None])
def entity_collate_fn(batch):
    return EntityBatch(
        batch_trans=[instance[0] for instance in batch],
        batch_text=[instance[1] for instance in batch]
    )

def process_data_translate_ner(data, batch_size):
    dataloader_text = DataLoader(data[0], batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=text_collate_fn)
    dataloader_entity = DataLoader(data[1], batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=text_collate_fn)
    return dataloader_text, dataloader_entity

def process_data_contextual_ner(src_data, trans_text, batch_size):
    assert len(src_data[0]) == len(trans_text), (len(src_data[0]), len(trans_text))
    _, entities, _, num_entities = src_data
    processed_data = []
    en_idx = 0
    for i, (text, num_entity) in enumerate(zip(trans_text, num_entities)):
        for _ in range(num_entity):
            processed_data.append((text, entities[en_idx]))
            en_idx += 1

    dataloader_entity = DataLoader(processed_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=entity_collate_fn)
    return dataloader_entity

def load_ner_data(src_filename, lang, max_input_length):
    texts = []
    entities = []
    labels = []
    num_entities = []
    delimiter = " " if lang not in ["zh", "ja"] else ""
    trunc_sentences = 0
    with open(src_filename, encoding="utf-8") as f:
        words = []
        entity = []
        label = ""
        num_entity = 0
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    if len(words) > max_input_length:
                        words = words[:max_input_length]
                        trunc_sentences += 1
                    texts.append(delimiter.join(words))
                    words = []
                    if len(entity) > 0:
                        entities.append(delimiter.join(entity))
                        labels.append(label)
                        entity = []
                        label = ""
                        num_entity += 1
                    num_entities.append(num_entity)
                    num_entity = 0
            else:
                splits = line.strip().split("\t")
                word = splits[0]
                words.append(splits[0])

                if splits[-1][0] == "B":
                    if len(entity) > 0:
                        entities.append(delimiter.join(entity))
                        labels.append(label)
                        entity = []
                        label = ""
                        num_entity += 1
                    entity.append(word)
                    label = splits[-1][2:]
                elif splits[-1][0] == "I":
                    entity.append(word)
                elif splits[-1][0] == "O":
                    if len(entity) > 0:
                        entities.append(delimiter.join(entity))
                        labels.append(label)
                        entity = []
                        label = ""
                        num_entity += 1
                else:
                    raise NotImplementedError

        if words:
            texts.append(delimiter.join(words))
            words = []
            if len(entity) > 0:
                entities.append(delimiter.join(entity))
                labels.append(label)
                entity = []
                label = ""
                num_entity += 1
            num_entities.append(num_entity)
            num_entity = 0

    assert len(texts) == len(num_entities), (len(texts), len(num_entities))
    assert len(entities) == len(labels) == sum(num_entities)
    print ("Read %s file with %d instances with %d entities" % (lang, len(texts), len(entities)))
    print ("Truncated %d sentences" % trunc_sentences)
    return (texts, entities, labels, num_entities)

def get_char2token(text, tokens=None):
    char2token = [-1] * len(text)
    token_num, char_num = 0, 0
    tokens = text.split() if tokens is None else tokens
    tmp_text = text
    while char_num != len(text):
        while char_num != len(text) and (text[char_num] == " " or text[char_num] == "\u3000"):
            char_num += 1
        if char_num == len(text):
            break
        assert text[char_num:].startswith(tokens[token_num]), (text, tokens, char_num, token_num, tokens[token_num])
        for c in range(char_num, char_num + len(tokens[token_num])):
            char2token[c] = token_num
        char_num += len(tokens[token_num])
        if char_num == len(text):
            break
        while char_num != len(text) and (text[char_num] == " " or text[char_num] == "\u3000"):
            char_num += 1
        token_num += 1
    return char2token

def get_start_end(text, phrase):
    if phrase == "":
        return []

    all_matches = []
    for match in re.finditer(re.escape(phrase), text):
        all_matches.append((match.start(), match.end()))
    return all_matches

def create_data_and_save_ner(trans_data, config, split, all_entity_included=False):
    texts, entities, labels, num_entities = trans_data
    en_idx, conflicts, matched = 0, 0, 0
    final_data = []
    for i, (text, num_entity) in enumerate(zip(texts, num_entities)):
        tokens = text.split() if config.tgt_lang not in ["zh", "ja"] else text.replace(" ", "")
        label = ['O'] * len(tokens)
        all_entity = 1
        for _ in range(num_entity):
            role = labels[en_idx]
            if config.tgt_lang in ["zh", "ja"]:
                all_matches = get_start_end(text.replace(" ", ""), entities[en_idx].strip())
                char2token = get_char2token(text.replace(" ", ""), tokens=tokens)
            else:
                all_matches = get_start_end(text, entities[en_idx].strip())
                char2token = get_char2token(text, tokens=tokens)
            if len(all_matches) == 0:
                all_entity = 0
            
            matched += len(all_matches)
            for match_start, match_end in all_matches:
                token_start, token_end = char2token[match_start], char2token[match_end-1]+1
                assert token_start < len(label) and token_end <= len(label)
                conflict = 0
                for idx in range(token_start, token_end):
                    if label[idx] == 'O':
                        label[idx] = ("B-" if idx == token_start else "I-") + role
                    else:
                        if label[idx][2:] != role:
                            conflict = 1
                if conflict:
                    conflict += 1

            en_idx += 1
        final_data.append((tokens, label, all_entity))

    all_entity_sentences = sum([ dt[2] for dt in final_data ])
    print ("Found %d conflicts for a total of %d entities: %.2f" % (conflicts, en_idx, (100-conflicts/en_idx*100)))
    print ("Found %d matches for a total of %d entities: %.2f" % (matched, en_idx, matched/en_idx*100))
    print ("Found %d all-entity sentences for a total of %d sentences: %.2f" % (all_entity_sentences, len(final_data), all_entity_sentences/len(final_data)*100))
    with open(config.tgt_folder + "/trans_%s_%s.txt" % (config.tgt_lang, split), 'w') as f:
        for dt in final_data:
            if all_entity_included:
                if dt[2]:
                    for word, tag in zip(dt[0], dt[1]):
                        f.write("\t".join([word, tag]) + "\n")
                    f.write("\n")
            else:
                for word, tag in zip(dt[0], dt[1]):
                    f.write("\t".join([word, tag]) + "\n")
                f.write("\n")
    return