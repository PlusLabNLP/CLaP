import numpy as np
from tqdm import tqdm
import torch

lang_map = {
    "english": "en_XX", 
    "chinese": "zh_CN", 
    "arabic": "ar_AR", 
    "spanish": "es_XX",
    "ar": "ar_AR",
    "en": "en_XX",
    "zh": "zh_CN",
    "es": "es_XX"
}

def create_example_str(input_str, target_str, in_sep_template):
    assert input_str.find(in_sep_template) == input_str.rfind(in_sep_template)
    template_start_idx = input_str.find(in_sep_template)
    final_example_str = input_str[:template_start_idx] + in_sep_template + " " + target_str
    return final_example_str

def get_sentence_embedding(input_sentence, trigger_text, sim_model, sim_dim):
    if "passage" in sim_dim and "trigger" in sim_dim:
        return torch.from_numpy(np.concatenate((sim_model.encode(input_sentence, show_progress_bar=False), 
                                            sim_model.encode(trigger_text, show_progress_bar=False)),
                                            axis=None))
    elif "passage" in sim_dim:
        return torch.from_numpy(sim_model.encode(input_sentence, show_progress_bar=False))
    elif "trigger" in sim_dim:
        return torch.from_numpy(sim_model.encode(trigger_text, show_progress_bar=False))
    else:
        return None

def embed_instances(data, sim_model, sim_dim):
    instance_embeddings = []
    for dt in tqdm(data):
        trigger_text, passage = dt["trigger"][-1], dt["text"]
        instance_embeddings.append(get_sentence_embedding(passage, trigger_text, sim_model, sim_dim))
    return instance_embeddings

def distance_matrix(tokens, stanza_result):
    n_tokens = len(tokens)
    parents = np.zeros(n_tokens, dtype=np.int32)
    for item in stanza_result[0]:
        i1 = item['id'] - 1
        i2 = item['head'] - 1
        parents[i1] = i2
        
    hights = np.zeros(n_tokens, dtype=np.int32)
    for i in range(n_tokens):
        p = i
        while parents[p] != -1:
            hights[i] += 1
            p = parents[p]
            
    dist = np.zeros((n_tokens, n_tokens), dtype=np.int32)
    for i in range(n_tokens):
        for j in range(n_tokens):
            d = 0
            pi = i
            pj = j
            while pi != pj:
                d += 1
                if hights[pi] > hights[pj]:
                    pi = parents[pi]
                else:
                    pj = parents[pj]
            dist[i, j] = d
            
    for i in range(n_tokens):
        dist[i, i] = 1
        
    assert not np.any(dist == 0)

    return dist.tolist()

def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.
    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 
    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]
        
def get_span_idxs_zh(tokens, span, trigger_span=None):
    candidates = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            c_string = "".join(tokens[i:j+1])
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