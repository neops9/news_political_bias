import os
import io
import network.special_tokens
import torch
import bcolz
import pickle
import json 
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.utils import class_weight

def load_embeddings(dir):
    vectors = bcolz.open(f'{dir}/6B.300.dat')[:]
    words = pickle.load(open(f'{dir}/6B.300_words.pkl', 'rb'))
    word_to_id = pickle.load(open(f'{dir}/6B.300_idx.pkl', 'rb'))
    id_to_word = list(word_to_id.keys())
    data = {w: vectors[word_to_id[w]] for w in words}
    n_embs = len(data.keys())
    return data, word_to_id, id_to_word, n_embs


# Write data to pkl for one section
def write_data(in_path, split_path, section, output_path):
    doclst = []
    filenames = pd.read_csv(split_path, sep='\t')
    n_filtered = 0

    for i, row in filenames.iterrows():
        if not os.path.exists(in_path+row['ID']+".json"):
            print("File", row['ID'], "doesn't exist.")
            n_filtered += 1
            continue

        with open(in_path+row['ID']+".json") as f:
            data = json.load(f)

        # Convert bracket format to segments list
        seg_lst = []
        segments = re.findall(r'\[ .*? \]', data['segments'])
        for s in segments:
            seg_lst.append(s[2:-2])

        seg_token_lst = [seg.split() for seg in seg_lst]
        seg_token_lst = [seg_tokens for seg_tokens in seg_token_lst if(len(seg_tokens) != 0)]

        doclst.append({
            "id": data['ID'],
            "topic": data['topic'] ,
            "source": data['source'],
            "url": data['url'],
            "source_url": data['source_url'],
            "segments": seg_token_lst,
            "original_content": data['content'],
            "authors": data['authors'],
            "title": data['title'],
            "int_label": data['bias'],
            "text_label": data['bias_text']
        })

    print('n_filtered in {}: {}'.format(section, n_filtered))
    print('n_documents in {}: {}'.format(section, len(doclst)))

    with open(output_path, "wb") as output_file:
        pickle.dump(doclst, output_file)
        

def read_data(dir, section, word_to_id, unk_idx, device="cpu", max_seg_length=-1, max_doc_length=-1, min_media_count=1000):
    ret = list()
    special_words = network.special_tokens.Dict(word_to_id, unk_idx)
    path = os.path.join(dir, section + ".pkl")
    data = pickle.load(open(path, 'rb'))
    source_to_id = dict()
    count_source = dict()

    # first loop to count media
    for doc in data:
        # Skip documents using document/segments length condition
        skip = False
        if max_doc_length > 0:
            if len(doc["segments"]) > max_doc_length:
                skip = True

        if max_seg_length > 0:
            for s in doc["segments"]:
                if len(s) > max_seg_length:
                    skip = True
                    break

        if skip:
            continue

        if doc["source"] in count_source.keys():
            count_source[doc["source"]] += 1
        else:
            count_source[doc["source"]] = 1

    for k, v in count_source.items():
        if v >= min_media_count: # temp change : was 1000
            source_to_id[k] = len(source_to_id.keys())

    source_to_id["other"] = len(source_to_id.keys())

    def media_to_id(media_name):
        if media_name in source_to_id.keys():
            return source_to_id[media_name]
        else:
            return len(source_to_id.keys()) - 1

    for doc in data:
        in_data = dict()

        # Skip documents using document/segments length condition
        skip = False
        if max_doc_length > 0:
            if len(doc["segments"]) > max_doc_length:
                skip = True

        if max_seg_length > 0:
            for s in doc["segments"]:
                if len(s) > max_seg_length:
                    skip = True
                    break

        if skip:
            continue

        segments_words_ids = []
        for s in doc["segments"]:
            words = [word_to_id.get(w.lower(), unk_idx) for w in s]
            segments_words_ids.append(words)
            
        in_data["segments"] = segments_words_ids

        segments_special_words_ids = []
        for s in doc["segments"]:
            special_words_lst = [special_words.to_id(w.lower()) for w in s]
            segments_special_words_ids.append(special_words_lst)
        in_data["special_words"] = segments_special_words_ids

        in_data["label"] = doc["int_label"]
        in_data["original_content"] = doc["original_content"]
        in_data["original_segmented"] = doc["segments"]
        in_data["title"] = doc["title"]
        in_data["source"] = media_to_id(doc["source"])
        in_data["source_string"] = doc["source"]
        in_data["url"] = doc['source_url']
        ret.append(in_data)

    return ret, source_to_id
    

def get_input_dict(batch, tokens_padding_idx, device='cpu'):
    batch_size = len(batch)

    special_words_padding_idx = len(network.special_tokens.Dict())

    doc_l_matrix = np.zeros([batch_size], np.int32)

    for i, document in enumerate(batch):
        n_segments = len(document["segments"])
        doc_l_matrix[i] = n_segments

    max_doc_l = np.max(doc_l_matrix)
    max_segment_l = max([max([len(segment) for segment in doc["segments"]]) for doc in batch])
    tokens_idxs_matrix = np.full([batch_size, max_doc_l, max_segment_l], tokens_padding_idx)
    special_words_idxs_matrix = np.full([batch_size, max_doc_l, max_segment_l], special_words_padding_idx)
    segments_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
    gold_matrix = np.zeros([batch_size], np.int32)

    # Padding for documents, we set to 0 padded values for segments
    mask_segments_matrix = np.ones([batch_size, max_doc_l], np.float64)

    # Media adaptation
    media_matrix = np.zeros([batch_size], np.int32)

    # Masked heads for structure explanations via perturbations
    mask_heads_matrix = np.ones([batch_size, max_doc_l], np.float64)
    
    for i, document in enumerate(batch):
        n_segments = len(document["segments"])
        gold_matrix[i] = document["label"]
        
        if "source" in document.keys():
            media_matrix[i] = document["source"]
            
        if "masked_heads" in document.keys():
            mask_heads_matrix[i] = document["masked_heads"]

        for j, segment in enumerate(document["segments"]):
            tokens_idxs_matrix[i, j, :len(segment)] = np.asarray(segment)
            segments_l_matrix[i, j] = len(segment)
        for j, special_words in enumerate(document["special_words"]):
            special_words_idxs_matrix[i, j, :len(special_words)] = np.asarray(special_words)
        mask_segments_matrix[i, n_segments:] = 0
    mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float64)
    mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float64)
    mask_parser_1[:, :, 0] = 0 # zero out 1st column for each doc
    mask_parser_2[:, 0, :] = 0 # zero out 1st row for each doc

    input_dict = { 'token_idxs': torch.LongTensor(tokens_idxs_matrix).to(device),
                   'special_words_idxs': torch.LongTensor(special_words_idxs_matrix).to(device),
                   'segments_l': torch.LongTensor(segments_l_matrix).to(device),
                   'mask_segments': torch.FloatTensor(mask_segments_matrix).to(device),
                   'doc_l': torch.LongTensor(doc_l_matrix).to(device),
                   'gold_labels': torch.LongTensor(gold_matrix).to(device),
                   'max_segment_l': max_segment_l,
                   'max_doc_l': max_doc_l,
                   'mask_parser_1': torch.FloatTensor(mask_parser_1).to(device),
                   'mask_parser_2': torch.FloatTensor(mask_parser_2).to(device),
                   'batch_l': batch_size,
                   'media_labels': torch.LongTensor(media_matrix).to(device),
                   'mask_heads_matrix': mask_heads_matrix }

    return input_dict
