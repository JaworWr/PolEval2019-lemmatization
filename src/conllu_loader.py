import pandas as pd
import os
import sys
import torch
from random import shuffle
from collections import defaultdict
import conllu

word_to_ix = dict()
tag_to_ix = dict()
extra_to_ix = dict()
ix_to_tag = None
word_freqs = defaultdict(lambda: 0)

class TagError(Exception):
    def __init__(self, reason, *args):
        super(TagError, self).__init__(*args)
        self.reason = reason

def save_dicts(path):
    d = dict(
        word_to_ix=word_to_ix,
        tag_to_ix=tag_to_ix,
        extra_to_ix=extra_to_ix,
        word_freqs=dict(word_freqs)
    )
    torch.save(d, path)

def load_dicts(path):
    global word_to_ix, tag_to_ix, extra_to_ix, word_freqs
    d = torch.load(path)
    word_to_ix = d['word_to_ix']
    tag_to_ix = d['tag_to_ix']
    extra_to_ix = d['extra_to_ix']
    word_freqs = defaultdict(lambda: 0, d['word_freqs'])
    calc_ix_to_tag()

def calc_ix_to_tag():
    global ix_to_tag
    ix_to_tag = [None] * len(tag_to_ix)
    for tag, ix in tag_to_ix.items():
        ix_to_tag[ix] = tag

def process_tag(tag):
    tags = tag[1:-1].split()
    if len(tags) == 0:
        raise TagError('empty')
    if tags[0] == 'ERROR':
        raise TagError('error')
    return tags

def register_words(line):
    for w in line:
        word_freqs[w] += 1

def process_word(w, suffix_len=3, prefix_len=0, threshold=20):
    if not w.isalpha() or word_freqs[w] >= threshold:
        return w

    if prefix_len > 0:
        pref = w[:prefix_len] + '|'
    else:
        pref = ""

    if not w[1:].islower():
        case = 's'
    elif w[0].isupper():
        case = 'c'
    else:
        case = 'l'
    return case + '|' + pref + w[-suffix_len:]

def process_line(line, **kw_args):
    return [process_word(w, **kw_args) for w in line]

def validate_tag(tag, phrase_b, phrase_e):
    phrase_span = phrase_e - phrase_b
    if len(tag) != phrase_span:
        raise TagError('len')

def prepare_sequence(seq, to_ix, device='cpu'):
    seq = [to_ix[x] for x in seq]
    seq = torch.tensor(seq, dtype=torch.long, device=device)
    return seq

def parse_metadata(s):
    return [int(x) for x in s.split(',')]

def text_property(text, prop):
    return [w[prop] for w in text]

def process_feats(feats):
    if feats is None:
        return ""
    return "|".join(k + "=" + v for k, v in feats.items())

class ConlluLoader:
    def __init__(self, data, index=None, device=None, print_broken_tags=[], update_freqs=True):
        global word_to_ix, tag_to_ix
        
        if device is not None:
            self.device = device
        else:
            device = self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.broken_tags = 0

        # load the TSV file
        with open(index) as file:
            lines = []
            for line in file:
                line = line.strip().split('\t')
                line[0] = int(line[0])
                line[1] = int(line[1])
                if len(line) < 5:
                    line.append('-')
                lines.append(line)
            self.phrase_data = pd.DataFrame(lines, columns=['phrase_id', 'document_id', 'phrase', 'lemma', 'tag'])
            assert self.phrase_data['phrase_id'].is_unique
            self.phrase_data.set_index('phrase_id', inplace=True)
        
        # load the conllu file
        self.text_data = []
        with open(data) as file:
            text = file.read()
            for sample in conllu.parse(text):
                if update_freqs:
                    register_words(text_property(sample, 'form'))
                phrase_id = parse_metadata(sample.metadata['phrase_id'])
                phrase_s = parse_metadata(sample.metadata['phrase_s'])
                phrase_e = parse_metadata(sample.metadata['phrase_e'])
                for p_id, ps, pe in zip(phrase_id, phrase_s, phrase_e):
                    try:
                        tag = self.phrase_data.loc[p_id]['tag']
                        doc_id = self.phrase_data.loc[p_id]['document_id']
                        tag = process_tag(tag)
                        validate_tag(tag, ps, pe)
                    except KeyError:
                        pass
                    except TagError as e:
                        self.broken_tags += 1
                        if e.reason in print_broken_tags:
                            print("Tag error in {:d}:{:d}".format(doc_id, p_id), file=sys.stderr)
                            print(sample, file=sys.stderr)
                            print(self.phrase_data.loc[p_id]['phrase'], file=sys.stderr)
                            print(tag, file=sys.stderr)
                    else:
                        for t in tag:
                            if t not in tag_to_ix:
                                tag_to_ix[t] = len(tag_to_ix)
                        self.text_data.append((sample, p_id, tag, (ps, pe)))

    def prepare_data(self, extra_data=['deprel', 'upostag'], include_feats=True, include_head=True, **kw_args):
        self.data = []
        for sample, _, tag, bounds in self.text_data:
            text = text_property(sample, 'form')
            text = process_line(text, **kw_args)
            extras = [text_property(sample, e) for e in extra_data]
            if include_feats:
                extras.append([process_feats(w['feats']) for w in sample])
            if include_head:
                heads = text_property(sample, 'head')
                heads = [str(bounds[0] <= int(h) - 1 < bounds[1]) for h in heads]
                extras.append(heads)
            extras = ["|".join(p) for p in zip(*extras)]
            for w in text:
                if w not in word_to_ix:
                    word_to_ix[w] = len(word_to_ix)
            for r in extras:
                if r not in extra_to_ix:
                    extra_to_ix[r] = len(extra_to_ix)
            p_text = prepare_sequence(text, word_to_ix, self.device)
            p_tag = prepare_sequence(tag, tag_to_ix, self.device)
            p_extras = prepare_sequence(extras, extra_to_ix, self.device)
            self.data.append((p_text, p_tag, p_extras, bounds))

    def shuffle(self):
        shuffle(self.data)

    def __iter__(self):
        yield from self.data
        
    def iter_phrases(self):
        for t, d in zip(self.text_data, self.data):
            s = self.phrase_data.loc[t[1]]
            yield s, d[0], d[2], d[3], t[0]

