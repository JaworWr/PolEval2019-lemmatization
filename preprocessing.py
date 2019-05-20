from tokenizer import Tokenizer, sentence_to_list
from xml.etree import ElementTree
import ufal.udpipe
import pandas as pd
import os
import argparse

def first_with_prefix(l, pref):
    for i, s in enumerate(l):
        if s.startswith(pref):
            return i
        
def non_whitespace(s):
    return sum(not c.isspace() for c in s)

class SentenceLoader:
    def __init__(self, data_path, tokenizer, index_path, device=None, print_broken_tags=[], update_freqs=True):
        self.tokenizer = tokenizer

        # load the TSV file
        with open(index_path) as file:
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
        
        # load the XML files
        self.text_data = []
        self.data = []
        sent_id = 0
        for f in os.listdir(data_path):
            if f[-4:] == '.xml':
                with open(data_path + '/' + f) as file:
                    t = ElementTree.parse(file)
                    root = t.getroot()
                    self._entire_text = root.text
                    self._phrases = []
                    self._i = non_whitespace(root.text)
                    for c in root.getchildren():
                        self._process_xml_elem(c)
                    if root.tail is not None:
                        self._entire_text += root.tail
                    
                    sentences = tokenizer.tokenize(self._entire_text)
                    i = 0
                    for s in sentences:
                        s_text = sentence_to_list(s)
                        j = i + sum(len(w) for w in s_text)
                        ids = []
                        starts = []
                        ends = []
                        for p in self._phrases:
                            if i <= p[1] and p[2] <= j:
                                phrase_s = phrase_e = 0
                                pos = p[1] - i
                                while pos > 0:
                                    pos -= len(s_text[phrase_s])
                                    phrase_s += 1
                                pos_flag = pos < 0
                                phrase_s -= pos_flag
                                pos = p[2] - i
                                while pos > 0:
                                    pos -= len(s_text[phrase_e])
                                    phrase_e += 1
                                self.text_data.append((s_text, p[0], (phrase_s, phrase_e)))
                                ids.append(p[0])
                                starts.append(phrase_s)
                                ends.append(phrase_e)
                        if len(ids) > 0:
                            s.comments.append('# doc_id = {:d}'.format(self.phrase_data.loc[ids[0]].document_id))
                            s.comments.append('# phrase_id = ' + ",".join([str(ind) for ind in ids]))
                            s.comments.append('# phrase_s = ' + ",".join([str(ind) for ind in starts]))
                            s.comments.append('# phrase_e = ' + ",".join([str(ind) for ind in ends]))
                            ind = first_with_prefix(s.comments, '# sent_id')
                            s.comments[ind] = '# sent_id = {:d}'.format(sent_id)
                            self.data.append(s)
                        i = j
                        sent_id += 1
                                
        del self._phrases
        del self._i
        del self._entire_text

    def _process_xml_elem(self, e):
        old_i = self._i
        if e.text is not None:
            self._entire_text += e.text
            self._i += non_whitespace(e.text)
        for c in e.getchildren():
            self._process_xml_elem(c)
        self._phrases.append((int(e.attrib['id']), old_i, self._i))
        if e.tail is not None:
            self._entire_text += e.tail
            self._i += non_whitespace(e.tail)
            
    def __iter__(self):
        yield from self.text_data
        
    def tag_and_parse(self):
        for sample in self.data:
            self.tokenizer.tag(sample)
            self.tokenizer.parse(sample)
        
    def write(self, path, f):
        text = self.tokenizer.write(self.data, f)
        with open(path, 'w') as file:
            file.write(text)

def main(args):
    tok = Tokenizer(args['tokenizer_model'])
    loader = SentenceLoader(args['data'], tok, args['index'])
    if args['parse']:
        loader.tag_and_parse()
    loader.write(args['output'], 'conllu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_model', help='path to tokenizer model', type=str, required=True)
    parser.add_argument('--data', help='path to data in the XML format', type=str, required=True)
    parser.add_argument('--index', help='path to data index', type=str, required=True)
    parser.add_argument('--output', help='output path', type=str, required=True)
    parser.add_argument('--parse', help='use UDPipe parser', action='store_true')
    args = vars(parser.parse_args())
    main(args)