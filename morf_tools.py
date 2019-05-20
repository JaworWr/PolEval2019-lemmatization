# -*- encoding: utf8 -*-

from collections import defaultdict as dd
import morfeusz2
import sys

data_prefix = 'data/'
special = 'aeyow'

U = {}
M = morfeusz2.Morfeusz()
lemma_rules = {}

adj_suff = set()

for x in open(data_prefix + 'koncowki_przymiotnikowe.txt'):
    adj_suff.add(x.strip())
    

for x in open(data_prefix + 'lemmatization_rules_with_aeyow.txt'):
    l,r = x.split()
    
    if l != r:
        lemma_rules[l] = r 
        
for x in open(data_prefix + 'extra_lemmatization_rules.txt'):
    l,r = x.split()
    
    if l != r:
        lemma_rules[l] = r 


def letter_type(a):
    if a not in special:
        return 'x'
    return a

def guess_lemma_with_special(w, last_letter):
    last_letter = letter_type(last_letter)
        
    for i in range(len(w)):
        pref, suf = w[:i], w[i:]
        key = suf + '@' + last_letter
        if key in lemma_rules:
            return pref + lemma_rules[key]
    return '' 

def is_adj(w):
    w = '^^^^' + w
    return w[-5:] in adj_suff
    

def get_all_bases(word):
    res = set()
    for a,b,des in M.analyse(word):
        orth,base,tag, p1, p2 = des
        if tag != 'ign':
            res.add(base)
    return res  

def forms_for_base(base, form):
    candidates = M.generate(base)
    res = set()
    for tpl in candidates:
        w, base, tags, p1, p2 = tpl         
        for tag in M._expand_tag(tags):
            #print 'tag=', tag
            if form in tag:
                res.add(w)
    return res  
    
def generate_forms(word, form):
    bs = get_all_bases(word)
    #print word, bs
    res = set()
    for b in bs:
        res.update(forms_for_base(b, form))    
    return res

def get_all_tags(word):
    tags = set()
    for a,b,des in M.analyse(word):
        orth,base,tag, p1, p2 = des
        tags.add(tuple(tag.split(':')))
    return tags