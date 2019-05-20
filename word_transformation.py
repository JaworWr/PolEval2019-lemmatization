# -*- encoding: utf8 -*-
from adj_lemmatization import gen_adj
from morf_tools import generate_forms, guess_lemma_with_special
import random

def capitalize(word, c):
    if c == 'l':
        return word.lower()
    elif c == 'c':
        return word[0].upper() + word[1:]
    else:
        return word

def application(w, op):
    if op == '=':
        return w
    
    comma = ''
    if w[-1] in ',': # '.,':
        comma = w[-1]
        w = w[:-1]
            
    if 'adj:' in op:
        return gen_adj(w.lower(), op.replace('adj:','')) # + comma
    
    if 'minus-' in op:
        op = op.replace('minus-', '')
        n = int(op)
        return w[:-n] + comma
                
    if 'subst:' in op:
        L = op.split(':')
        L[2:2] = ['nom']
        op = ':'.join(L)
        #print op
        
        for a in u"’'":
            if a in w:
                return a.join(w.split(a)[:-1]) + comma
        
        L = w.split('-')
        if len(L) == 2 and L[1] in ['u', 'em', 'owi', 'a']:
            return L[0] + comma
                
        # forms = list(generate_forms(w, op) - {w})
        forms = list(generate_forms(w, op))
        if forms:
            return random.choice(forms) + comma #TODO: choose more frequent
        else:
            # return '<guessed-form>'   
            L1 = op.split(':')
            candidates = [':'.join(L1[:-1])]
            if L1[-1] in ('m1', 'm2', 'm3'): # wrong masculine
                for g in {'m1', 'm2', 'm3'} - {L1[-1]}:
                    candidates.append(candidates[0] + ':' + g)
            for op1 in reversed(candidates):
                forms = list(generate_forms(w, op1))
                # forms = list(generate_forms(w, op1) - {w})
                if forms:
                    return random.choice(forms) + comma #TODO: choose more frequent
            return '<guessed-form>'
    
    if 'form-' in op:
        last = op.split('-')[-1]
        
        return guess_lemma_with_special(w, last) + comma

    return '<guessed-form2>'

def application_c(word, op):
    c, op1 = op.split('|')
    lemma = application(word, op1)
    return capitalize(lemma, c)

#print application('Fryderyka', 'subst:sg:m1')
#sys.exit(0)

if __name__== "__main__":
    good = 0.0
    cnt = 0.0

    tests = []
        
    for x in open('index_pl_v2.tsv'):
        n1,n2, phrase, lemma, ops = x.split('\t')
        lemma = lemma.split()
        phrase = phrase.split()
        
        ops = ops.strip()[1:-1].split()
        
        if len(ops) == len(lemma) == len(phrase):
            c = 0
            for i in range(len(ops)):
                p, lem, op = phrase[i], lemma[i], ops[i]
                tests.append( (p, lem, op))
                res = application(p, op)
                if lem.lower() == res.lower():
                    c += 1
            if c == len(ops):
                good += 1
            cnt += 1
                
    print( good / cnt)
                
    good = 0.0
    cnt = 0.0
                
    for p, lem, op in tests:
        res = application(p, op)
        if lem.lower() == res.lower():
            good += 1
        else:
            pass
            print ("%s gt=%s %s --> res=%s" % (p, lem, op, res))
        cnt += 1
        
    print( good/cnt  )              
    
