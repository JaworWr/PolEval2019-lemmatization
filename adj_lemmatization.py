import sys

f_ladny, f_ladna, f_ladne, f_ladnej, f_ladnom, f_ladnemu, f_ladnego, f_ladni, f_ladnych, f_ladnymi, f_ladnym = range(11)

forms = {}

for x in open('data/przymiotniki.txt'):
    L = x.split()
    
    for s in L:
        forms[s] = [L[f_ladny], L[f_ladna], L[f_ladne], L[f_ladni]]
        
forms['scy'] = ['ski', 'ska', 'skie', 'scy']
forms['dcy'] = ['dki', 'dka', 'dkie', 'dcy']
forms['tcy'] = ['tki', 'tka', 'tkie', 'tcy']

rule = {
    'nom:sg:m3' : 0, 
    'nom:sg:m1' : 0, 
    'nom:sg:m2' : 0, 
    'nom:sg:f'  : 1, 
    'nom:sg:n'  : 2,     
    'nom:pl:m1' : 3, 
    'nom:pl:m2' : 2, 
    'nom:pl:m3' : 2, 
    'nom:pl:f'  : 2, 
    'nom:pl:n'  : 2     
}


names = ['nom:sg:f', 'nom:sg:m1', 'nom:sg:m2', 'nom:sg:m3', 'nom:sg:n', 
         'nom:pl:f', 'nom:pl:m1', 'nom:pl:m2', 'nom:pl:m3', 'nom:pl:n']


def gen_adj(word, r):
    for i in range(1, 8)[::-1]:
        if word[-i:] in forms:
            pref = word[:-i]
            ss = forms[word[-i:]]    
            return pref + ss[rule[r]]
    return '?'

def adj_rule_for_words(w1, w2):
    for r in rule:
        if gen_adj(w1, r) == w2:
            return r
    return '?'     
           
if __name__ == "__main__":
    for x in open('/home/prych/data/tagsAndBases.dat'):
        if 'adj:' not in x:
            continue
        word = x.split()[0]
        for r in rule:
            f = gen_adj(word, r)
            if f == '?':
                print (word)
            #print ('   ', r, gen_adj(word, r))
        
         
