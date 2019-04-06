# PolEval2019-lemmatization
## Requirements
-Pandas  
-PyTorch  
-[pytorch-crf](https://github.com/kmkurn/pytorch-crf)  
-[UDPipe](https://pypi.org/project/ufal.udpipe/)  
-[CoNLL-U Parser](https://github.com/EmilStenstrom/conllu)  
-[marisa-trie](https://github.com/pytries/marisa-trie)  
-[Morfeusz2](http://sgjp.pl/morfeusz/index.html)

## Usage
`main.py --model model3.pt --tokenizer_model <udpipe-model> --data poleval_task2_test.conllu --index index.tsv --output <output-file>`

The data was tokenized using the [UDPipe](http://ufal.mff.cuni.cz/udpipe) tool and parsed with the [COMBO](https://github.com/360er0/COMBO) parser. The program also uses the [Morfeusz2](http://sgjp.pl/morfeusz/) tool.

## Preprocessing
`preprocessing.py --tokenizer_model <udpipe-model> --data <path-to-xml-data> --index <path-to-index> --output <output-file>`
