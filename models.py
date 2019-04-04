import torch
from torch import nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, device='cuda'):
        super(BiLSTM_CRF, self).__init__()
        self.tagset_size = tagset_size
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layer
        self.half_hidden = hidden_dim // 2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        
        # Mapping lstm output to tags
        self.fc = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size)
        
        self.device = device
        self.to(device)
        self.init_hidden()
        
    def init_hidden(self):
        self.hidden = (
            torch.zeros(2, 1, self.half_hidden, device=self.device),
            torch.zeros(2, 1, self.half_hidden, device=self.device)
        )
        
    def nll(self, sentence, phrase_bounds, tags):
        feats = self._lstm_feats(sentence)
        b, e = phrase_bounds
        feats = feats[b:e, :]
        ll = self.crf(feats.view(-1, 1, self.tagset_size), tags.view(-1, 1))
        return -ll
        
    def forward(self, sentence, phrase_bounds):
        feats = self._lstm_feats(sentence)
        b, e = phrase_bounds
        feats = feats[b:e, :]
        tags = self.crf.decode(feats.view(-1, 1, self.tagset_size))
        return torch.tensor(tags, device=self.device)
    
    def _lstm_feats(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        feats = self.fc(lstm_out)
        return feats
    
class BiLSTM_CRF_extra(nn.Module):
    def __init__(self, word_embed_dim, extra_embed_dim, hidden_dim, vocab_size, extra_size, tagset_size, device='cuda'):
        super(BiLSTM_CRF_extra, self).__init__()
        self.tagset_size = tagset_size
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.extra_embedding = nn.Embedding(extra_size, extra_embed_dim)
        
        # Bidirectional LSTM layer
        self.half_hidden = hidden_dim // 2
        self.lstm = nn.LSTM(word_embed_dim + extra_embed_dim, hidden_dim // 2, bidirectional=True)
        
        # Mapping lstm output to tags
        self.fc = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size)
        
        self.device = device
        self.to(device)
        self.init_hidden()
        
    def init_hidden(self):
        self.hidden = (
            torch.zeros(2, 1, self.half_hidden, device=self.device),
            torch.zeros(2, 1, self.half_hidden, device=self.device)
        )
        
    def nll(self, sentence, extra, phrase_bounds, tags):
        feats = self._lstm_feats(sentence, extra)
        b, e = phrase_bounds
        feats = feats[b:e, :]
        ll = self.crf(feats.view(-1, 1, self.tagset_size), tags.view(-1, 1))
        return -ll
        
    def forward(self, sentence, extra, phrase_bounds):
        feats = self._lstm_feats(sentence, extra)
        b, e = phrase_bounds
        feats = feats[b:e, :]
        tags = self.crf.decode(feats.view(-1, 1, self.tagset_size))
        return torch.tensor(tags, device=self.device)
    
    def _lstm_feats(self, sentence, extra):
        embeds = self.embedding(sentence)
        extra_embeds = self.extra_embedding(extra)
        embeds = torch.cat([embeds, extra_embeds], 1)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        feats = self.fc(lstm_out)
        return feats   
                
