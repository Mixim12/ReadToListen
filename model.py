import torch
import torch.nn as nn
from encoder_decoder import Encoder, Decoder 

class TransformerTTS(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len, dropout):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_len, dropout)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(-2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != 0).unsqueeze(-2)
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        final_output = self.final_layer(dec_output)
        return final_output
