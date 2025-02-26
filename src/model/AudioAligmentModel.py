import torch
import torch.nn as nn
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor = 4.0, dropout = 0.2):
        super(FeedForwardModule, self).__init__()
        self.fnn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + 0.5 * self.fnn(x)

class MutliHead_SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super(MutliHead_SelfAttention,self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self,x):
        input = x
        x,_ = self.attention(x,x,x)
        return input + x

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size = 31, dropout = 0.1): #10 - fixed value
        super(ConvolutionModule,self).__init__()
        self.conv = nn.Sequential(
            #nn.LayerNorm(d_model),
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1, groups = d_model),
            nn.GLU(dim = 1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            Swish(),
            nn.Conv1d(d_model, d_model, kernel_size=1, groups=d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        input = x
        #print(x.size())
        x = self.conv(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x + input
    
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_expansion_factor = 4, kernel_size = 31, dropout = 0.1):
        super(ConformerBlock, self).__init__()
        self.fnn1 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.self_attention = MutliHead_SelfAttention(d_model, nhead, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.fnn2 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.fnn1(x)
        x = self.self_attention(x)
        x = self.conv(x)
        x = self.fnn2(x)
        return self.norm(x)

class AudioAligmentModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks = 4, d_model = 256, nhead = 2, ffn_expansion_factor = 4, 
                 kernel_size = 31, dropout = 0.1):
        super(AudioAligmentModel, self).__init__()
        self.d_model = d_model
        self.conv_layer = nn.Conv1d(input_dim, d_model, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(d_model)
        self.conformer = nn.ModuleList([
            ConformerBlock(d_model, nhead,ffn_expansion_factor, kernel_size, dropout)
            for i in range(num_blocks)
        ])
        self.final_block1 = ConformerBlock(d_model, nhead,ffn_expansion_factor, kernel_size, dropout)
        self.final_block2 = ConformerBlock(d_model, nhead,ffn_expansion_factor, kernel_size, dropout)
        self.output1 = nn.Linear(d_model,num_classes)
        self.output2 = nn.Linear(d_model, num_classes)
        #self.output = nn.Linear(d_model, num_classes)
        self.log_softmax =  nn.LogSoftmax(dim = 1)

    def forward(self, audio, **batch):
        # print(audio.size())
        x = self.conv_layer(audio)
        # print(x.size())
        #print(x.size())
        #x = self.linear(x)
        x = x.permute(0,2,1)
        #print(x.size())
        x = self.dropout(x)
        x = self.positional_encoding(x)
        #print(x.size())target_size
        for block in self.conformer:
            x = block(x)
        #output = self.output(x)
        out_false = self.output1(self.final_block1(x))
        out_true = self.output2(self.final_block2(x))
        stacked = torch.stack([out_false, out_true], dim=1)
        output_logits = self.log_softmax(stacked)
        return output_logits
        #return torch.sigmoid(output)
        #return output
        #return self.output(x)