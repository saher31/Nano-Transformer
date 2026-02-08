import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

token_to_id={
    'what':0,
    'is':1,
    'your':2,
    'name':3,
    'saher':4,
    '<EOS>':5,
}
id_to_token = dict(map(reversed, token_to_id.items()))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        pe=torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index=torch.arange(start=0,end= d_model,step= 2).float()
        div_term=1/torch.tensor(10000.0)**(embedding_index/d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]


class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model,bias=False)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model,bias=False)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model,bias=False)
        self.row_dim=0
        self.col_dim=1

    def forward(self,encodings_for_q,encodings_for_k,encodings_for_v,mask=None):
        q=self.W_Q(encodings_for_q)
        k=self.W_K(encodings_for_k)
        v=self.W_V(encodings_for_v)

        sims=torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask,value=-1e9)
        attention_percent=F.softmax(scaled_sims,dim=self.col_dim)
        attention_score=torch.matmul(attention_percent,v)
        return attention_score



class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_tokens=len(token_to_id),d_model=2,max_len=6):
        super().__init__()
        self.we=nn.Embedding(num_embeddings=num_tokens,embedding_dim=d_model)
        self.pe=PositionalEncoding(d_model=d_model,max_len=max_len)
        self.attention=Attention(d_model=d_model)
        self.fc_layer=nn.Linear(in_features=d_model,out_features=num_tokens)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,tokens_ids):
        word_embeddings=self.we(tokens_ids)
        position_encoding=self.pe(word_embeddings)

        mask = torch.tril( torch.ones((tokens_ids.size(0), tokens_ids.size(0)),
        device=tokens_ids.device ))
        mask=mask== 0
        self_attention_values=self.attention(position_encoding,
                                            position_encoding,
                                            position_encoding
                                            ,mask=mask)
        residual_connections_values=position_encoding+self_attention_values
        fc_layer_output=self.fc_layer(residual_connections_values)
        return fc_layer_output

model = DecoderOnlyTransformer()
model.load_state_dict(torch.load("model.pth"))
model.eval()        

def predict():
    model_input = torch.tensor([
        token_to_id['what'],
        token_to_id['is'],
        token_to_id['your'],
        token_to_id['name'],
        token_to_id['<EOS>']
    ])

    input_length = model_input.size(0)
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    predicted_ids = predicted_id

    max_length = 6
    for i in range(input_length, max_length): 
        if (predicted_id == token_to_id["<EOS>"]):
            break

        model_input = torch.cat((model_input, predicted_id))
        predictions = model(model_input)
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    # بدل الطباعة → رجوع النتيجة
    result = []
    for id in predicted_ids:
        result.append(id_to_token[id.item()])

    return result

