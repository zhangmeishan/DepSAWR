from transformers.modeling_bert import *


class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

    def forward(self, bert_indices, bert_segments):
        attention_mask = torch.ones_like(bert_indices)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)

        last_output, encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        return encoder_outputs


class BertExtractor(nn.Module):
    def __init__(self, config):
        super(BertExtractor, self).__init__()
        self.config = config
        self.bert = MyBertModel.from_pretrained(config.bert_path)
        self.bert.encoder.output_hidden_states = config.output_hidden_states
        self.bert.encoder.output_attentions = config.output_attentions
        self.bert_hidden_size = self.bert.config.hidden_size
        self.bert_layers = config.bert_layers
        self.start_layer = self.bert.config.num_hidden_layers + 1 - self.bert_layers
        self.end_layer = self.bert.config.num_hidden_layers
        self.tune = True if config.bert_tune == 1 else False

        for p in self.bert.named_parameters():
            p[1].requires_grad = self.tune

    def forward(self, bert_indices, bert_segments, bert_pieces):
        all_outputs = self.bert(bert_indices, bert_segments)
        outputs = []
        if self.config.bert_tune == 1:
            cur_output = torch.bmm(bert_pieces, all_outputs[self.end_layer-1])
            outputs.append(cur_output)
        else:
            for idx in range(self.start_layer, self.end_layer+1):
                cur_output = torch.bmm(bert_pieces, all_outputs[idx])
                outputs.append(cur_output)

        return outputs


