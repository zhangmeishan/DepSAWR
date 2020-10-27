import torch
import torch.nn as nn


def drop_tri_input_independent(word_embeddings, tag_embeddings, context_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    context_masks = context_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    context_masks = torch.bernoulli(context_masks)
    scale = 3.0 / (word_masks + tag_masks + context_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    context_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    context_masks = context_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    context_embeddings = context_embeddings * context_masks

    return word_embeddings, tag_embeddings, context_embeddings


def drop_bi_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    scale = 2.0 / (word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings


def drop_uni_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.new_full((batch_size, hidden_size), 1 - dropout)
    drop_masks = torch.bernoulli(drop_masks)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


def default_init(tensor):
    if tensor.ndimension() == 1:
        nn.init.constant(tensor, val=0.0)
    else:
        nn.init.xavier_normal(tensor)

    return tensor


def embedding_init(tensor, val=0.1):
    nn.init.uniform(tensor, -val, val)

    return tensor


def rnn_init(tensor):
    if tensor.ndimension() != 2:
        return default_init(tensor)

    r, c = tensor.size()

    if r % c == 0:
        dim = 0
        n = r // c
        sub_size = (c, c)
    elif c % r == 0:
        dim = 1
        n = c // r
        sub_size = (r, r)
    else:
        return default_init(tensor)

    sub_tensors = [torch.Tensor(*sub_size).normal_(0, 1) for _ in range(n)]
    sub_tensors = [torch.svd(w, some=True)[0] for w in sub_tensors]

    init_tensor = torch.cat(sub_tensors, dim=dim)  # [r, c]

    tensor.copy_(init_tensor)

    return tensor
