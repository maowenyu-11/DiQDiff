import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
import faiss
import copy
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# class KMeans(object):
#     def __init__(self, num_cluster, seed, hidden_size, gpu_id, device):
#         """
#         Args:
#             k: number of clusters
#         """
#         self.seed = seed
#         self.num_cluster =num_cluster
#         self.max_points_per_centroid = 4096
#         self.min_points_per_centroid = 0
#         self.gpu_id = gpu_id
#         self.device = device
#         self.first_batch = True
#         self.hidden_size = hidden_size
#         self.clus, self.index = self.__init_cluster(self.hidden_size)
#         self.centroids = []

#     def __init_cluster(
#         self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
#     ):
#         print(" cluster train iterations:", niter)
#         clus = faiss.Clustering(hidden_size, self.num_cluster)
#         clus.verbose = verbose
#         clus.niter = niter
#         clus.nredo = nredo
#         clus.seed = self.seed
#         clus.max_points_per_centroid = max_points_per_centroid
#         clus.min_points_per_centroid = min_points_per_centroid

#         res = faiss.StandardGpuResources()
#         res.noTempMemory()
#         cfg = faiss.GpuIndexFlatConfig()
#         cfg.useFloat16 = False
#         cfg.device = int(self.gpu_id)
#         index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
#         return clus, index

#     def train(self, x):
#         # train to get centroids
#         if x.shape[0] > self.num_cluster:
#             self.clus.train(x, self.index)
#         # get cluster centroids
#         centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
#         # convert to cuda Tensors for broadcast
#         centroids = torch.Tensor(centroids).to(self.device)
#         self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

#     def query(self, x):
#         # self.index.add(x)
#         D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
#         seq2cluster = [int(n[0]) for n in I]
#         # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
#         seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
#         return seq2cluster, self.centroids[seq2cluster]


# def KMeans(X, n_clusters, n_iters,device):
#    centroids = X[torch.randperm(len(X))[:n_clusters]].to(device)
   
#    for _ in range(n_iters):
#        distances = torch.cdist(X, centroids).to(device)
#        labels = torch.argmin(distances, dim=1).to(device)
       
#        for i in range(n_clusters):
#            centroids[i] = X[labels == i].mean(dim=0)
   
#    return centroids, labels

def KMeans(input, n_clusters, n_iters):
    # 随机初始化聚类中心
    X=input.view(input.shape[0],-1)
    centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)
    
    for _ in range(n_iters):
        distances = torch.cdist(X, centers).to(input.device)
        labels = torch.argmin(distances, dim=1).to(input.device)
        for i in range(n_clusters):
            if torch.sum(labels == i) == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = torch.mean(X[labels == i], dim=0)
       
    return centers.view(n_clusters,input.shape[1],input.shape[-1]), labels



def k_medoids(input, n_clusters, max_iter):
    X=input.view(input.shape[0],-1)
    medoids_idx = torch.randperm(X.size(0))[:n_clusters]
    medoids = X[medoids_idx]

    for _ in range(max_iter):
        # Assign each point to the closest medoid
        distances = torch.cdist(X, medoids)
        _, labels = torch.min(distances, dim=1)

        # Update medoids
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if cluster_points.size(0) == 0:
                # If a cluster is empty, randomly choose a point as the medoid
                random_idx = torch.randint(0, X.size(0), (1,))
                medoids[i] = X[random_idx]
            else:
                cluster_distances = torch.cdist(cluster_points, cluster_points)
                total_distances = torch.sum(cluster_distances, dim=1)
                new_medoid_idx = torch.argmin(total_distances)
                medoids[i] = cluster_points[new_medoid_idx]

        # Reassign points to the closest medoid
        distances = torch.cdist(X, medoids)
        _, labels = torch.min(distances, dim=1)

  
    return  medoids.view(n_clusters,input.shape[1],input.shape[-1]),labels



class SASRecModel(nn.Module):
    def __init__(self, args,item_num):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(item_num, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.emb_dropout)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

    
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        sequence_output = self.item_encoder(sequence_emb, extended_attention_mask)
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# def swish(x):
    # return x * torch.sigmoid(x)
# 
# 
# ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}





class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = 4
        if args.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, self.num_attention_heads)
            )
        self.num_attention_heads = self.num_attention_heads
        self.attention_head_size = int(args.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.emb_dropout)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.emb_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
       
        self.intermediate_act_fn = gelu
        
        

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.emb_dropout)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
  
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            
        return hidden_states
