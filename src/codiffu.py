import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from Modules import *




class CoDiffu(nn.Module):
    def __init__(self, args):
        super(CoDiffu, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)

        self.rescale_timesteps = args.rescale_timesteps
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4), SiLU(), nn.Linear(self.hidden_size * 4, self.hidden_size))
        self.att = Transformer_rep(args)
        self.lambda_history = args.lambda_history
        self.lambda_intent = args.lambda_intent
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.item_num = args.item_num+1
        self.item_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        # self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.loss_ce = nn.CrossEntropyLoss()
        # self.encoder=SASRecModel(args,self.item_num)
        self.n_clusters = args.num_cluster
        self.n_iters = args.num_iter
        self.mlp= nn.Sequential(nn.Linear(self.hidden_size*args.max_len, self.n_clusters))
        # self.centroids = torch.zeros(self.n_clusters,args.max_len,args.hidden_size).cuda()

        

        
      
       
          
      
        # self.mlp= nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size))
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t


    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output= self.denoise(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        
        x_0 = model_output  ##output predict
        # model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t 
    def denoise(self,item_rep, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        res= self.att(item_rep*self.lambda_history + 0.001 * x_t.unsqueeze(1)+self.centroids[self.labels]*self.lambda_intent, mask_seq)
        res= self.norm_diffu_rep(self.dropout(res))
       
        # out=self.mlp(x_t)
        out=res[:, -1, :]
        return out
    


    def diffu(self, item_rep, item_tag, mask_seq):        
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) ## t is sampled from schedule_sampler
        x_t = self.q_sample(item_tag, t, noise=noise)      
        x_0 = self.denoise(item_rep, x_t, self._scale_timesteps(t), mask_seq) ##output predict
        return x_0
    
    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        contra=self.contra_loss(rep_diffu)
        return self.loss_ce(scores, labels.squeeze(-1)),contra
    
    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    def intent_cluster(self,input,n_clusters):
        X=input.view(input.shape[0],-1)
        centers = X[torch.randperm(X.size(0))[:n_clusters]].to(input.device)
        labels=self.mlp(X)
        labels = F.gumbel_softmax(labels, tau=0.1, hard=True)
        # labels = torch.argmax(labels, dim=1)

        for i in range(self.n_clusters):
            if torch.sum(labels[:, i]) == 0:
                centers[i] = X[torch.randint(0, X.size(0), (1,))]
            else:
                centers[i] = torch.mean(X[labels[:, i].bool()], dim=0)
        return centers.view(n_clusters,input.shape[1],input.shape[-1]), torch.argmax(labels, dim=1)

    def contra_loss(self,diffu):
        temperature=0.07
        similarities = (F.cosine_similarity(diffu.unsqueeze(1), diffu.unsqueeze(0), dim=2) / temperature).to(diffu.device)
        
        # Positive pairs
        positives = torch.diag(similarities).to(diffu.device)
        
        # Negative pairs
        mask = torch.eye(len(diffu)).bool().to(diffu.device)
        mask = ~mask.to(diffu.device)
        negatives = similarities[mask].view(len(diffu), -1).to(diffu.device)
        
        # Calculate NCE Loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(len(diffu), dtype=torch.long).to(diffu.device)
        loss = F.cross_entropy(logits, labels)        
        return loss


    def forward(self, sequence, tag, train_flag): 

        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        item_embeddings = self.LayerNorm(item_embeddings)
        # input=self.encoder(sequence)[:,-1,:]
        # self.centroids, self.labels = KMeans(item_embeddings, self.n_clusters, self.n_iters)
        self.centroids, self.labels = self.intent_cluster(item_embeddings, self.n_clusters)

        mask_seq = (sequence>0).float()
        
        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H
            rep_diffu= self.diffu(item_embeddings, tag_emb, mask_seq)
          
        else:
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            rep_diffu = self.reverse_p_sample(item_embeddings, noise_x_t, mask_seq)

        return rep_diffu,self.centroids




