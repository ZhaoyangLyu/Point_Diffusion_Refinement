import torch
import torch.nn  as nn
import torch.nn.functional as F


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out

def count_to_mask(count, K):
    # counts is of shape (B, npoint)
    # its value range from 0 to K-1
    # return a mask of shape (B, npoint, K)
    mask = torch.arange(K, device=count.device, dtype=count.dtype)
    B, npoint = count.size()
    mask = mask.repeat(B, npoint).view(B, npoint,-1) # shape (B, npoint, K)
    mask = mask < count.unsqueeze(-1)
    return mask

class AttentionModule(nn.Module):
    def __init__(self, C_in1, C_in2, C1, C2, C_out, attention_bn=True, transform_grouped_feat_out=True, last_activation=True):
        super(AttentionModule, self).__init__()
        C1 = max(C1, 32)
        C2 = max(C2, 32)
        self.feat_conv = nn.Conv2d(C_in1, C1, kernel_size=1)
        self.grouped_feat_conv = nn.Conv2d(C_in2, C2, kernel_size=1)

        inter_C = min(C1+C2, C_out)
        if attention_bn:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(inplace=True),
                        MyGroupNorm(min(32, C1+C2), C1+C2),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1),
                        nn.ReLU(inplace=True),
                        MyGroupNorm(min(32, inter_C), inter_C),
                        nn.Conv2d(inter_C, C_out,kernel_size=1))
        else:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(inter_C, C_out,kernel_size=1))

        self.transform_grouped_feat_out = transform_grouped_feat_out
        if transform_grouped_feat_out:
            self.feat_out_conv = [nn.Conv2d(C_out, C_out, kernel_size=1)]
            if last_activation:
                if attention_bn:
                    self.feat_out_conv.append(MyGroupNorm(min(32, C_out), C_out))
                self.feat_out_conv.append(nn.ReLU(inplace=True))

            self.feat_out_conv = nn.Sequential(*self.feat_out_conv)


    def forward(self, feat, grouped_feat, grouped_feat_out, count):
        # feat (B,C_in1,N), acts like query
        # grouped_feat (B,C_in2,N,K), acts like key
        # grouped_feat_out (B,C_out,N,K) # acts like value
        # count is of shape (B,N)
        K = grouped_feat.shape[-1]
        feat1 = self.feat_conv(feat.unsqueeze(-1)) # (B,C1,N,1)
        feat1 = feat1.expand(-1,-1,-1,K) # (B,C1,N,K)

        grouped_feat1 = self.grouped_feat_conv(grouped_feat) # (B,C2,N,K)

        total_feat = torch.cat([feat1, grouped_feat1], dim=1) # (B,C1+C2,N,K)
        scores = self.weight_conv(total_feat) # (B,C_out,N,K)

        if not count == 'all':
            count = torch.clamp(count, min=1)
            mask = count_to_mask(count, K) # (B,N,K)
            mask = mask.unsqueeze(1).float() # (B,1, N,K)
            scores = scores * mask + (-1e9)*(1-mask)

        weight = F.softmax(scores, dim=-1) # (B,C_out,N,K)
        # pdb.set_trace()
        if self.transform_grouped_feat_out:
            grouped_feat_out = self.feat_out_conv(grouped_feat_out) # B,C_out,N,K
        out = grouped_feat_out * weight # B,C_out,N,K
        out = out.sum(dim=-1) # B,C_out,N
        return out

class GlobalAttentionModule(nn.Module):
    def __init__(self, C, additional_dim=0, attention_bn=True, last_activation=True):
        super(GlobalAttentionModule, self).__init__()
        self.key_conv = nn.Conv2d(C+additional_dim, C, kernel_size=1)
        self.query_conv = nn.Conv2d(C+additional_dim, C, kernel_size=1)

        self.value_conv = [nn.Conv2d(C+additional_dim, C, kernel_size=1)]
        if last_activation:
            if attention_bn:
                self.value_conv.append(MyGroupNorm(min(32, C), C))
            self.value_conv.append(nn.ReLU(inplace=True))
        self.value_conv = nn.Sequential(*self.value_conv)

        if attention_bn:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(inplace=True),
                        MyGroupNorm(min(32, 2*C), 2*C),
                        nn.Conv2d(2*C, C, kernel_size=1),
                        nn.ReLU(inplace=True),
                        MyGroupNorm(min(32, C), C),
                        nn.Conv2d(C, C,kernel_size=1))
        else:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Conv2d(2*C, C, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(C, C,kernel_size=1))

    def forward(self, feat):
        # feat is of shape (B C N) 
        # key = conv(feat), B,C,N
        # query = conv(feat), B,C,N
        # value = mlp(feat), B,C,N

        # key expand to B,C,1,N
        # query expand to B,C,N,1  
        # [key, query] is of shape B,2*C,N,N
        # score = mlp([key, query]) is of shape B,C,N,N
        # weight = softmax(score, dim=-1) is of shape B,C,N,N
        # value expand to shape B,C,N,N
        # new_feat = (value * weight).sum(dim=-1), shape (B,C,N)
        _,_,N = feat.size()
        key = self.key_conv(feat.unsqueeze(-1)).squeeze(-1) # (B,C,N)
        query = self.query_conv(feat.unsqueeze(-1)).squeeze(-1) # (B,C,N)
        value = self.value_conv(feat.unsqueeze(-1)).squeeze(-1) # (B,C,N)

        key = key.unsqueeze(-2)
        key = key.expand(-1,-1,N,-1) # (B,C,N,N)
        query = query.unsqueeze(-1)
        query = query.expand(-1,-1,-1,N) # (B,C,N,N)

        pair = torch.cat([query, key], dim=1) # (B,2*C,N,N)
        score = self.weight_conv(pair) # (B,C,N,N)
        weight = F.softmax(score, dim=-1) # (B,C,N,N)

        out_feat = (value.unsqueeze(-1) * weight).sum(dim=-1) # (B,C,N)
        return out_feat


if __name__ == '__main__':
    import pdb
    B = 5
    N = 1024
    C_in1 = 3
    C_in2 = 13
    C_out = 32
    C1 = C_in1
    C2 = C_in2
    K = 8
    feat =  torch.rand(B,C_in1,N)
    grouped_feat = torch.rand(B,C_in2,N,K)
    grouped_feat_out = torch.rand(B,C_out,N,K)
    count = torch.randint(K, size=(B,N))
    model = AttentionModule(C_in1, C_in2, C1, C2, C_out, attention_bn=False, 
                            transform_grouped_feat_out=False, last_activation=False)
    print(model)
    out = model(feat, grouped_feat, grouped_feat_out, count)
    pdb.set_trace()

    B = 4
    C = 256
    N = 64
    feat = torch.rand(B,C,N)
    xyz = torch.rand(B,3,N)
    feat = torch.cat([feat, xyz], dim=1)
    global_attention = GlobalAttentionModule(C, additional_dim=3, attention_bn=True, last_activation=True)
    feat_out = global_attention(feat)
    pdb.set_trace()