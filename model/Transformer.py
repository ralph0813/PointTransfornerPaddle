import paddle
import paddle.nn as nn


class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Layer):
    def __init__(self, dim, dim_hidden, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/sqrt(dim_head)
        self.activate_fn = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias_attr=False)  # *3 -> q, k, v

        # assume the output dim is the same as input dim
        # Even if the dimensions can be matched,
        # the information output by multiple headers needs to be integrated
        # so, only when the heads == 1 and dim matches this step can be omitted
        self.to_out = nn.Identity() if (heads == 1 and dim == dim_head) else nn.Sequential(
            nn.Linear(heads * dim_head, dim),
            nn.Dropout(dropout)
        )  # nn.Identity is a placeholder

    def forward(self, x):
        B, N, C = x.shape  # x -> n, n_patches , dim
        qkv = self.to_qkv(x)  # n, n_patches, heads * dim_head * 3
        qkv = qkv.reshape([B, N, 3, self.heads, -1])  # n, n_patches, 3, heads, dim_head
        qkv = qkv.transpose([2, 0, 3, 1, 4])  # 3, n, heads, n_patches, dim_head
        q, k, v = qkv[0], qkv[1], qkv[2]
        qk = paddle.matmul(q, k.transpose([0, 1, 3, 2]))  # n, heads, n_patches, n_patches
        qk = qk * self.scale  # n, heads, n_patches, n_patches
        attn = self.activate_fn(qk)  # n, heads, n_patches, n_patches
        attn = self.dropout(attn)  # n, heads, n_patches, n_patches
        out = paddle.matmul(attn, v)  # n, heads, n_patches, dim_head
        out = out.transpose([0, 2, 1, 3])  # n, n_patches, heads, dim_head
        out = out.reshape([B, N, -1])  # n, n_patches, heads * dim_head
        out = self.to_out(out)  # n, n_patches, dim
        return out


class Transformer(nn.Layer):
    def __init__(self, depth, dim, heads, dim_head, dim_mlp, dropout=0.):
        super().__init__()
        self.layers = nn.LayerList()
        for _ in range(0, depth):
            self.layers.append(
                nn.LayerList([
                    PreNorm(dim, AttentionBlock(dim, heads, dim_head, dropout)),
                    PreNorm(dim, FeedForward(dim, dim_mlp, dropout))
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerEncoder(nn.Layer):
    def __init__(self, dim=384, heads=8, dim_head=384, dim_mlp=2048, dropout=0.):
        super().__init__()
        self.layers = nn.LayerList()
        for _ in range(0, 12):
            self.layers.append(
                nn.LayerList([
                    PreNorm(dim, AttentionBlock(dim, heads, dim_head, dropout)),
                    PreNorm(dim, FeedForward(dim, dim_mlp, dropout))
                ])
            )

    def forward(self, x):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(x)
            x = ff(x)
            if idx in fetch_idx:
                feature_list.append(x)
        return feature_list


if __name__ == '__main__':
    x = paddle.randn([4, 16, 384])
    transformer = Transformer(4, 384, 8, 64, 2048)
    out = transformer(x)
