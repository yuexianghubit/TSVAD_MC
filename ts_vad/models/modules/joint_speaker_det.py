from torch import nn, Tensor

class JointSpeakerDet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.time_layers = nn.ModuleList(
            [self.build_lstm(cfg) for _ in range(cfg.num_transformer_layer)]
        )
        self.spk_layers = nn.ModuleList(
            [self.build_transformer(cfg) for _ in range(cfg.num_transformer_layer)]
        )

    def build_transformer(self, cfg):
        layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.transformer_embed_dim,
                dim_feedforward=cfg.transformer_ffn_embed_dim,
                nhead=cfg.num_attention_head,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            ), 
            num_layers=1
        )
        return layer

    def build_lstm(self, cfg):
        layer = nn.LSTM(
            input_size=cfg.transformer_embed_dim,
            hidden_size=cfg.transformer_embed_dim // 2,
            num_layers=1,
            dropout=cfg.dropout,
            bidirectional=True,
            batch_first=True,
        )
        return layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        B, S, T, D = x.size()
        # x: B, 4, T, 384
        for time_layer, spk_layer in zip(self.time_layers, self.spk_layers):
            x = x.reshape(B * S, T, D)
            x, _ = time_layer(x)
            x = x.reshape(B * T, S, D)
            x = spk_layer(x)

        return x.reshape(B, S, T, D)
