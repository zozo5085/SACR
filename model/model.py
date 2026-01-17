from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import clip
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class MiniASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[1, 2, 4], hidden_ch=128):
        super(MiniASPP, self).__init__()
        self.stages = nn.ModuleList()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True)
        )

        for rate in rates:
            self.stages.append(nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True)
            ))

        self.bot_conv = nn.Sequential(
            nn.Conv2d(hidden_ch * len(rates), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        nn.init.constant_(self.bot_conv[1].weight, 0)
        nn.init.constant_(self.bot_conv[1].bias, 0)

    def forward(self, x):
        x_red = self.reduce(x)
        res = [stage(x_red) for stage in self.stages]
        out = torch.cat(res, dim=1)
        return self.bot_conv(out)

class StripPooling(nn.Module):
    def __init__(self, in_channels, pool_size=None, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.norm1 = norm_layer(in_channels)

        nn.init.constant_(self.norm1.weight, 0)
        nn.init.constant_(self.norm1.bias, 0)
        self.act = nn.GELU()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels)
        )
        nn.init.constant_(self.out_conv[1].weight, 0)
        nn.init.constant_(self.out_conv[1].bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape

        x1 = self.pool1(x)
        x1 = F.interpolate(x1, (h, w), mode='bilinear', align_corners=False)

        x2 = self.pool2(x)
        x2 = F.interpolate(x2, (h, w), mode='bilinear', align_corners=False)

        out = self.conv1(x1 + x2)
        out = self.norm1(out)
        out = self.act(out)

        out = self.out_conv(out)
        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=True, attn_mask=None)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        self._initialize_weights(clip_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + self.mlp(self.ln_2(v))
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        out = [x, q, k, v]
        return out

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = []
        for i in range(self.layers - 1):
            self.resblock.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblock.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self._initialize_weights(clip_model)
        self.resblocks = nn.Sequential(*self.resblock)

    def forward(self, x: torch.Tensor):
        z, q, k, v = self.resblocks(x)
        return z, q, k, v

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblock[i]._initialize_weights(clip_model, i)
        return


class VisionTransformer(nn.Module):
    def __init__(self, clip_model: clip, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dilation = [1, 1]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=self.patch_size, stride=self.patch_size,
                               bias=False)

        self.cls_token = torch.load('utils/cls_token.pt')

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(clip_model, width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)
        self._initialize_weights(clip_model)

    def forward(self, x, train=False, img_metas=None):
        B = x.shape[0]
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = (self.patch_size, self.patch_size)
        stride_h, stride_w = (self.patch_size, self.patch_size)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        x = x.to(device)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        positional_embedding = self.positional_embedding.unsqueeze(dim=0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size
        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, positional_embedding.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w), mode='bicubic',
                                         align_corners=False)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        positional_embedding = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

        x = x + positional_embedding
        x = self.ln_pre(x)
        x, q, k, v = self.transformer(x)
        x = self.ln_post(x)
        v = self.ln_post(v)

        out = x[:, 1:]
        B, _, C = out.shape
        out = out.reshape(B, output_h, output_w, C).permute(0, 3, 1, 2).contiguous()
        q = q[:, 1:]
        k = k[:, 1:]
        v = v[:, 1:]
        v = v.reshape(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()
        out = [out, q, k, v]
        cls_token = x[:, 0]

        if self.proj is not None:
            z_global = cls_token @ self.proj

        return [v, (output_h, output_w), z_global, k, positional_embedding[:, 1:, :]]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        for p in self.parameters():
            p.requires_grad = False
        return

class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.dtype = torch.float32
        self.device = device
        token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
        prompt_token = self.token_embedding(token)
        for p in self.parameters():
            p.requires_grad = False
        self.prompt_token = nn.Parameter(prompt_token)
        self.weight = False

    def forward(self, cls_name_token):
        device = self.device
        prompt_token = self.prompt_token.repeat(cls_name_token.shape[0], 1, 1).to(device)
        cls_name_token = cls_name_token.to(device)

        start_token = self.token_embedding(torch.tensor(49406, dtype=torch.int, device=device)).repeat(
            cls_name_token.shape[0], 1, 1).to(device)
        cls_token = self.token_embedding(cls_name_token).to(device)
        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection
        return x

class SACR(nn.Module):
    def __init__(self, cfg, clip_model, rank):
        super(SACR, self).__init__()
        self.vit = VisionTransformer(clip_model=clip_model,
                                     input_resolution=224,
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=768)
        self.clip = clip_model
        self.k = cfg.DATASET.K
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)
        self.logit_scale = clip_model.logit_scale
        for p in self.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        input_channels = text_channel + self.cnum

        self.coord_att = CoordAtt(inp=input_channels, oup=input_channels, reduction=32)
        self.aspp = MiniASPP(in_ch=input_channels, out_ch=input_channels, rates=[1, 2, 4], hidden_ch=128)
        self.spm = StripPooling(input_channels, pool_size=None)

        if cfg.MODEL.TRAINING:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(input_channels, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)

        else:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)
        v, shape, z_global, k, positional_embedding = self.vit(image, train=False, img_metas=img_metas)
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / (prompt.norm(dim=1, keepdim=True) + 1e-6)

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        feature = torch.cat((feat, output), dim=1)
        feature = self.coord_att(feature)
        aspp_out = self.aspp(feature)
        feature = feature + aspp_out
        spm_out = self.spm(feature)
        feature = feature + spm_out

        if img_metas is not None:
            edge_maps = []
            valid_edge = False
            if isinstance(img_metas, (list, tuple)):
                batch_metas = img_metas
            else:
                batch_metas = [img_metas]

            for meta in batch_metas:
                if meta is not None and 'edge_map' in meta:
                    edge_maps.append(meta['edge_map'])
                    valid_edge = True
                else:
                    edge_maps.append(torch.zeros(1, image.shape[2], image.shape[3]))

            if valid_edge:
                edge_tensor = torch.stack(edge_maps).to(device)
                edge_tensor = F.interpolate(edge_tensor, size=(feature.shape[2], feature.shape[3]), mode='bilinear',
                                            align_corners=False)
                feature = feature * (1.0 + edge_tensor)

        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature

        if return_feat:
            return output[0], feat[0], shape

        if training:
            output_scale = torch.mul(output.reshape(batch_size, cnum, -1).permute(0, 2, 1), 100)
            output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0],
                                                                                            shape[1], -1)

            loss = 0
            for j in range(batch_size):
                masked_image_features = []
                if len(gt_cls[j]) == 0:
                    continue
                for i in gt_cls[j]:
                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)
                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j]).to(device)

                sample_loss = F.cross_entropy(similarity_img, labels)

                loss += sample_loss

            return output, loss / batch_size

        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)