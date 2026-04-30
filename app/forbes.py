import torch
import torch.nn as nn
from torch.nn import Dropout, MaxPool2d, Sequential, Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d, ReLU, Sigmoid, Module, PReLU
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from collections import namedtuple
import types

#  1. AdaFace Model Components

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1,1), stride=(1,1), padding=(0,0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))

class BasicBlockIR(Module):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        self.shortcut_layer = (MaxPool2d(1, stride) if in_channel == depth else
            Sequential(Conv2d(in_channel, depth, (1,1), stride, bias=False), BatchNorm2d(depth)))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1), 1, bias=False),
            BatchNorm2d(depth), PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1, bias=False),
            BatchNorm2d(depth))
    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    pass

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]

def get_blocks(num_layers):
    return [
        get_block(64,  64,  3),
        get_block(64,  128, 13),
        get_block(128, 256, 30),
        get_block(256, 512, 3)
    ]

class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        self.input_layer = Sequential(Conv2d(3, 64, (3,3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        unit_module = BasicBlockIR 
        output_channel = 512
        s = 7 * 7 if input_size[0] == 112 else 14 * 14
        self.output_layer = Sequential(
            BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
            Linear(output_channel * s, 512), BatchNorm1d(512, affine=False))
        modules = [unit_module(b.in_channel, b.depth, b.stride)
                   for block in get_blocks(num_layers) for b in block]
        self.body = Sequential(*modules)
        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for m in self.body:
            x = m(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.div(x, norm), norm

class AdaFace(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.adaface = Backbone([112, 112], 100, 'ir')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = {k[6:]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        self.adaface.load_state_dict(sd)

    def forward(self, x):
        x = self.adaface.input_layer(x)
        for m in self.adaface.body:
            x = m(x)
        x = self.adaface.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.div(x, norm), norm

# ── 2. Pattern Utilities ───────────────────────────────────────────────────

def Mosaicking(x, alpha=None):
    b, c, h, w = x.size()
    y = torch.stack(torch.stack(x.chunk(7, dim=2)).chunk(7, dim=4)).mean(dim=(-1,-2)).permute(2,3,1,0)
    return F.interpolate(y, (h, w), mode='nearest')

def Horiz_mean(x, param=None):
    b, c, h, w = x.size()
    return torch.mean(x, dim=3, keepdim=True).expand(b, c, h, w)

def Vertic_mean(x, param=None):
    b, c, h, w = x.size()
    return torch.mean(x, dim=2, keepdim=True).expand(b, c, h, w)

def Warping(x, flo):
    B, C, H, W = x.size()
    _, _, h, w = flo.size()
    xx = torch.arange(0, W).view(1,1,1,W).expand(B,1,H,W)
    yy = torch.arange(0, H).view(1,1,H,1).expand(B,1,H,W)
    grid = torch.cat((xx, yy), 1).float()
    flo = F.interpolate(flo, (H, W), mode='bilinear', align_corners=True)
    flo[:, 0] = flo[:, 0] * H / h
    flo[:, 1] = flo[:, 1] * W / w
    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    return nn.functional.grid_sample(x, vgrid.permute(0,2,3,1), align_corners=True)

def Sinusoid(x, param=None):
    b, c, h, w = x.size()
    wp, hp, _, _, _, _ = param.size()
    wg, hg = w // wp, h // hp
    yy = torch.linspace(-np.pi, np.pi, hg).view(1,1,1,1,1,hg,1).expand(wp,hp,1,3,1,hg,wg)
    xx = torch.linspace(-np.pi, np.pi, wg).view(1,1,1,1,1,1,wg).expand(wp,hp,1,3,1,hg,wg)
    grid = torch.cat((xx, yy), 4).float().to(x.device)
    alpha = param
    grid = grid[:,:,:,:,0] * alpha + grid[:,:,:,:,1] * (1 - alpha)
    return (0.2 * torch.sin(grid * 4)).permute(2,3,1,4,0,5).reshape(b,c,h,w)

def Checkerboard(x, param=None):
    b, c, h, w = x.size()
    param = torch.ones(7, 7, 1, 3, 1, 1).to(x.device)
    xx = torch.arange(0, 4).view(1,1,1,4).expand(1,1,4,4)
    yy = torch.arange(0, 4).view(1,1,4,1).expand(1,1,4,4)
    grid = ((xx + yy) % 2).float() - 0.5
    block_size = h // 7
    grid = F.interpolate(grid, (block_size, block_size), mode='nearest').to(x.device)
    return (param * grid).permute(2,3,1,4,0,5).reshape(1,c,h,w).expand(b,c,h,w) * 0.3

def Speckle(x, param=None):
    b, c, h, w = x.size()
    return F.interpolate(param, (h, w), mode='bilinear')

def Scaling(x, input_color):
    b, c, h, w = x.size()
    bh = input_color.size()[1]
    bw = input_color.size()[0]
    x = torch.stack(torch.stack(x.chunk(bh, dim=2)).chunk(bw, dim=4))
    x = torch.clamp(((x + 1) * input_color / 2) * 2 - 1, min=-1, max=1)
    return x.permute(2,3,1,4,0,5).reshape(b,c,h,w)

def generate_param(transform, size, device):
    n = int(np.prod(size))
    if transform == 'Warping':
        p = torch.rand(n, device=device).float() * (0.3 - -0.3) + -0.3
        pv = p.view(size)
        pv[:, :, [0, -1], :] = 0
        pv[:, :, :, [0, -1]] = 0
        p = pv.view(-1)
        bound = [(-0.3, 0.3)] * n
        mask = (p == 0).cpu().tolist()
        return p, [(0, 0) if m else b for m, b in zip(mask, bound)]
    elif transform == 'Scaling':
        value = 1.1
        prob = torch.rand(n, device=device).float()
        p = torch.zeros(n, device=device).float()
        _min = 1.0 / value
        mask = prob > 0.5
        p[mask] = torch.rand(int(mask.sum()), device=device) * (value - 1) + 1
        p[~mask] = torch.rand(int((~mask).sum()), device=device) * (1 - _min) + _min
        return p, [(_min, value)] * n
    elif transform == 'Sinusoid':
        return torch.rand(n, device=device).float(), [(0, 1)] * n
    elif transform == 'Speckle':
        return torch.rand(n, device=device).float() * (0.5 - -0.5) + -0.5, [(-0.5, 0.5)] * n
    elif 'blend' in transform:
        return torch.rand(n, device=device).float(), [(None, None)] * n

# ── 3. BRS Optimizer ───────────────────────────────────────────────────────

class BRS:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.transform_size = {}
        self.orig_emb = None
        self.in_img   = None

    def _apply_one(self, img, key, param):
        """Apply a single named transform to img."""
        if key == 'Warping':      return Warping(img, param)
        if key == 'Scaling':      return Scaling(img, param)
        if key == 'Sinusoid':     return img + Sinusoid(img, param)
        if key == 'Speckle':      return img + Speckle(img, param)
        if key == 'Mosaicking':   return Mosaicking(img)
        if key == 'Horiz_mean':   return Horiz_mean(img)
        if key == 'Vertic_mean':  return Vertic_mean(img)
        if key == 'Checkerboard': return img + Checkerboard(img)
        return img  # blend transforms: no-op (no standalone function defined)

    def optimize(self, in_img):
        self.in_img = in_img
        b, c, h, w  = in_img.size()

        # Only transforms that have optimisable parameters
        self.transform_size = {
            'Warping':  [b, 2, 7, 7],
            'Scaling':  [7, 7, b, 3, 1, 1],
            'Sinusoid': [7, 7, b, 3, 1, 1],
            'Speckle':  [b, 3, 7, 7],
        }

        # Cache original embedding — adversarial objective moves away from this
        self.model.eval()
        with torch.no_grad():
            self.orig_emb, _ = self.model(in_img)

        # Build flat parameter vector and bounds for L-BFGS
        init_params = None
        bounds = []
        for key in self.args.transform_type:
            if key in self.transform_size:
                kp, bound = generate_param(key, self.transform_size[key], in_img.device)
                bounds.extend(bound)
                init_params = kp if init_params is None else torch.cat((init_params, kp))

        if init_params is None:
            return Mosaicking(in_img)

        x0 = init_params.detach().cpu().numpy().ravel().astype(np.float64)
        result = fmin_l_bfgs_b(
            func=self._optimize_function, x0=x0,
            m=20, factr=0, pgtol=1e-8, maxfun=20, maxiter=40, bounds=bounds
        )

        # Apply the OPTIMISED parameters to produce the final obfuscated image
        op = torch.from_numpy(result[0]).float().to(self.device)
        img = in_img.clone()
        offset = 0

        with torch.no_grad():
            for key in self.args.transform_type:
                if key in self.transform_size:
                    size = self.transform_size[key]
                    n    = int(np.prod(size))
                    param = op[offset:offset + n].view(*size)
                    offset += n
                    img = self._apply_one(img, key, param)
                else:
                    img = self._apply_one(img, key, None)
                img = torch.clamp(img, -1, 1)

        return img

    def _optimize_function(self, in_params):
        """
        Called by scipy L-BFGS-B at each iteration.
        Returns (loss, gradient) where loss = cosine_similarity(orig_emb, obfuscated_emb).
        Minimising this pushes the obfuscated face embedding away from the original.
        """
        with torch.enable_grad():   # override any outer torch.no_grad() context
            x = torch.from_numpy(in_params.copy()).float().to(self.device)
            x.requires_grad_(True)

            img = self.in_img.detach().clone()
            offset = 0

            for key in self.args.transform_type:
                if key in self.transform_size:
                    size  = self.transform_size[key]
                    n     = int(np.prod(size))
                    param = x[offset:offset + n].view(*size)
                    offset += n
                    img = self._apply_one(img, key, param)
                else:
                    img = self._apply_one(img, key, None)
                img = torch.clamp(img, -1, 1)

            emb, _ = self.model(img)

            # Minimise similarity → maximise embedding distance from original
            loss = F.cosine_similarity(self.orig_emb.detach(), emb).mean()
            loss.backward()

            grad = (x.grad.detach().cpu().numpy().ravel().astype(np.float64)
                    if x.grad is not None else np.zeros_like(in_params))

            return float(loss.item()), grad

# ── 4. Initialization Configuration ─────────────────────────────────────────

def init_forbes(ckpt_path, device):
    args = types.SimpleNamespace()
    args.transform_type = [
        'Mosaicking', 'Horiz_mean', 'Vertic_mean', 'Averaging_blend',
        'Warping', 'Sinusoid', 'Checkerboard', 'Speckle', 'Noising_blend', 'Scaling'
    ]
    args.transform_margin = {
        'Warping': 0.05,
        'Scaling': 1.05,
        'Speckle': 0.1
    }
    
    try:
        model = AdaFace(ckpt_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Forbes warning: could not load model from {ckpt_path}. Check file. Error: {e}")
        return None
        
    optimizer_brs = BRS(model, device, args)
    return optimizer_brs