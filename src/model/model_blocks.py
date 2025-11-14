import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, activation: nn.Module, k: int = 1, s: int = 1, p: int = 0, g: int = 1):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        activation : callable or torch.nn.Module
            Activation function or module applied after batch normalization (e.g. torch.nn.SiLU() or torch.nn.Identity()).
        k : int, optional
            Convolution kernel size. Default is 1.
        s : int, optional
            Stride for the convolution. Default is 1.
        p : int, optional
            Padding for the convolution. Default is 0.
        g : int, optional
            Number of groups for grouped convolution. Default is 1.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        # Passing the input by convolution layer and using the activation function
        # on the normalized output
        return self.relu(self.norm(self.conv(x)))
    
    def fuse_forward(self, x):
        return self.relu(self.conv(x))

class Residual(nn.Module):
    def __init__(self, ch: int, e: float = 0.5):
        """
        Builds a two-layer bottleneck-style block that first reduces
        the channel dimensionality by a factor of `e` and then restores it to the
        original number of channels. Both internal convolutions use SiLU activations
        and 3x3 kernels with padding of 1.
        
        Parameters
        ----------
        ch : int
            Number of input (and output) channels for the block.
        e : float, optional
            Expansion (bottleneck) ratio used to compute the intermediate channel
            count as int(ch * e). Defaults to 0.5.
        """
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, nn.SiLU(), k=3, p=1)

    def forward(self, x):
        # The input is passed through 2 Conv blocks and if the shortcut is true and
        # if input and output channels are same, then it will the input as residual
        return x + self.conv2(self.conv1(x))

class C3K(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """
        Initialize the C3K-like block.
        This constructor builds a composite block composed of three convolutional
        submodules and a residual module container.
        
        Parameters
        ----------
        in_ch : int
            Number of input channels to the block.
        out_ch : int
            Number of output channels produced by the block. This value
            is expected to be even because intermediate convolutions divide it by 2.
        """
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, nn.SiLU())
        self.res_m = nn.Sequential(
            Residual(out_ch // 2, e=1.0),
            Residual(out_ch // 2, e=1.0)
        )

    def forward(self, x):
        # Process half of the input channels
        y = self.res_m(self.conv1(x))
        # Process the other half directly, Concatenate along the channel dimension
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))

class C3K2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, csp: bool, r: int):
        """
        Initialize the model block.
        
        Parameters:
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        n : int
            Number of residual blocks or bottlenecks to use.
        csp : bool
            Flag to indicate whether to use the CSP module (True) or bottlenecks (False).
        r : int
            Reduction factor for output channels.
        """
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, nn.SiLU())

        if not csp:
            # Using the CSP Module when mentioned True at shortcut
            self.res_m = nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            # Using the Bottlenecks when mentioned False at shortcut
            self.res_m = nn.ModuleList(C3K(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))

class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF (Spatial Pyramid Pooling Fast) spatial processing block.
        This block implements a lightweight spatial pyramid pooling variant (SPPF) that
        pools features at multiple spatial scales to improve the network's ability to
        capture objects of different sizes, particularly small objects.
        
        Parameters
        ----------
        c1 : int
            Number of input channels.
        c2 : int
            Number of output channels.
        k : int
            Kernel size for the internal MaxPool2d layer used for multi-scale pooling.
            Default is 5. Padding is set to k // 2 and stride to 1 so spatial
            dimensions are preserved when k is odd.
        """
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, nn.SiLU(), 1, 1)
        self.cv2    = Conv(c_ * 4, c2, nn.SiLU(), 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Attention(nn.Module):
    def __init__(self, ch: int, num_head: int):
        """
        Attention module class for multi-head convolutional attention used in image segmentation models.
        This initializer sets up a lightweight multi-head attention block that combines convolutional
        projections with per-head key/value subspaces. The design produces a query tensor that
        retains the full channel dimensionality and a concatenated set of keys and values split
        across heads, while also providing a pair of convolutional layers (depthwise followed by
        pointwise) for local feature processing.
        
        Parameters
        ----------
        ch : int
            Number of input channels.
        num_head : int
            Number of attention heads.
        """
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, nn.Identity())

        self.conv1 = Conv(ch, ch, nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)

class PSABlock(nn.Module):
    def __init__(self, ch: int, num_head: int):
        """
        Position-sensitive attention block that combines a multi-head Attention module with a small
        convolutional bottleneck to enrich feature maps with global context while preserving the
        original channel dimensionality.
        
        Parameters
        ----------
        ch : int
            Number of input (and output) channels expected by the block. The Attention module and
            the convolutional layers operate on tensors with this channel dimension.
        num_head : int
            Number of attention heads passed to the Attention module.
        """
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = nn.Sequential(
            Conv(ch, ch * 2, nn.SiLU()),
            Conv(ch * 2, ch, nn.Identity())
        )

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)

class PSA(torch.nn.Module):
    def __init__(self, ch: int, n: int):
        """
        High-level wrapper module that composes two Conv layers with SiLU activations
        and a sequence of PSA sub-blocks as a residual module.

        Parameters
        ----------
        ch : int
            Number of input/output channels for the overall block. This value determines
            the intermediate channel sizing used by conv1/conv2 and the channel sizes
            passed to each PSABlock in the residual module.
        n : int
            Number of PSABlock instances to stack inside the residual module `res_m`.
            If n == 0, `res_m` will be an empty sequential container.
        """
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, nn.SiLU())
        self.res_m = nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        # Passing the input to the Conv Block and splitting into two feature maps
        x, y = self.conv1(x).chunk(2, 1)
        # 'n' number of PSABlocks are made sequential, and then passes one them (y) of
        # the feature maps and concatenate with the remaining feature map (x)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))

class DFL(nn.Module):
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1: int = 16):
        """
        Dense-Foreground-Label (DFL) block.
        This class implements a simple, non-trainable 1x1 convolutional projection that
        maps an input tensor with c1 channels to a single-channel output. The convolution
        weights are fixed at initialization to the sequence [0, 1, 2, ..., c1-1] and
        the bias is disabled. The layer is intended for deterministic linear
        combinations of input channels (for example, creating a channel index encoding
        or deterministic scoring map).
        
        Parameters
        ----------
        c1 : int, optional
            Number of input channels expected by the block (default: 16). The block
            will create a Conv2d with in_channels=c1 and out_channels=1.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
