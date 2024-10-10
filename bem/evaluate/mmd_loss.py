import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    



def pdist(sample_1, sample_2, norm=2):
    return torch.cdist(sample_1, sample_2, p=norm)

class MMDStatistic:
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = 1. / (n_1 * n_1)
        self.a11 = 1. / (n_2 * n_2)
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        mmd_dict = {}
        for alpha in alphas:
            kernels = torch.exp(- alpha * distances**2)
            k_1 = kernels[:self.n_1, :self.n_1]
            k_2 = kernels[self.n_1:, self.n_1:]
            k_12 = kernels[:self.n_1, self.n_1:]
            mmd = self.a00 * k_1.sum() + self.a11 * k_2.sum() + 2 * self.a01 * k_12.sum()
            mmd_dict[alpha.item()] = f'{mmd.item()}' 
        if ret_matrix:
            return mmd_dict, kernels
        else:
            return mmd_dict

def get_MMD(X, Xhat, device, nmax = 1000, alpha_ls = [0.5]):
    #X, Xhat = torch.from_numpy(X).to(device), torch.from_numpy(Xhat).to(device)
    nmax1 = min(nmax, X.shape[0])
    nmax2 = min(nmax, Xhat.shape[0])
    X = X[torch.randperm(X.shape[0])[:nmax1]]
    Xhat = Xhat[torch.randperm(Xhat.shape[0])[:nmax2]]
    distances = pdist(X,Xhat)
    dist_median = torch.median(distances)
    gamma = 0.1*dist_median
    alpha_ls = [0.5/gamma**2]
    alpha_ls = torch.tensor(alpha_ls).to(device)
    mtd = MMDStatistic(nmax1, nmax2)
    mmd_dict = mtd(X, Xhat, alpha_ls)
    return mmd_dict




def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)