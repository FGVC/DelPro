import torch
import dnnutil
from dnnutil import tocuda
import time


def get_distance_matrix(x):
    distance_matrix = torch.norm(x[:, None] - x, dim=2, p=2)
    return distance_matrix


class MetricTrainer(dnnutil.Trainer):

    def __init__(self, net, optim, loss_fn):
        super().__init__(net, optim, loss_fn)
        self.measure_accuracy = CMCAccuracy()

    def eval(self, dataloader, epoch):
        self.net.eval()
        stats = self.run_eval(dataloader, epoch)
        self._set_test_stats(stats)
        return stats

    def train_batch(self, batch):
        self.optim.zero_grad()

        imgs, labels = tocuda(batch)

        embeddings = self.net(imgs)
        loss = self.loss_fn(embeddings, labels)
        loss.backward()
        self.optim.step()
        loss = loss.item()

        return loss, 0

    @torch.no_grad()
    def run_eval(self, dataloader, epoch):
        N = len(dataloader.batch_sampler)
        
        embeddings = []
        labels = []
        flanks = []
        for i, batch in enumerate(dataloader):
            t = time.time()
            imgs, labs, flnk = tocuda(batch)
            emb = self.net(imgs)
            embeddings.append(emb)
            labels.append(labs)
            flanks.append(flnk)
            t = time.time() - t
            
            if i == 0:
                at = t
            else:
                at = at * i / (i + 1) + t / (i + 1)
            print(f'\rEPOCH {epoch}: test '
                  f'batch {i + 1:04d}/{N} '
                  f'lr[ {self.optim.param_groups[0]["lr"]:1.3e} ] '
                  f'[ {t:.3f} ({at:.3f}) secs ]'
                  f'{" "*10}',
                  end='', flush=True)

        embeddings = torch.cat(embeddings, 0)
        labels = torch.cat(labels, 0)
        flanks = torch.cat(flanks, 0)
        accuracy = self.measure_accuracy(embeddings, labels, flanks)
        return 0, accuracy


class SiameseMetricTrainer(MetricTrainer):
    def train_batch(self, batch):
        self.optim.zero_grad()

        imgs1, labels1, imgs2, labels2 = tocuda(batch)
        imgs = torch.cat([imgs1, imgs2], 0)
        labels = torch.cat([labels1, labels2], 0)

        embeddings = self.net(imgs)
        loss = self.loss_fn(embeddings, labels)
        loss.backward()
        self.optim.step()
        loss = loss.item()

        return loss, 0


class PairwiseContrastLoss(torch.nn.Module):
    def __init__(self, p=2, margin=1.0):
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=p)
        self.contrast = torch.nn.HingeEmbeddingLoss(margin=margin)

    def __call__(self, embeddings, labels):
        n = embeddings.shape[0]
        pos = torch.Tensor([1]).to(embeddings)
        emb1, emb2 = embeddings.split(n // 2)
        lab = torch.where(labels[:n//2] == labels[n//2:], pos, -pos)
        x = self.distance(emb1, emb2)
        loss = self.contrast(x, lab)
        return loss


class AllPairContrastLoss(torch.nn.Module):
    def __init__(self, mp=0, mn=1.0):
        super().__init__()
        self.mp = mp
        self.mn = mn

    def __call__(self, embeddings, labels):
        distance_matrix = get_distance_matrix(embeddings)
        b, n = distance_matrix.shape
        triu_ind = torch.triu_indices(b, n, 1, device=distance_matrix.device)
        dists = torch.sqrt(distance_matrix[triu_ind[0], triu_ind[1]] + 1e-7)
        lab_pairs = labels.view(-1, 1).eq(labels.view(1, -1))
        lab_pairs = lab_pairs[triu_ind[0], triu_ind[1]]

        loss = torch.where(lab_pairs,
                           torch.clamp(dists - self.mp, min=0),
                           torch.clamp(self.mn - dists, min=0))

        return loss.mean()


class RankedListLoss(torch.nn.Module):
    def __init__(self, alpha=1.2, m=0.4, T=10):
        super().__init__()
        self.alpha = alpha
        self.m = m
        self.T = T
    
    def __call__(self, embeddings, labels):
        n = len(labels)
        m = self.m
        alpha = self.alpha
        T = self.T
        # get nontrivial positives and negatives
        d_mat = get_distance_matrix(embeddings)
        matches = labels.view(-1, 1) == labels
        pos = torch.clamp(d_mat[matches].view(n, -1) - (alpha - m), min=0)
        neg = torch.clamp(alpha - d_mat[1 - matches].view(n, -1), min=0)

        # negative sample weights
        w = torch.exp(T * neg)
        w = w * neg.gt(0).float()
        w = w / w.sum(dim=1, keepdim=True)

        nt_pos = pos.gt(0).sum(1).float()
        pos = pos.sum(dim=1)
        ind = pos.gt(0)
        Lp = pos[ind].div(nt_pos[ind]).sum()
        Ln = neg.mul(w).sum()
        loss = 1 / n * (Lp + Ln)
        if torch.isnan(loss).any():
            breakpoint()

        return loss


class CMCAccuracy(torch.nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def __call__(self, embeddings, labels, flanks=None):
        k = self.k
        cmc = []
        for flank in flanks.unique():
            lab = labels[flanks == flank]
            emb = embeddings[flanks == flank]
            squared = emb.pow(2).sum(1)
            distance_matrix = (squared.view(1, -1) + squared.view(-1, 1) -
                               2 * torch.mm(emb, emb.t()))
            rank = distance_matrix.argsort(1)
            matches = lab[rank].eq(lab.view(-1, 1))[:, 1:k+1]
            cmc.append(matches.any(1).float().mean().item())
        return sum(cmc) / len(cmc)

