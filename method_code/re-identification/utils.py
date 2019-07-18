import torch
import dnnutil
from dnnutil import tocuda
import time


class SiameseMetricTrainer(dnnutil.Trainer):

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

        imgs1, labels1, imgs2, labels2 = tocuda(batch)
        imgs = torch.cat([imgs1, imgs2], 0)
        labels = torch.cat([labels1, labels2], 0)

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
        for i, batch in enumerate(dataloader):
            t = time.time()
            imgs, labs = tocuda(batch)
            emb = self.net(imgs)
            embeddings.append(emb)
            labels.append(labs)
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
        accuracy = self.measure_accuracy(embeddings, labels)
        return 0, accuracy


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
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def __call__(self, embeddings, labels):
        squared = embeddings.pow(2).sum(1)
        distance_matrix = (squared.view(1, -1) + squared.view(-1, 1) -
                           2 * torch.mm(embeddings, embeddings.t()))
        b, n = distance_matrix.shape
        triu_ind = torch.triu_indices(b, n, 1, device=distance_matrix.device)
        dists = torch.sqrt(distance_matrix[triu_ind[0], triu_ind[1]] + 1e-7)
        lab_pairs = labels.view(-1, 1).eq(labels.view(1, -1))
        lab_pairs = lab_pairs[triu_ind[0], triu_ind[1]]
        loss = torch.where(lab_pairs, dists, torch.clamp(self.margin - dists, min=0))

        return loss.mean()


class CMCAccuracy(torch.nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def __call__(self, embeddings, labels):
        k = self.k
        squared = embeddings.pow(2).sum(1)
        distance_matrix = (squared.view(1, -1) + squared.view(-1, 1) -
                           2 * torch.mm(embeddings, embeddings.t()))
        rank = distance_matrix.argsort(1)
        cmc = labels[rank].eq(labels.view(-1, 1))[:, 1:k+1]
        cmc = cmc.any(1).float().mean().item()
        return cmc

