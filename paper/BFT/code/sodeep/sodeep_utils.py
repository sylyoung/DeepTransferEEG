import os
import torch


def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)
    rank = torch.argsort(rank, dim=dim)
    rank = (rank * -1) + batch_score.size(dim)
    rank = rank.float()
    rank = rank / batch_score.size(dim)

    return rank


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def save_checkpoint(state, is_best, model_name, epoch):
    # Write the exact filename the loaders open.  losspredictor.py and
    # BFT-regression/train_loss_model.py read
    # <sodeep>/weights/<name>_best_model.pth.tar, but this used to write
    # ./weights/best_<name>.pth.tar, which puts the two halves in the opposite
    # order and resolves against the working directory, so the file the trainer
    # produced was never the file the loaders asked for.
    if is_best:
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(state, os.path.join(weights_dir, model_name + "_best_model.pth.tar"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_epoch(logger, epoch, train_loss, val_loss, lr, batch_train, batch_val, data_train, data_val):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Loss/Val', val_loss, epoch)
    logger.add_scalar('Learning/Rate', lr, epoch)
    logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
    logger.add_scalar('Time/Train/Batch Processing', batch_train, epoch)
    logger.add_scalar('Time/Val/Batch Processing', batch_val, epoch)
    logger.add_scalar('Time/Train/Data loading', data_train, epoch)
    logger.add_scalar('Time/Val/Data loading', data_val, epoch)


def flatten(l):
    return [item for sublist in l for item in sublist]
