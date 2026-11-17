import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ChainedScheduler, LinearLR

def build_optimizer(model, optimizer_name='adamw', lr=1e-3, weight_decay=0.1):
    """
    Build and return an optimizer for the given model.

    Args:
        model (nn.Module): The model whose parameters will be optimized.
        optimizer_name (str): Optimizer type. One of:
            ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'].
        lr (float): Learning rate.
        weight_decay (float): L2 regularization term.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs, eta_min=1e-5)])