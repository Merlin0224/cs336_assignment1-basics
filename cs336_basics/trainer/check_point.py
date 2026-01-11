import torch

def save_checkpoint(model, optimizer, iteration, out_path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out_path)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']