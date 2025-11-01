import torch

def save_checkpoint(model, optimizer, iteration, out):
    """Dump all state from model, optimizer, and iteration into `out`."""
    # Use state_dict() for both model and optimizer
    # Use torch.save(obj, out) to dump obj into out

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration_number": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """Load a checkpoint from `src` and recover model and optimizer states."""
    # Use torch.load(src) to recover saved state
    # Call load_state_dict on both model and optimizer
    # Return the saved iteration number

    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration_number"]
