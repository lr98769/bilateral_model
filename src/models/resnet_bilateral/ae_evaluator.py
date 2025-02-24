from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.models import ResNet18_Weights
from src.preprocessing.pytorch_preprocessing import AEImageDataset

from src.configs.image_config import device
from src.models.resnet_bilateral.resnet_bilateral_model import ResNet18Bilateral
from src.image_eda import transform_for_visualisation


def evaluate_w_ae(model: ResNet18Bilateral, dl: DataLoader, verbose: bool=True):
    model.eval()
    model.to(device)
    loss_fn = MSELoss() 
    num_batches = len(dl)
    all_loss, total = 0, 0
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for input_batch in pbar:
                input_batch = input_batch.to(device)
                output_logits = model.reconstruction(input_batch)
                loss = loss_fn(output_logits, input_batch)
                num_samples = len(input_batch)
                all_loss += loss.item() * num_samples
                total += num_samples
    return all_loss/total

def get_ae_losses(model: ResNet18Bilateral, dl: DataLoader, verbose: bool=True):
    model.eval()
    model.to(device)
    num_batches = len(dl)
    losses = []
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for input_batch in pbar:
                input_batch = input_batch.to(device)
                output_logits = model.reconstruction(input_batch)
                loss_vec = torch.square(
                    output_logits.cpu()-input_batch.cpu()).mean(axis=[-3, -2, -1]).numpy()
                losses.append(loss_vec)
    return np.concatenate(losses)

def show_worst_reconstructions(model, df, col_info, losses, num_examples,pic_size, dpi, fp_fig):
    print("- Get Indices")
    worst_indices = np.argpartition(losses, -num_examples)[-num_examples:]
    ds = AEImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    ds = Subset(ds, worst_indices)
    show_reconstructions(model, ds, pic_size, dpi, fp_fig)

def show_best_reconstructions(model, df, col_info, losses, num_examples, pic_size, dpi, fp_fig):
    print("- Get Indices")
    best_indices = np.argpartition(losses, num_examples)[:num_examples]
    ds = AEImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    ds = Subset(ds, best_indices)
    show_reconstructions(model, ds, pic_size, dpi, fp_fig)

def show_reconstructions(model : ResNet18Bilateral, ds, pic_size, dpi, fp_fig):
    nrows = 2
    num_examples = len(ds)
    # Get Images and their Reconstructions
    print("- Images and reconstructions")
    images = []
    for image in tqdm(ds):
        images.append(image)
    reconstructions = model.reconstruction(torch.stack(images).to(device)).detach().cpu()
    print("- Plotting")
    # Show Images and their reconstructions
    fig, axes = plt.subplots(
        nrows, num_examples, figsize=(num_examples*pic_size, nrows*pic_size), dpi=dpi)
    for i, (image, reconstruction) in enumerate(zip(images, reconstructions)):
        axes[0, i].imshow(transform_for_visualisation(image))
        axes[1, i].imshow(transform_for_visualisation(reconstruction)) 
        axes[0, i].axes.get_xaxis().set_ticks([])
        axes[0, i].axes.get_yaxis().set_ticks([])
        axes[1, i].axes.get_xaxis().set_ticks([])
        axes[1, i].axes.get_yaxis().set_ticks([])
    axes[0, 0].set_ylabel("Image")
    axes[1, 0].set_ylabel("Reconstruction")
    plt.tight_layout()
    plt.savefig(fp_fig)
    plt.show()


    
