import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .diffusion import Sampler
from .utils import Settings, num_to_groups, set_all_seeds

__all__ = ["train"]


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    sampler: Sampler,
    settings: Settings,
    epochs=6,
    visualize_type="grid",
    save_first_sample=True,
):
    device = settings.device
    visualize = visualize_type is not None
    if visualize and save_first_sample:
        save_images(model, sampler, settings, visualize_type, 0)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, sampler.timesteps, (batch_size,), device=device).long()

            loss = sampler.p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

        if visualize:
            save_images(model, sampler, settings, visualize_type, epoch + 1)


def save_images(model, sampler, settings, visialize_type, milestone):
    batch_size = 64

    set_all_seeds()
    if visialize_type == "diffusion":
        batches = num_to_groups(8, batch_size)
        all_images_list = list(
            map(
                lambda n: sampler.sample(
                    model,
                    image_size=settings["image_size"],
                    batch_size=n,
                    channels=settings["channels"],
                ),
                batches,
            )
        )
        all_images_list = [torch.tensor(x) for x in all_images_list[0]]
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(
            all_images,
            str(settings["result_folder"] / f"sample-{milestone}.png"),
            nrow=4,
        )
    elif visialize_type == "grid":
        batches = num_to_groups(64, batch_size)
        all_images_list = list(
            map(
                lambda n: sampler.sample(
                    model,
                    image_size=settings.image_size,
                    batch_size=n,
                    channels=settings.channels,
                ),
                batches,
            )
        )
        all_images = torch.tensor(all_images_list[0][-1])
        all_images = (all_images + 1) * 0.5
        save_image(
            all_images, str(settings.results_folder / f"sample-{milestone}.png"), nrow=8
        )
