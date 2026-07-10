# ==============================================================================
# Training loop: each dataset item is a track split into overlapping sections;
# gradients are accumulated over the sections of a track, then one optimizer
# step is taken per track. Best checkpoints saved on epoch-loss improvement.
# ==============================================================================

import os

import torch
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, num_epochs=10,
          checkpoint_dir="models", device=None):
    """Trains the model; returns (per-epoch losses, (first, last) batch losses).

    first/last batch losses per epoch are kept to inspect intra-epoch progress.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    best_loss = float('inf')
    losses = []              # loss for each epoch
    first_last_losses = []   # (first batch, last batch) loss of each epoch

    for epoch in range(num_epochs):
        running_loss = 0.0

        first_batch_loss = None
        last_batch_loss = None

        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:

            for batch_idx, (cqt_sections, piano_roll_sections, num_time_bins) in enumerate(tepoch):

                # remove the DataLoader batch dimension (batch size = 1): sections act as the true batch
                cqt_sections = cqt_sections.squeeze(0).to(device)              # (S, 1, 156, 86)
                piano_roll_sections = piano_roll_sections.squeeze(0).to(device)  # (S, 53, 86)

                optimizer.zero_grad()

                batch_loss = 0.0

                # iterate over the sections of this track, accumulating gradients
                for i in range(cqt_sections.shape[0]):
                    cqt_section = cqt_sections[i].unsqueeze(0)                 # (1, 1, 156, 86)
                    piano_roll_section = piano_roll_sections[i].unsqueeze(0)   # (1, 53, 86)

                    output = model(cqt_section)                                # (1, 53, 86)

                    assert output.shape == piano_roll_section.shape, \
                        f"Output shape {output.shape} != Target shape {piano_roll_section.shape}"

                    loss = criterion(output, piano_roll_section)
                    batch_loss += loss.item()
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                batch_loss /= cqt_sections.shape[0]  # normalize by number of sections
                running_loss += batch_loss

                if batch_idx == 0:
                    first_batch_loss = batch_loss
                if batch_idx == len(tepoch) - 1:
                    last_batch_loss = batch_loss

                tepoch.set_postfix(loss=running_loss / (batch_idx + 1))

            epoch_loss = running_loss / len(tepoch)
            losses.append(epoch_loss)

            if first_batch_loss is not None and last_batch_loss is not None:
                first_last_losses.append((first_batch_loss, last_batch_loss))

            print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                print(f"Saving model with loss {epoch_loss:.4f} (improved from {best_loss:.4f})")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
                best_loss = epoch_loss

    return losses, first_last_losses
