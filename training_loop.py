import torch
import tqdm


def save_checkpoint(state, filename="model.pth"):
    print("Saving weights-->")
    torch.save(state, filename)


def training_loop(
    num_epochs, optimizer, model, criterion, train_loader, model_name, device
):
    model.train()
    min_loss = None

    for epoch in range(num_epochs):
        losses = []

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for _, (imgs, labels) in loop:

            imgs = imgs.to(device)
            
            labels = labels.to(device)

            output = model(imgs)

            loss = criterion(output, labels)

            losses.append(loss.detach().item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}] ")
            loop.set_postfix(loss=loss.detach())

        moving_loss = sum(losses) / len(losses)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        del imgs, labels
        torch.cuda.empty_cache()

        if min_loss == None:
            min_loss = moving_loss
            save_checkpoint(checkpoint, model_name)

        elif moving_loss < min_loss:
            min_loss = moving_loss
            save_checkpoint(checkpoint, model_name)

        print("Epoch {0} : Loss = {1}".format(epoch + 1, moving_loss))
