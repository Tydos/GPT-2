import torch
from dataclasses import replace
from src.data.pretrain import create_dataloaders_from_text
from src.data.tokenizer import SimpleTokenizer
from src.model.config import NANO_MODEL, NANO_TRAIN
from src.data.utils import load_text
from src.model.gpt import GPTModel
import torch.nn.functional as F
from src.engine.generate import generate
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_text = load_text(file_path="artifacts/harrypotter.txt")
    tokenizer = SimpleTokenizer.from_text(raw_text)
    tokenizer.save("artifacts")
    train_loader, val_loader, test_loader = create_dataloaders_from_text(
        raw_text, tokenizer, seq_len=256, stride=64, batch_size=8,
        split=(0.7, 0.2, 0.1),
    )
    nano_model_cfg = replace(NANO_MODEL, vocab_size=len(tokenizer.str_to_int))
    model = GPTModel(nano_model_cfg).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=NANO_TRAIN.lr)

    print("num parameters: ", sum(p.numel() for p in model.parameters()))
    history = {"train": [], "val": [], "test": []}
    for epoch in range(NANO_TRAIN.num_epochs):
        print(f"Epoch {epoch + 1} of {NANO_TRAIN.num_epochs}")
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            output = model(inputs)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history["train"].append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                output = model(inputs)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)), targets.view(-1)
                )
                val_loss += loss.item()
        history["val"].append(val_loss / len(val_loader))

        print(f"Train loss: {train_loss / len(train_loader)}")
        print(f"Val loss: {val_loss / len(val_loader)}")

        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                output = model(inputs)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)), targets.view(-1)
                )
                test_loss += loss.item()
        history["test"].append(test_loss / len(test_loader))

        print(f"Test loss: {test_loss / len(test_loader)}")
        gen = generate(model, tokenizer, device, "The", num_tokens=20,
                       strategy="temperature", temperature=0.8,
                       context_length=nano_model_cfg.context_length)
        print(gen)

    torch.save(model.state_dict(), "artifacts/nano.pth")

    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.plot(history["test"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("artifacts/train_validation_curve.png")
