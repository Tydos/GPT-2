import torch
from src.data.dataset import create_gpt_dataloader2
from src.data.tokenizer import build_vocab, save_vocab, SimpleTokenizer
from src.model.config import NANO_GPT_CONFIG
from src.data.utils import load_text
from src.model.gpt import GPTModel
import torch.nn.functional as F
from src.engine.generate import generate_temperature
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_text = load_text(file_path="artifacts/harrypotter.txt")
    vocab = build_vocab(raw_text)
    save_vocab(vocab, file_path="artifacts")
    tokenizer = SimpleTokenizer(vocab)
    train_loader, val_loader, test_loader = create_gpt_dataloader2(
        raw_text,
        tokenizer,
        max_len=256,
        stride=64,
        batch_size=8,
    )
    model = GPTModel(NANO_GPT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=NANO_GPT_CONFIG.lr)

    print("num parameters: ", sum(p.numel() for p in model.parameters()))
    history = {"train": [], "val": [], "test": []}
    for epoch in range(NANO_GPT_CONFIG.num_epochs):
        print(f"Epoch {epoch + 1} of {NANO_GPT_CONFIG.num_epochs}")
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
        gen = generate_temperature(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt="The",
            num_tokens=20,
            temperature=0.8,
        )
        print(gen)

    torch.save(model.state_dict(), "artifacts/nano.pth")

    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.plot(history["test"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("artifacts/train_validation_curve.png")
