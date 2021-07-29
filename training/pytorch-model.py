from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

BATCH_SIZE = 16
MAX_LEN = 60

TEST_NUMBER_OF_ROWS = 500

PRE_TRAINED_MODEL_NAME = "bert-base-cased"

device = torch.device("cuda:0")


PATH = "train_October_9_2012.csv"


def read_train(path: str):
    train = pd.read_csv(path, nrows=TEST_NUMBER_OF_ROWS)
    return train.sample(TEST_NUMBER_OF_ROWS)[
        [
            "Title",
            "BodyMarkdown",
            "OpenStatus",
        ]
    ]


df_tiny = read_train(PATH)

df_tiny.loc[:, "OpenStatus"] = df_tiny["OpenStatus"].map(
    lambda x: (x == "open") * 1
)
df_tiny.loc[:, "Title"] = df_tiny["Title"] + " " + df_tiny["BodyMarkdown"]

del df_tiny["BodyMarkdown"]

X_train, X_test, y_train, y_test = train_test_split(df_tiny, df_tiny)


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df["Title"].to_numpy(),
        targets=df["OpenStatus"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

train_data_loader = create_data_loader(X_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(X_test, tokenizer, MAX_LEN, BATCH_SIZE)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.quant_ids = torch.quantization.QuantStub()
        self.quant_masks = torch.quantization.QuantStub()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids, attention_mask):
        input_ids = self.quant_ids(input_ids)
        attention_mask = self.quant_masks(attention_mask)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out[-1]
        output = self.drop(hidden_states)
        output = self.dequant(output)
        return self.out(hidden_states)


model = SentimentClassifier(2)

for param in model.bert.parameters():
    param.requires_grad = False

model = model.to(device)


# Training

EPOCHS = 2

optimizer = AdamW(model.parameters(), lr=2e-4)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
weight = torch.tensor([40.0, 1.0])
loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)


def train_epoch(
    model, data_loader, loss_fn, optimizer, device, scheduler, n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for data in train_data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


best_accuracy = 0

for epoch in range(EPOCHS):

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(X_train),
    )

    print(f"Train loss {train_loss} accuracy {train_acc}")

    # val_acc, val_loss = eval_model(
    #     model, val_data_loader, loss_fn, device, len(X_test)
    # )

    # print(f"Val   loss {val_loss} accuracy {val_acc}")
    # print()


import time

start = time.time()

x = next(iter(val_data_loader))


input_ids = x["input_ids"].to(device)
attention_mask = x["attention_mask"].to(device)

end = time.time()

print("Total time : %.1f ms" % (1000 * (end - start) / 10))

# model must be set to eval mode for static quantization logic to work
model.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
# model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# model_fused = torch.quantization.fuse_modules(model, [["conv", "relu"]])
model_int8 = torch.quantization.quantize_dynamic(
    model,  # the original model
    {
        BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME),
        # torch.nn.Linear,
    },  # a set of layers to dynamically quantize
    dtype=torch.qint8,
)  # the target dtype for quantized weights
# model_prepared(input_ids=input_ids, attention_mask=input_ids)
# model_prepared.to(device)

# model_int8 = torch.quantization.convert(model_prepared)
torch.onnx.export(
    model_int8,
    args=(input_ids, attention_mask),
    f="onnx_model.onnx",
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input_0", "input_1"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input_0": {0: "batch_size"},  # variable length axes
        "input_1": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)
