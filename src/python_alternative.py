import time

start = time.time()

import onnxruntime as rt
import pandas as pd
from transformers import (
    BertTokenizer,
)
import csv
import numpy as np

PRE_TRAINED_MODEL_NAME = "bert-base-cased"

sess = rt.InferenceSession("onnx_model.onnx")

sess.set_providers(["CUDAExecutionProvider"])
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 256
MAX_LEN = 60

PATH = "../medium.csv"

CHUNK_SIZE = 256

df = pd.read_csv(PATH, nrows=10000, chunksize=CHUNK_SIZE)

pre_infering = time.time()
l = []
for df_tiny in df:
    df_tiny = df_tiny[
        [
            "Title",
            "BodyMarkdown",
            "OpenStatus",
        ]
    ]

    df_tiny.loc[:, "Title"] = df_tiny["Title"] + " " + df_tiny["BodyMarkdown"]

    encoding = tokenizer(
        df_tiny["Title"].to_numpy().tolist(),
        add_special_tokens=True,
        max_length=60,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    pred_onx = sess.run(
        None,
        {
            sess.get_inputs()[0].name: encoding["input_ids"],
            sess.get_inputs()[1].name: encoding["attention_mask"],
        },
    )[0]
    l.append(pred_onx)

after_infering = time.time()

with open("python_output.csv", "w") as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(["ouput_0", "output_1"])
    write.writerows(np.concatenate(l))

print("pre infering time : %.1f ms" % (1000 * (pre_infering - start)))
print("after infering time : %.1f s" % ((after_infering - pre_infering)))
# print(pred_onx)
