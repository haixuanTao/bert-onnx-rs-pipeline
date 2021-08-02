import time

start = time.time()

import onnxruntime as rt
import pandas as pd
from transformers import (
    BertTokenizerFast,
)
import csv
import numpy as np

PRE_TRAINED_MODEL_NAME = "bert-base-cased"

sess = rt.InferenceSession("src/onnx_model.onnx")

sess.set_providers(["CUDAExecutionProvider"])
tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 256
MAX_LEN = 60

PATH = "medium.csv"

CHUNK_SIZE = 256

setup = time.time()

df = pd.read_csv(PATH, nrows=10000, chunksize=CHUNK_SIZE)
pre_infering = time.time()

delta_read = 0
delta_encoding = 0
delta_onnx = 0

l = []
for df_tiny in df:

    batch_start = time.time()
    df_tiny = df_tiny[
        [
            "Title",
            "BodyMarkdown",
            "OpenStatus",
        ]
    ]
    df_tiny.loc[:, "Title"] = df_tiny["Title"] + " " + df_tiny["BodyMarkdown"]
    delta_read += time.time() - batch_start

    encoding_start = time.time()

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

    delta_encoding += time.time() - encoding_start
    onnx_start = time.time()
    pred_onx = sess.run(
        None,
        {
            sess.get_inputs()[0].name: encoding["input_ids"],
            sess.get_inputs()[1].name: encoding["attention_mask"],
        },
    )[0]

    delta_onnx += time.time() - onnx_start
    l.append(pred_onx)

after_infering = time.time()

with open("python_output.csv", "w") as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(["ouput_0", "output_1"])
    write.writerows(np.concatenate(l))

save = time.time()


print("Boot up time : %.3f s" % ((setup - start)))
print("Read time : %.3f s" % ((delta_read)))
print("Encoding time : %.3f s" % ((delta_encoding)))
print("onnx time : %.3f s" % ((delta_onnx)))
print("after infering time : %.3f s" % ((save - after_infering)))
# print(pred_onx)
