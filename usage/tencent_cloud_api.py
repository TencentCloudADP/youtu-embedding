# -*- coding: utf-8 -*-
# Reference: https://cloud.tencent.com/document/product/1772/115343
import os
import json
import types
import numpy as np
from typing import List
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.lkeap.v20240522 import lkeap_client, models


def encode(client, inputs, is_query=False):
    if is_query:
        instruction = "Instruction: Given a search query, retrieve passages that answer the question \nQuery:"
    else:
        instruction = ""

    params = {
        "Model": model_name,
        "Inputs": inputs,
        "Instruction": instruction
    }

    req = models.GetEmbeddingRequest()
    req.from_json_string(json.dumps(params))

    resp = client.GetEmbedding(req)
    resp = json.loads(resp.to_json_string())
    outputs =[item["Embedding"] for item in resp["Data"]]
    return outputs

secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")

cred = credential.Credential(secret_id, secret_key)

httpProfile = HttpProfile()
httpProfile.endpoint = "lkeap.ap-guangzhou.tencentcloudapi.woa.com"

clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = lkeap_client.LkeapClient(cred, "ap-guangzhou", clientProfile)
model_name = "youtu-embedding-llm-v1"

inputs = ["Regular exercise is the key to staying healthy."]
embeddings = encode(client, inputs, is_query=False)
print(embeddings)