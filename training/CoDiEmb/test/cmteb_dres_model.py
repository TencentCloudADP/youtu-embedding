#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from tqdm import tqdm
from typing import List, Dict, Union

import torch
import numpy as np
from mteb import DRESModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class CmtebDRESModel(DRESModel):
    def __init__(
            self,
            model_object: torch.nn.Module = None,
            model_name_or_path: str = None,
            pooling_method: str = "mean",
            normalize_embeddings: bool = True,
            max_length: int = 1024,
            batch_size: int = 16,
            gpu_id: int = 0,
            **kwargs
    ) -> None:
        print(f"Init model {model_name_or_path} ...")

        self.input_is_causal = False
        if model_object is not None:
            print("Initializing CmtebDRESModel with a pre-loaded model object.")
            self.model = model_object
        elif model_name_or_path is not None:
            print(f"Initializing CmtebDRESModel from path: {model_name_or_path} ...")

            if "minicpm" in model_name_or_path.lower() or "jina" in model_name_or_path.lower():
                print("Using MiniCPM or Jina model, setting torch_dtype to bfloat16 and attn_implementation to sdpa.")
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            elif "youtu_" in model_name_or_path.lower():
                base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
                self.model = base_model.model
                self.input_is_causal = True
            else:
                print("Initializing with default settings.")
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            raise ValueError("Either model_name_or_path or model_object must be provided.")
        
        tokenizer_path = model_name_or_path if model_object is None else kwargs.get("tokenizer_path", model_name_or_path)
        
        if tokenizer_path is None:
             raise ValueError("tokenizer_path must be provided when using a model_object.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        self.normalize_embeddings = normalize_embeddings
        self.model_name_or_path = model_name_or_path
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.max_length = max_length

        # These will be set dynamically by the evaluation script
        self.sts_instruction = None
        self.ir_instruction_for_query = None
        self.ir_instruction_for_passage = None

        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device).eval()

    def _prepare_texts(self, texts: List[str], instruction: str) -> List[str]:
        """
        A centralized method to prepare texts by adding model-specific
        prefixes (instructions) and suffixes.
        """
        suffix = ""
        model_name_lower = self.model_name_or_path.lower()
        
        if "e5" in model_name_lower:
            suffix = "</s>"
        elif "bge" in model_name_lower:
            suffix = "[SEP]"

        prefix = instruction if instruction is not None else ""
        return [f"{prefix}{text}{suffix}" for text in texts]

    @torch.no_grad()
    def get_embedding(self, sentences: List[str], instruction="", **kwargs) -> np.ndarray:
        all_embeddings = []
        
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences) <= 512):
            batch = sentences[start_index : start_index + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                add_special_tokens=False, # Crucial: we are adding special tokens manually
                return_offsets_mapping=True,
            ).to(self.device)

            offsets = inputs.pop("offset_mapping")

            with torch.no_grad():
                if self.input_is_causal:
                    inputs['is_causal'] = False

                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state

                if self.pooling_method == 'mean':
                    instruction_char_lens = [len(instruction)] * len(batch)
                    
                    # 将指令长度列表转为 Tensor，并增加一个维度用于广播，shape: (batch_size, 1)
                    instruction_lens_tensor = torch.tensor(
                        instruction_char_lens, 
                        device=offsets.device
                    ).unsqueeze(1)

                    # 获取所有 token 的结束偏移量
                    #    offsets shape: (batch_size, seq_len, 2)
                    #    end_offsets shape: (batch_size, seq_len)
                    end_offsets = offsets[:, :, 1]

                    # 通过广播进行一次性向量化比较
                    #    (batch_size, seq_len) > (batch_size, 1) -> (batch_size, seq_len)
                    pooling_mask = (end_offsets > instruction_lens_tensor).to(inputs["attention_mask"].dtype)
                    
                    # 与原始 attention_mask 相乘，屏蔽 padding
                    pooling_mask = pooling_mask * inputs["attention_mask"]
                    embeddings = self.mean_pooling(last_hidden_state, pooling_mask)
                elif self.pooling_method == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unsupported pooling method: {self.pooling_method}.")
                
                if self.normalize_embeddings:
                    in_dtype = embeddings.dtype
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
                
                embeddings = embeddings.detach().cpu().to(torch.float32).numpy()

            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for sts task
        '''
        input_texts = self._prepare_texts(sentences, self.sts_instruction)
        print(f"encode {len(sentences)} sentences; samples: {[item[:50] for item in input_texts[:2]]}")
        return self.get_embedding(input_texts, self.sts_instruction)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for ir task
        '''
        input_texts = self._prepare_texts(queries, self.ir_instruction_for_query)
        print(f"encode {len(queries)} queries; samples: {[item[:50] for item in input_texts[:2]]}")
        return self.get_embedding(input_texts, self.ir_instruction_for_query)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for ir task
        '''
        if isinstance(corpus[0], dict):
            texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            texts = corpus

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        input_texts = self._prepare_texts(texts, self.ir_instruction_for_passage)
        
        print(f"encode {len(corpus)} docs; samples: {[item[:50] for item in input_texts[:2]]}")
        return self.get_embedding(input_texts, self.ir_instruction_for_passage)

    def mean_pooling(self, hidden_state, attention_mask):
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding
