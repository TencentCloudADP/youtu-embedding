import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


class LLMEmbeddingModel():

    def __init__(self, 
                model_name_or_path, 
                batch_size=128, 
                max_length=1024, 
                gpu_id=0):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")

        self.device = torch.device(f"cuda:{gpu_id}")
        self.model.to(self.device).eval()

        self.max_length = max_length
        self.batch_size = batch_size

        query_instruction = "Given a search query, retrieve passages that answer the question"
        if query_instruction:
            self.query_instruction = f"Instruction: {query_instruction} \nQuery:"
        else:
            self.query_instruction = "Query:"

        self.doc_instruction = ""
        print(f"query instruction: {[self.query_instruction]}\ndoc instruction: {[self.doc_instruction]}")

    def mean_pooling(self, hidden_state, attention_mask):
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding
    
    @torch.no_grad()
    def encode(self, sentences_batch, instruction):
        inputs = self.tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            add_special_tokens=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs[0]

            instruction_tokens = self.tokenizer(
                instruction,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]
            if len(np.shape(np.array(instruction_tokens))) == 1:
                inputs["attention_mask"][:, :len(instruction_tokens)] = 0
            else:
                instruction_length = [len(item) for item in instruction_tokens]
                assert len(instruction) == len(sentences_batch)
                for idx in range(len(instruction_length)):
                    inputs["attention_mask"][idx, :instruction_length[idx]] = 0

            embeddings = self.mean_pooling(last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    def encode_queries(self, queries):
        queries = queries if isinstance(queries, list) else [queries]
        queries = [f"{self.query_instruction}{query}" for query in queries]
        return self.encode(queries, self.query_instruction)

    def encode_passages(self, passages):
        passages = passages if isinstance(passages, list) else [passages]
        passages = [f"{self.doc_instruction}{passage}" for passage in passages]
        return self.encode(passages, self.doc_instruction)

    def compute_similarity_for_vectors(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_similarity(self, queries, passages):
        q_reps = self.encode_queries(queries)
        p_reps = self.encode_passages(passages)
        scores = self.compute_similarity_for_vectors(q_reps, p_reps)
        scores = scores.detach().cpu().tolist()
        return scores


queries = ["What's the weather like?"]
passages = [
    'The weather is lovely today.',
    "It's so sunny outside!",
    'He drove to the stadium.'
]

model_name_or_path = "tencent/Youtu-Embedding"
model = LLMEmbeddingModel(model_name_or_path)
scores = model.compute_similarity(queries, passages)
print(f"scores: {scores}")
