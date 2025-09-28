#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
from mteb import MTEB
from transformers import AutoModel, AutoModelForCausalLM
from collections import OrderedDict
from cmteb_dres_model import CmtebDRESModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

task2instruction = {
    "STS": {
        "AFQMC": "为金融领域句子生成表示",
        "ATEC": "为金融领域客服场景的句子生成表示",
        "BQ": "为银行客服对话中的句子生成表示",
        "LCQMC": "为通用领域句子生成表示",
        "PAWSX": "为英文翻译得到的中文句子生成表示",
        "QBQTC": "为搜索引擎的搜索文本生成表示",
        "STSB": "为简短的通用领域句子生成表示",
        "STS22": "为新闻报道文本生成表示",
    },
    "IR": {
        "CmedqaRetrieval": "为医学问题生成表示，用于匹配相关医学文本",
        "CovidRetrieval": "为新冠疫情相关问题生成表示，用于匹配相关文本",
        "DuRetrieval": "为用户日常搜索问题生成表示，用于匹配网页内容",
        "EcomRetrieval": "为用户购物搜索问题生成表示，用于匹配商品属性文本",
        "MedicalRetrieval": "为患者症状描述文本生成表示，用于匹配诊疗方案或医学指南",
        "MMarcoRetrieval": "为简短搜索问题生成表示，用于匹配相关长文本网页内容",
        "T2Retrieval": "为通用领域问题生成表示，用于匹配相关文本",
        "VideoRetrieval": "为简短用户问题生成表示，用于匹配视频标题或视频描述",
    }
}

task2eninstruction = {
    "STS": {
        "AFQMC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
        "ATEC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
        "BQ": "Represent the user problem descriptions when handling bank credit business, retrieve semantically similar text",
        "LCQMC": "Represent the user question descriptions on general question-answering platforms, retrieve semantically similar text",
        "PAWSX": "Represent the Chinese Translations of English Encyclopedias, retrieve semantically similar text",
        "QBQTC": "Represent the web search query, retrieve semantically similar text",
        "STSB": "Represent the short general domain sentences, retrieve semantically similar text",
        "STS22": "Retrieve semantically similar text",
    },
    "IR": {
        "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
        "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
        "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
        "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
        "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
        "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    }
}

def build_instruction_prefix(model_name: str, instruction: str = "") -> str:
    """
    根据模型名称、任务指令和文本类型，生成与训练时一致的指令前缀。

    :param model_name: 模型名称或路径，用于判断模型类型。
    :param instruction: 具体的任务指令文本，例如 "为这个句子生成表示以用于检索"。
    :return: 构建好的指令前缀字符串。
    """
    model_name_lower = model_name.lower()
    is_minicpm_or_e5 = "minicpm" in model_name_lower or "e5" in model_name_lower
    is_bge = "bge" in model_name_lower

    prefix = ""
    # IR 任务的 Query 部分 或 STS 任务
    if instruction:
        prefix = f"Instruction: {instruction} Query: "

    # 根据模型类型添加起始 Token
    if is_minicpm_or_e5:
        prefix = f"<s>{prefix}"
    elif is_bge:
        prefix = f"[CLS]{prefix}"

    return prefix


def build_passage_instruction_prefix(model_name: str, instruction: str = "") -> str:
    """
    根据模型名称、任务指令和文本类型，生成与训练时一致的指令前缀。

    :param model_name: 模型名称或路径，用于判断模型类型。
    :param instruction: 具体的任务指令文本，例如 "为这个句子生成表示以用于检索"。
    :return: 构建好的指令前缀字符串。
    """
    model_name_lower = model_name.lower()
    is_minicpm_or_e5 = "minicpm" in model_name_lower or "e5" in model_name_lower
    is_bge = "bge" in model_name_lower

    prefix = ""
    # IR 任务的 Passage 部分
    if is_minicpm_or_e5:
        prefix = "<s>"
    elif is_bge:
        prefix = "[CLS]Passage: "

    return prefix


def load_model(args):
    base_model_path = args.base_model_path
    checkpoint_path = args.checkpoint_path

    if base_model_path != checkpoint_path:
        print("=" * 60)
        print("Mode: Evaluating fine-tuned CHECKPOINT")
        print(f"Base model for architecture: {base_model_path}")
        print(f"Checkpoint for weights:      {checkpoint_path}")
        print("=" * 60)

        print(f"Step 1: Creating a PURE AutoModel shell from '{base_model_path}' with correct dtype...")
        if "youtu_" in base_model_path:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
            model_shell = base_model.model
        else:
            model_shell = AutoModel.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )

        print(f"Step 2: Loading state_dict from checkpoint '{checkpoint_path}'...")
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        
        print("Step 3: Cleaning state_dict keys by stripping the 'model.' prefix...")
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('model.'):
                cleaned_state_dict[k[6:]] = v

        print("Step 4: Loading the cleaned state_dict into the PURE AutoModel shell...")
        missing_keys, unexpected_keys = model_shell.load_state_dict(cleaned_state_dict, strict=False)
        print(f"State_dict loaded. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        print("Step 5: Initializing the evaluation model (CmtebDRESModel)...")
        model = CmtebDRESModel(
            model_object=model_shell,
            model_name_or_path=checkpoint_path,
            pooling_method=args.pooling_method,
            normalize_embeddings=True,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
        )
    else:
        print("=" * 60)
        print("Mode: Evaluating RAW model (base_model_path == checkpoint_path)")
        print(f"Loading raw model directly from: {base_model_path}")
        print("=" * 60)

        # 直接使用最简单的方式初始化评测模型，让 CmtebDRESModel 内部处理加载
        # 此时不传递 model_object，触发其内部的 from_pretrained 逻辑
        model = CmtebDRESModel(
            model_name_or_path=base_model_path,
            pooling_method=args.pooling_method,
            normalize_embeddings=True,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
        )

    return model


def set_instruction(model, task, args):
    print("========== use task instruction ==========")

    if task in task2instruction["IR"]:
        ir_instruction_for_query = task2instruction["IR"][task]

        model.ir_instruction_for_query = build_instruction_prefix(
            model_name=args.base_model_path,
            instruction=ir_instruction_for_query
        )

        model.ir_instruction_for_passage = build_passage_instruction_prefix(
            model_name=args.base_model_path
        )

        print(f"query_instruction: {model.ir_instruction_for_query}")
        print(f"passage_instruction: {model.ir_instruction_for_passage}")
    elif task in task2instruction["STS"]:
        sts_instruction = task2instruction["STS"][task]

        model.sts_instruction = build_instruction_prefix(
            model_name=args.base_model_path,
            instruction=sts_instruction
        )

        print(f"sts_instruction: {model.sts_instruction}")
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', default="/data/workspace/zhangbowen/model/MiniCPM-Embedding", type=str)
    parser.add_argument('--checkpoint_path', default="/data/workspace/zhangbowen/output/checkpoint-5013", type=str)
    parser.add_argument('--pooling_method', default="mean", type=str)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    parser.add_argument('--task_names', default="DuRetrieval", type=str)  # 自 task2instruction 中选择即可
    parser.add_argument('--use_task_instruction', action="store_true", default=True, help="use task prompt")
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Args: {args}")

    model = load_model(args)

    ir_tasks = ["VideoRetrieval", "EcomRetrieval", "CmedqaRetrieval", "MMarcoRetrieval",
                "MedicalRetrieval", "CovidRetrieval", "DuRetrieval", "T2Retrieval"]
    
    sts_tasks = ['AFQMC', 'STS22', 'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'QBQTC']
    task_names = ir_tasks + sts_tasks

    print(task_names)

    for task in task_names:
        print(f"========== {task} ==========")

        model.sts_instruction = None
        model.ir_instruction_for_query = None
        model.ir_instruction_for_passage = None
        if args.use_task_instruction:
            model = set_instruction(model, task, args)

        evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
        exp_name, ckpt_name = args.checkpoint_path.split('/')[-2:]

        task_type = "IR" if task in task2instruction["IR"] else "STS"
        output_folder = f"result/{task_type}/{exp_name}_{ckpt_name}"
        print(f"output folder: {output_folder}")

        try:
            evaluation.run(model, output_folder=output_folder)
        except:
            pass


if __name__ == '__main__':
    main()
