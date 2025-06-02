import pandas as pd
import numpy as np
import os
from utils import get_openai_key
from openai import OpenAI
import json
import re


def load_acid(avail=True):
    """
    Load the Acid dataset.
    """
    if avail:
        df = pd.read_excel('data/acids_llm.xlsx')
        return df[df.columns[0]]
    if not os.path.exists('data/acid_names.npy'):
        df = pd.read_excel('data/acids_llm.xlsx')
        return df[df.columns[0]]
    return np.load('data/acid_names.npy')


def load_base(avail=True):
    """
    Load the base dataset.
    """
    if avail:
        df = pd.read_excel('data/bases_llm.xlsx')
        return df[df.columns[0]]
    if not os.path.exists('data/base_names.npy'):
        df = pd.read_excel('data/bases_llm.xlsx')
        return df[df.columns[0]]
    return np.load('data/base_names.npy')


def get_llm_embedding(model='text-embedding-3-small', chat_model='gpt-4o'):
    openai_key = get_openai_key()
    client = OpenAI(api_key=openai_key)
    acid = load_acid()
    base = load_base()
    acid_embedding = []
    base_embedding = []

    for text in acid:
        result = get_one_embedding(text, client, model, chat_model, type='acid')
        acid_embedding.append(result)

    for text in base:
        result = get_one_embedding(text, client, model, chat_model, type='base')
        base_embedding.append(result)

    return acid_embedding, base_embedding


def ask_llm_prompt(name):
    prompt = f"Please give me some chemical knowledge and properties of {name}."
    return prompt


def get_one_embedding(text, client, embedding_model='text-embedding-3-small', chat_model='gpt-4o', type='acid'):
    if os.path.exists(f"data/embeddings/{text}.json"):
        return json.load(open(f"data/embeddings/{text}.json", "r"))['embedding']

    uname_dict = get_uname_dict()
    if text in uname_dict:
        return json.load(open(f"data/embeddings/{uname_dict[text]}.json", "r"))['embedding']

    prompt = ask_llm_prompt(text)
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful chemical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    result = completion.choices[0].message.content

    embedding = client.embeddings.create(input=[result], model=embedding_model).data[0].embedding

    json_data = {
        "molecule": text,
        "property": result,
        "embedding": embedding,
        "type": type
    }

    if is_valid_windows_filename(text):
        json.dump(json_data, open(f"data/embeddings/{text}.json", "w"))
    else:
        uname_dict = get_uname_dict()
        value = 'name' + str(len(uname_dict))
        uname_dict[text] = value
        json.dump(json_data, open(f"data/embeddings/{value}.json", "w"))
        json.dump(uname_dict, open('data/uname_dict.json', 'w'))

    return embedding


def get_uname_dict():
    if not os.path.exists('data/uname_dict.json'):
        data = {}
        json.dump(data, open('data/uname_dict.json', 'w'))
        return data
    return json.load(open('data/uname_dict.json', 'r'))


def is_valid_windows_filename(filename):
    # 检查长度
    if len(filename) > 255:
        return False
    # 检查特殊字符
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, filename):
        return False
    # 检查是否与系统保留名称相同
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if filename.upper() in reserved_names:
        return False
    # 检查是否以空格或点结束
    if filename.endswith(' ') or filename.endswith('.'):
        return False
    return True
