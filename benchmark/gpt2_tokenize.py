from transformers import GPT2Tokenizer
import os

# 加载 GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 输入和输出文件路径
input_file = "benchmark/datasets/lambada_test_plain_text.txt"
output_file = "benchmark/tokenized/lambada_token_ids.txt"

# 自动创建输出目录（如果不存在）
output_dir = "benchmark/tokenized"
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # 将整行文本编码为token id
        token_ids = tokenizer.encode(line, add_special_tokens=False)

        # 输出为空格分隔的 token ids
        fout.write(" ".join(str(tid) for tid in token_ids) + "\n")
