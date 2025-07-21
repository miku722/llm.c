from transformers import GPT2Tokenizer
import os

def tokenize_file(input_file, output_file):
    # 加载 GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 自动创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 逐行读取并写入 token ids
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            fout.write(" ".join(str(tid) for tid in token_ids) + "\n")

# 示例调用（可根据需要替换路径）
if __name__ == "__main__":
    lambada_test_input_path = "benchmark/datasets/lambada_test_plain_text.txt"
    lambada_test_output_path = "benchmark/tokenized/lambada_token_ids.txt"
    tokenize_file(lambada_test_input_path, lambada_test_output_path)

    cbt_test_input_path = "benchmark/datasets/CBTest/data/cbt_full_context.txt"
    cbt_test_output_path = "benchmark/tokenized/cbtest_CN_token_ids.txt"
    tokenize_file(cbt_test_input_path, cbt_test_output_path)
