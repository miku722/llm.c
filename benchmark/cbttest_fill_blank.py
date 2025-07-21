def read_cbtest_with_candidates(filename):
    """
    读取cbtest格式文件，每组21行，最后一行包含候选词和正确答案。
    返回：
      [
        {
          'group_id': int,
          'context_lines': [str,...20行],
          'question_template': str,
          'candidates': [str,...],
          'answer': str
        },
        ...
      ]
    """
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        group_id = 0

        for line in f:
            line = line.strip()
            if line == "":
                if len(lines) == 21:
                    context_lines = lines[:20]

                    # 解析第21行
                    # 例如：21 Do you think she will see that young XXXXX sitting under the tree ? ' \t man \t birds|food|maiden|man|middle|place|sight|south|wash|wings
                    last_line = lines[20]
                    parts = last_line.split('\t')
                    question_with_num = parts[0]
                    answer = parts[1] if len(parts) > 1 else None
                    candidates_str = parts[3] if len(parts) > 3 else None

                    question_template = question_with_num.split(' ', 1)[1]  # 去掉编号

                    candidates = candidates_str.split('|') if candidates_str else []

                    results.append({
                        'group_id': group_id,
                        'context_lines': context_lines,
                        'question_template': question_template,
                        'candidates': candidates,
                        'answer': answer
                    })

                lines = []
                group_id += 1
            else:
                # 去除行首编号（如 "1 Mary went home." -> "Mary went home."）
                line_no_content = line.split(' ', 1)[1] if ' ' in line else line
                lines.append(line_no_content)

        # 处理最后一组无空行结尾
        if len(lines) == 21:
            context_lines = lines[:20]
            last_line = lines[20]
            parts = last_line.split('\t')
            question_with_num = parts[0]
            answer = parts[1] if len(parts) > 1 else None
            candidates_str = parts[2] if len(parts) > 2 else None

            question_template = question_with_num.split(' ', 1)[1]

            candidates = candidates_str.split('|') if candidates_str else []

            results.append({
                'group_id': group_id,
                'context_lines': context_lines,
                'question_template': question_template,
                'candidates': candidates,
                'answer': answer
            })

    return results


def generate_full_texts(data):
    """
    对每组数据，将question_template中的XXXXX替换为每个候选词，
    拼接完整上下文文本，返回[(group_id, candidate, full_text), ...]
    """
    full_texts = []
    for item in data:
        group_id = item['group_id']
        context = item['context_lines']
        template = item['question_template']
        candidates = item['candidates']
        answer = item['answer']

        for cand in candidates:
            replaced = template.replace('XXXXX', cand)
            full_text = " ".join(context + [replaced])
            full_texts.append((group_id, cand, full_text, answer))

    return full_texts


if __name__ == "__main__":
    filename = "benchmark/datasets/CBTest/data/cbtest_CN_test_2500ex.txt"  # 你的文件名
    data = read_cbtest_with_candidates(filename)
    full_texts = generate_full_texts(data)

    with open("benchmark/datasets/CBTest/data/cbt_full_context.txt", "w", encoding="utf-8") as out_file:
        for group_id, cand, text, answer in full_texts:
            out_file.write(f"Group {group_id} Candidate {cand}: Answer: {answer} \n{text}\n")

