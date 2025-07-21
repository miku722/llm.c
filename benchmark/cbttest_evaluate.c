#define TESTING
#include "../train_gpt2.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKENS 2048
#define GROUP_SIZE 10

float compute_sequence_log_prob(GPT2 *model, int *tokens, int len) {
    float total_logprob = 0.0f;
    for (int i = 1; i < len; ++i) {
        gpt2_forward(model, tokens, NULL, 1, i);
        float *logits = model->acts.probs + (i - 1) * model->config.padded_vocab_size;

        float max_logit = logits[0];
        for (int j = 1; j < model->config.vocab_size; ++j) {
            if (logits[j] > max_logit) max_logit = logits[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < model->config.vocab_size; ++j) {
            sum_exp += expf(logits[j] - max_logit);
        }

        float prob = expf(logits[tokens[i]] - max_logit) / sum_exp;
        if (prob < 1e-9f) prob = 1e-9f;
        total_logprob += logf(prob);
    }
    return total_logprob;
}

int main(int argc, char *argv[]) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    FILE *fp = fopen("benchmark/tokenized/cbtcn_token_ids.txt", "r");
    if (!fp) {
        fprintf(stderr, "Cannot open input file.\n");
        return 1;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    int total = 0, correct = 0;
    double log_prob_sum = 0.0;

    // 每组最多 10 个候选项
    int tokens_group[GROUP_SIZE][MAX_TOKENS];
    int lengths[GROUP_SIZE];
    float log_probs[GROUP_SIZE];
    int is_corrects[GROUP_SIZE];

    while (1) {
        int group_count = 0;

        // 读取一组（10个）候选项
        while (group_count < GROUP_SIZE && (read = getline(&line, &len, fp)) != -1) {
            if (read < 10) continue;

            char *answer_ptr = strstr(line, "Answer:");
            char *cand_ptr = strstr(line, "Candidate");
            if (!answer_ptr || !cand_ptr) continue;

            char answer_word[128], cand_word[128];
            sscanf(answer_ptr, "Answer: %s", answer_word);
            sscanf(cand_ptr, "Candidate %s:", cand_word);

            is_corrects[group_count] = (strcmp(answer_word, cand_word) == 0);

            // 读取下一行（token ids）
            read = getline(&line, &len, fp);
            if (read == -1) break;

            int *tokens = tokens_group[group_count];
            int n_tokens = 0;
            char *tok = strtok(line, " \n");
            while (tok && n_tokens < MAX_TOKENS) {
                tokens[n_tokens++] = atoi(tok);
                tok = strtok(NULL, " \n");
            }
            if (n_tokens < 2) continue;

            lengths[group_count] = n_tokens;
            log_probs[group_count] = compute_sequence_log_prob(&model, tokens, n_tokens);

            group_count++;
        }

        // if (group_count < GROUP_SIZE) break; // 文件读完，结束

        // 找出最大 logp 的候选索引
        int best_idx = 0;
        for (int i = 1; i < GROUP_SIZE; ++i) {
            if (log_probs[i] > log_probs[best_idx]) {
                best_idx = i;
            }
        }

        log_prob_sum += log_probs[best_idx];

        if (is_corrects[best_idx]) {
            correct++;
        }
        total++;

        if (total % 10 == 0) {
            printf("[Progress] %d samples → Acc: %.4f | PPL: %.2f\n",
                   total, (double)correct / total, expf(-log_prob_sum / total));
        }
    }

    free(line);
    fclose(fp);

    printf("\n== CBT-CN Evaluation (Grouped Sentence LogP Max Selection) ==\n");
    printf("Total samples      : %d\n", total);
    printf("Correct predictions: %d\n", correct);
    printf("Final Accuracy     : %.4f\n", (double)correct / total);
    printf("Final Perplexity   : %.4f\n", expf(-log_prob_sum / total));

    return 0;
}
