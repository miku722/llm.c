#define TESTING
#include "../train_gpt2.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKENS_PER_LINE 1024
#define LOG_FIRST_N 0  // 前多少个样本打印详细 token
#define PRINT_WRONG_CASES 0 // 是否打印错误样本日志

int main(int argc, char *argv[]) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    FILE *fp = fopen("benchmark/tokenized/lambada_token_ids.txt", "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file.\n");
        perror("fopen failed");
        return 1;
    }

    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;

    int total = 0, correct = 0;
    double log_prob_sum = 0.0;

    while ((linelen = getline(&line, &linecap, fp)) != -1) {
        if (linelen <= 1) continue;

        if (line[linelen - 1] == '\n' || line[linelen - 1] == '\r') {
            line[--linelen] = '\0';
        }

        int tokens[MAX_TOKENS_PER_LINE];
        int n_tokens = 0;
        char *token_str = strtok(line, " ");
        while (token_str && n_tokens < MAX_TOKENS_PER_LINE) {
            tokens[n_tokens++] = (int)atoi(token_str);
            token_str = strtok(NULL, " ");
        }

        if (n_tokens < 2) continue;

        int prompt_len = n_tokens - 1;
        int target_token = tokens[n_tokens - 1]; // gukai@20250715: The last token is the target

        gpt2_forward(&model, tokens, NULL, 1, prompt_len);

        float *logits = model.acts.probs + (prompt_len - 1) * model.config.padded_vocab_size;

        // Find max logit
        int predicted_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < model.config.vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                predicted_token = i;
            }
        }

        // Compute softmax probability
        double sum_exp = 0.0;
        for (int i = 0; i < model.config.vocab_size; ++i) {
            sum_exp += expf(logits[i] - max_logit);
        }

        float target_logit = logits[target_token];
        float prob = expf(target_logit - max_logit) / (float)sum_exp;
        if (prob < 1e-9f) prob = 1e-9f;
        double logp = logf(prob);
        log_prob_sum += logp;

        // 输出日志：前几条样本详细信息
        if (total < LOG_FIRST_N) {
            printf("=== Sample #%d ===\n", total + 1);
            printf("Tokens: ");
            for (int i = 0; i < n_tokens; ++i) printf("%d ", tokens[i]);
            printf("\nTarget token: %d\n", target_token);
            printf("Predicted token: %d\n", predicted_token);
            printf("Prob(target) = %.8f\n", prob);
            printf("LogProb(target) = %.6f\n", logp);
            printf("Perplexity(sample) = %.2f\n", expf(-logp));
        }

        // 错误样本日志
        #if PRINT_WRONG_CASES
        if (predicted_token != target_token && total < 50) {
            printf("[Wrong] #%d → True: %d | Pred: %d | P(target): %.6f\n",
                   total + 1, target_token, predicted_token, prob);
        }
        #endif

        if (predicted_token == target_token) correct++;
        total++;

        if (total % 10 == 0) {
            printf("[Progress] %d samples → Acc: %.4f | PPL so far: %.2f\n",
                   total, (double)correct / total, expf(-log_prob_sum / total));
        }
    }

    free(line);
    fclose(fp);

    printf("\n== LAMBADA Benchmark Evaluation ==\n");
    printf("Total samples      : %d\n", total);
    printf("Correct predictions: %d\n", correct);
    printf("Final Accuracy     : %.4f\n", (double)correct / total);
    printf("Final Perplexity   : %.4f\n", expf(-log_prob_sum / total));

    return 0;
}
