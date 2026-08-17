#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>


typedef struct {
    int dim;            // dimension de vecteur (les fameux 288)
    int hidden_dim;     // dimensions cachés du FFN
    int n_layers;       // nombres de couches
    int n_heads;        // nombre de tetes d'attention
    int n_kv_heads;     // nombre de tetes clé/valeurs
    int vocab_size;     // taille du vocabulaire
    int seq_len;        // longueur séquence max
} Config;



typedef struct {
    float* token_embedding_table;   // vocab_size * dim = 32000 * 288
    float* rms_att_weights;         // dim = 288 * n_layers
    float* wq;                      // dim * dim = 288 * 288 * n_layers
    float* wk;                      // same
    float* wv;                      // same
    float* wo;                      // same
    float* rms_ffn_weights;         // dim = 288 * n_layers
    float* w1;                      // dim * hidden_dim = 288 * 768
    float* w2;                      // hidden_dim * dim = 768 * 288
    float* w3;                      // dim * hidden_dim = 288 * 768
    float* rms_final_weights;       // dim = 288
    float* freq_cis_real;           // seq_len * (dim / n_heads) / 2
    float* freq_cis_imag;          // same as freq_cis_real
} TransformerWeights;


float* load_model(char* filename, Config* config) {

    FILE* file = fopen(filename, "rb");
    if(!file) {
        fprintf(stderr, "No such file !\n");
        exit(EXIT_FAILURE);
    }

    if(fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    
    if(fseek(file, 0, SEEK_END) != 0 ) { exit(EXIT_FAILURE); }
    long fileSize = ftell(file);
    if (fileSize == -1) { exit(EXIT_FAILURE); }

    fclose(file);


    size_t token_embedding_table = (size_t)config->vocab_size * (size_t)config->dim;

    size_t rms_final_weights = (size_t)config->dim;

    size_t freq_cis_real = (size_t)config->seq_len * ((size_t)config->dim / (size_t)config->n_heads) / 2;
    size_t freq_cis_imag = (size_t)config->seq_len * ((size_t)config->dim / (size_t)config->n_heads) / 2;

    size_t wq, wk, wv, wo, rms_ffn_weights, rms_att_weights, w1, w2, w3;
    wq = (size_t)config->dim * (size_t)config->dim;
    wk = (size_t)config->dim * (size_t)config->dim;
    wv = (size_t)config->dim * (size_t)config->dim;
    wo = (size_t)config->dim * (size_t)config->dim;
    rms_ffn_weights = (size_t)config->dim;
    rms_att_weights = (size_t)config->dim;
    w1 = (size_t)config->dim * (size_t)config->hidden_dim;
    w2 = (size_t)config->hidden_dim * (size_t)config->dim;
    w3 = (size_t)config->dim * (size_t)config->hidden_dim;
    
    size_t total = 0;
    size_t layer = 0;
    size_t layer_weights[9] = {
        wq,
        wk,
        wv,
        wo,
        rms_att_weights,
        rms_ffn_weights,
        w1,
        w2,
        w3
    };

    for(int i = 0; i<sizeof(layer_weights)/sizeof(size_t);i++) {
        if(layer > SIZE_MAX - layer_weights[i]) {
            fprintf(stderr,"Overflow detected in layer weights !\nWeight overflow: %zu\n", layer_weights[i]);
            exit(EXIT_FAILURE);
        }
        layer += layer_weights[i];
    }
    

    if(total > SIZE_MAX - token_embedding_table) {
        fprintf(stderr,"Overflow detected at token embedding table !\n");
        exit(EXIT_FAILURE);
    }
    total += token_embedding_table;

    if(total > SIZE_MAX - rms_final_weights) {
        fprintf(stderr,"Overflow detected at rms final weights !\n");
        exit(EXIT_FAILURE);
    }
    total += rms_final_weights;

    if(total > SIZE_MAX - freq_cis_real) {
        fprintf(stderr,"Overflow detected at freq cis real !\n");
        exit(EXIT_FAILURE);
    }
    total += freq_cis_real;

    if(total > SIZE_MAX - freq_cis_imag) {
        fprintf(stderr,"Overflow detected at freq cis imag !\n");
        exit(EXIT_FAILURE);
    }
    total += freq_cis_imag;

    for(int i = 0; i<config->n_layers; i++) {
        if(total > SIZE_MAX - layer) {
            fprintf(stderr,"Overflow detected in layers total !\n");
            exit(EXIT_FAILURE);
        }
        total += layer;
    }

    if(total > SIZE_MAX / 4) {
        fprintf(stderr, "Overflow in total * 4 !\n");
        exit(EXIT_FAILURE);
    }
    size_t weights_bytes = total * 4;

    size_t expected_file_size = sizeof(Config);

    if(expected_file_size > SIZE_MAX -  weights_bytes) {
        fprintf(stderr, "Overflow in sum expected file size !\n");
        exit(EXIT_FAILURE);
    }
    expected_file_size += weights_bytes;

    printf("Total: %zu floats\n", total);
    printf("Expected file size: %zu bytes\n", expected_file_size);
    printf("File size: %ld bytes\n", fileSize);
        

    if(expected_file_size != fileSize) {
        fprintf(stderr, "Error, file size not matching !\n");
        exit(EXIT_FAILURE);
    }
    printf("\nFile size match !\n");



    // ------- Load weights ------------

    int weights_file = open(filename, O_RDONLY);
    if(weights_file == -1) {
        fprintf(stderr, "Error in open filename !\n");
        exit(EXIT_FAILURE);
    }

    float* mapped_weights = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, weights_file, 0);
    if(mapped_weights == MAP_FAILED) {
        fprintf(stderr, "Error mapping weights (mmap) !\n");
        exit(EXIT_FAILURE);
    }

    close(weights_file);

    return mapped_weights;
}

void map_weights(float* weights, Config* config, TransformerWeights* w) {

    float* weights_pointer = weights;
    
    w->token_embedding_table = weights_pointer;
    weights_pointer += (size_t)config->dim * config->vocab_size;
    
    w->rms_att_weights = weights_pointer;
    weights_pointer += (size_t)config->dim * config->n_layers;

    w->wq = weights_pointer;
    weights_pointer += (size_t)config->dim * config->dim * config->n_layers;

    w->wk = weights_pointer;
    weights_pointer += (size_t)config->dim * config->dim * config->n_layers;

    w->wv = weights_pointer;
    weights_pointer += (size_t)config->dim * config->dim * config->n_layers;

    w->wo = weights_pointer;
    weights_pointer += (size_t)config->dim * config->dim * config->n_layers;

    w->rms_ffn_weights = weights_pointer;
    weights_pointer += (size_t)config->dim * config->n_layers;

    w->w1 = weights_pointer;
    weights_pointer += (size_t)config->dim * config->hidden_dim * config->n_layers;

    w->w2 = weights_pointer;
    weights_pointer += (size_t)config->dim * config->hidden_dim * config->n_layers;

    w->w3 = weights_pointer;
    weights_pointer += (size_t)config->dim * config->hidden_dim * config->n_layers;

    w->rms_final_weights = weights_pointer;
    weights_pointer += (size_t)config->dim;

    w->freq_cis_real = weights_pointer;
    weights_pointer += (size_t)config->seq_len * (config->dim / config->n_heads) / 2;

    w->freq_cis_imag = weights_pointer;
    weights_pointer += (size_t)config->seq_len * (config->dim / config->n_heads) / 2;
    
    printf("[DEBUG] Final cursor position: %ld floats\n", weights_pointer - weights);

}

void matmul(float* output, float* input, float* weight, int input_size, int output_size) {

    for(int i = 0; i < output_size; i++) {

        float sum = 0;

        for(int j = 0; j < input_size; j++) {

            sum += weight[i * input_size + j] * input[j];
        }

        output[i] = sum;

    }
}



void forward(TransformerWeights* w, Config* config, int* tokens, int token_count) {


    float key_cache[config->dim * token_count];
    float value_cache[config->dim * token_count];

    memset(key_cache, 0, sizeof(key_cache));
    memset(value_cache, 0, sizeof(value_cache));


    for(int i = 0; i < token_count; i++) {

        float* token_embedding_pointer = w->token_embedding_table + tokens[i] * config->dim;
        matmul(key_cache + i * config->dim, token_embedding_pointer, w->wk, config->dim, config->dim);
        matmul(value_cache + i * config->dim, token_embedding_pointer, w->wv, config->dim, config->dim);
    }


    float query[config->dim];
    float* last_query_token = w->token_embedding_table + tokens[token_count - 1] * config->dim;
    matmul(query, last_query_token, w->wq, config->dim, config->dim);

    float scores[token_count];

    for(int i = 0; i < token_count; i++) {
        scores[i] = 0;
        for(int j = 0; j < config->dim; j++) {
            scores[i] += query[j] * key_cache[i * config->dim + j];
        }
    }

    float sum = 0;
    for(int i = 0; i < token_count; i++) {
        scores[i] = expf(scores[i]);
        sum += scores[i];
    }
    
    for(int i = 0; i < token_count; i++) {
        scores[i] = scores[i] / sum;
        printf("Score: %f\n", scores[i]); 

    }

    float attention_output[config->dim];
    for(int i = 0; i < config->dim; i++) {
        attention_output[i] = 0;
        for(int j = 0; j < token_count; j++) {
            attention_output[i] += scores[j] * value_cache[j * config->dim + i];
        }
    }

    for(int i = 0; i < 5; i++) {
        printf("Attnetion output: %f\n", attention_output[i]);
    }
    
}


int main() {

    Config data;
    TransformerWeights w;

    float* weights_raw = load_model("stories15M.bin", &data);
    float* weights = weights_raw + sizeof(Config)/sizeof(float);

    map_weights(weights, &data, &w);

    int tokens[3] = {306, 5169, 1002};
    forward(&w, &data, tokens, 3);


    return 0;
}

