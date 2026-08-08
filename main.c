#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


typedef struct {
    int dim;            // dimension de vecteur (les fameux 288)
    int hidden_dim;     // dimensions cachés du FFN
    int n_layers;       // nombres de couches
    int n_heads;        // nombre de tetes d'attention
    int n_kv_heads;     // nombre de tetes clé/valeurs
    int vocab_size;     // taille du vocabulaire
    int seq_len;        // longueur séquence max
} Config;


void read_file(char* filename, Config* config) {

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
    size_t freq_cis_image = (size_t)config->seq_len * ((size_t)config->dim / (size_t)config->n_heads) / 2;

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

    if(total > SIZE_MAX - freq_cis_image) {
        fprintf(stderr,"Overflow detected at freq cis image !\n");
        exit(EXIT_FAILURE);
    }
    total += freq_cis_image;

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

}

int main() {

    Config data;
    read_file("stories15M.bin", &data);

    printf("\n---Debug Info Values---\n");
    printf("dim: %d\n", data.dim);
    printf("hidden dim: %d\n", data.hidden_dim);
    printf("layers: %d\n", data.n_layers);
    printf("heads: %d\n", data.n_heads);
    printf("kv heads: %d\n", data.n_kv_heads);
    printf("vocab size: %d\n", data.vocab_size);
    printf("sequence lenght: %d\n", data.seq_len);

    printf("sizeof Config: %zu\n", sizeof(Config));


    return 0;
}

