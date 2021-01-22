#include "util.h"

__global__ void kmer(char *reference_str, char *reads, int reference_len, int read_len, int k, int *read_results) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j + k <= read_len) {
        int count = 0;
        int loc = i * MAX_READ_LENGTH + j;

        for (int p = 0; p <= reference_len - k; p++) {
            int equal = 1;
            for (int t = 0; t < k; t++)
                if (reads[loc + t] != reference_str[p + t])
                    equal = 0;

            count += equal;
        }

        read_results[loc] = count;
    }
}

int main(int argc, char** argv)
{
    if(argc != 5) {
        printf("Wrong argments usage: ./kmer_parallel [REFERENCE_FILE] [READ_FILE] [k] [OUTPUT_FILE]\n" );
    }

    FILE *fp;
    int k;

    char *reference_str = (char*) malloc(MAX_REF_LENGTH * sizeof(char));
    char *reference_filename, *read_filename, *output_filename;

    reference_filename = argv[1];
    read_filename = argv[2];
    k = atoi(argv[3]);
    output_filename = argv[4];

    fp = fopen(reference_filename, "r");
    if (fp == NULL) {
        printf("Could not open file %s!\n",reference_filename);
        return 1;
    }

    if (fgets(reference_str, MAX_REF_LENGTH, fp) == NULL) { //A single line only
        printf("Problem in file format!\n");
        return 1;
    }

    substring(reference_str, 0, strlen(reference_str)-1);
    fclose(fp);

    //Read queries
    StringList queries;

    initStringList(&queries, 3);  // initially 3 elements
    int success = read_file(read_filename, &queries);

    int read_len = strlen(queries.array[0]) - 1;
    int reference_len = strlen(reference_str);
    
    char *d_reads;
    char *d_reference_str; 
    int *d_read_results;
    cudaMalloc(&d_reads, queries.size * MAX_READ_LENGTH * sizeof(char));
    cudaMalloc(&d_read_results, queries.size * MAX_READ_LENGTH * sizeof(int));
    cudaMalloc(&d_reference_str, reference_len * sizeof(char));

    char *reads_1d = (char*) malloc(queries.size * MAX_READ_LENGTH * sizeof(char));
    for (int i = 0; i < queries.size; i++)
        for (int j = 0; j < MAX_READ_LENGTH; j++)
            reads_1d[i * MAX_READ_LENGTH + j] = queries.array[i][j];

    cudaMemcpy(d_reads, reads_1d, queries.size * MAX_READ_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reference_str, reference_str, reference_len * sizeof(char), cudaMemcpyHostToDevice);

    int n_blocks = queries.used;
    int n_threads = 256;
	kmer<<<n_blocks,n_threads>>>(d_reference_str, d_reads, reference_len, read_len, k, d_read_results);
    cudaDeviceSynchronize();

    int *read_results = (int*) malloc(queries.size * MAX_READ_LENGTH * sizeof(int));
    cudaMemcpy(read_results, d_read_results, queries.size * MAX_READ_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

    fp = fopen(output_filename, "w");
    for (int q = 0; q < queries.used; q++) {
        int count = 0;
        for (int i = 0; i < read_len; i++)
            count += read_results[q * MAX_READ_LENGTH + i];
        
        fprintf(fp, "%d\n", count);
    }

    // free allocated memory
    cudaFree(d_reads);
    cudaFree(d_read_results);
    cudaFree(d_reference_str);
    freeStringList(&queries);
    free(reference_str);

    return 0;
}
