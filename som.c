#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <ctype.h>
#include <complex.h>

#define INITIAL_CAPACITY 2048
#define MAX_UNIQUE_LABELS 100 
#define ALPHA 1.0
#define BETA 1.0
#define INITIAL_ETA 0.5
#define INITIAL_SIGMA 2.0
#define MIN_ETA 0.01
#define MIN_SIGMA 0.1
#define PI M_PI

typedef struct
{
    double *features;
    complex double *dft;
    char *label;
} Sample;

typedef struct
{
    double *means;
    double *std_devs;
    Sample *samples;
    int num_samples;
    int num_features;
    int capacity;
} Dataset;

double *g_means = NULL;
double *g_std_devs = NULL;
Dataset g_dataset;

typedef struct
{
    complex double *weights;
} Neuron;

Dataset load_csv(const char *filename);
void normalize_dataset(Dataset *dataset, double **means_out, double **std_devs_out);
void compute_dfts(Dataset *dataset);
void free_dataset(Dataset *dataset);
void init_som(Neuron ***grid, int grid_size, int num_features);
void free_som(Neuron ***grid, int grid_size);
double complex_distance(complex double *X, complex double *W, int n);
double neighborhood_function(double dist_sq, double radius_sq);
double get_grid_distance(int x1, int y1, int x2, int y2, int grid_size, int is_toroidal);
void train_som(Dataset *dataset, Neuron ***grid, int grid_size, int iterations, double initial_eta, double initial_sigma, int is_toroidal);
void calculate_u_matrix(Neuron ***grid, double ***u_matrix, int grid_size, int num_features, int is_toroidal);
void label_som_grid(Neuron ***grid, int ***label_grid, char ***unique_labels, int *num_unique_labels, int grid_size, int num_features);
void output_json(Neuron ***grid, double ***u_matrix, int ***label_grid, char **unique_labels, int num_unique_labels, int grid_size, int num_features);
int is_numeric(const char *s);
char *strip_quotes(char *str);

int is_numeric(const char *s) {
    if (s == NULL || *s == '\0' || isspace(*s)) return 0;
    char *p;
    strtod(s, &p);
    return *p == '\0';
}

char *strip_quotes(char *str) {
    if (str == NULL) return NULL;
    size_t len = strlen(str);
    if (len >= 2 && str[0] == '"' && str[len-1] == '"') {
        str[len-1] = '\0';
        memmove(str, str + 1, len - 1);
    }
    return str;
}

// --- CSV Loading ---
Dataset load_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File open failed");
        exit(EXIT_FAILURE);
    }

    char header_line[4096];
    char data_line[4096];

    if (fgets(header_line, sizeof(header_line), file) == NULL) {
        fprintf(stderr, "Error: Cannot read header from CSV file.\n");
        exit(EXIT_FAILURE);
    }
    if (fgets(data_line, sizeof(data_line), file) == NULL) {
        fprintf(stderr, "Error: Cannot read first data line from CSV file.\n");
        exit(EXIT_FAILURE);
    }

    int num_columns = 0;
    int label_column_index = -1;

    char *tmp = strdup(data_line);
    if (!tmp) { perror("strdup failed"); exit(EXIT_FAILURE); }
    char *token = strtok(tmp, ",\n");
    while (token) {
        if (!is_numeric(token) && label_column_index == -1) {
            label_column_index = num_columns;
            fprintf(stderr, "Auto-detected non-numeric label column at index: %d\n", label_column_index);
        }
        num_columns++;
        token = strtok(NULL, ",\n");
    }
    free(tmp);

    int num_features = (label_column_index != -1) ? num_columns - 1 : num_columns;
    if (num_features == 0) {
        fprintf(stderr, "Error: No numeric feature columns found in the dataset.\n");
        exit(EXIT_FAILURE);
    }

    fseek(file, (long)strlen(header_line), SEEK_SET);

    Dataset dataset = {0};
    dataset.num_features = num_features;
    dataset.capacity = INITIAL_CAPACITY;
    dataset.samples = malloc(dataset.capacity * sizeof(Sample));
    if (!dataset.samples) { perror("malloc failed"); exit(EXIT_FAILURE); }

    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        if (dataset.num_samples >= dataset.capacity) {
            dataset.capacity *= 2;
            dataset.samples = realloc(dataset.samples, dataset.capacity * sizeof(Sample));
            if (!dataset.samples) { perror("realloc failed"); exit(EXIT_FAILURE); }
        }

        Sample *sample = &dataset.samples[dataset.num_samples];
        sample->features = malloc(num_features * sizeof(double));
        if (!sample->features) { perror("malloc failed"); exit(EXIT_FAILURE); }
        sample->dft = NULL; // To be allocated later
        sample->label = NULL;

        int col = 0;
        int feature_idx = 0;
        char *line_copy = strdup(line);
        if (!line_copy) { perror("strdup failed"); exit(EXIT_FAILURE); }
        token = strtok(line_copy, ",\n");
        while (token && col < num_columns) {
            if (col == label_column_index) {
                sample->label = strdup(token);
                strip_quotes(sample->label);
                if (!sample->label) { perror("strdup failed"); exit(EXIT_FAILURE); }
            } else {
                if (feature_idx < num_features) {
                    char *endptr;
                    sample->features[feature_idx++] = strtod(token, &endptr);
                    if (endptr == token) sample->features[feature_idx - 1] = 0;
                }
            }
            col++;
            token = strtok(NULL, ",\n");
        }
        free(line_copy);
        dataset.num_samples++;
    }

    fclose(file);
    return dataset;
}

// --- Z-Score Normalization ---
void normalize_dataset(Dataset *dataset, double **means_out, double **std_devs_out) {
    if (dataset->num_samples == 0) return;

    double *means = calloc(dataset->num_features, sizeof(double));
    double *std_devs = calloc(dataset->num_features, sizeof(double));
    if (!means || !std_devs) { perror("calloc failed"); exit(EXIT_FAILURE); }

    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            means[j] += dataset->samples[i].features[j];
        }
    }
    for (int j = 0; j < dataset->num_features; j++) {
        means[j] /= dataset->num_samples;
    }

    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            double diff = dataset->samples[i].features[j] - means[j];
            std_devs[j] += diff * diff;
        }
    }
    for (int j = 0; j < dataset->num_features; j++) {
        std_devs[j] = sqrt(std_devs[j] / (dataset->num_samples - 1));
    }

    // Normalize
    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            if (std_devs[j] > 1e-9) {
                dataset->samples[i].features[j] = (dataset->samples[i].features[j] - means[j]) / std_devs[j];
            } else {
                dataset->samples[i].features[j] = 0.0;
            }
        }
    }

    *means_out = means;
    *std_devs_out = std_devs;
}

void compute_dfts(Dataset *dataset) {
    int N = dataset->num_features;
    for (int m = 0; m < dataset->num_samples; m++) {
        dataset->samples[m].dft = malloc(N * sizeof(complex double));
        if (!dataset->samples[m].dft) { perror("malloc failed"); exit(EXIT_FAILURE); }
        complex double *X = dataset->samples[m].dft;
        double *x = dataset->samples[m].features;
        for (int k = 0; k < N; k++) {
            X[k] = 0.0 + 0.0 * I;
            for (int n = 0; n < N; n++) {
                X[k] += x[n] * cexp(-I * 2 * PI * k * n / N);
            }
        }
    }
}

void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->num_samples; i++) {
        free(dataset->samples[i].features);
        if (dataset->samples[i].dft) free(dataset->samples[i].dft);
        if (dataset->samples[i].label) free(dataset->samples[i].label);
    }
    free(dataset->samples);
}

void init_som(Neuron ***grid, int grid_size, int num_features) {
    *grid = malloc(grid_size * sizeof(Neuron *));
    if (!*grid) { perror("malloc failed"); exit(EXIT_FAILURE); }
    srand(time(NULL));
    for (int i = 0; i < grid_size; i++) {
        (*grid)[i] = malloc(grid_size * sizeof(Neuron));
        if (!(*grid)[i]) { perror("malloc failed"); exit(EXIT_FAILURE); }
        for (int j = 0; j < grid_size; j++) {
            (*grid)[i][j].weights = malloc(num_features * sizeof(complex double));
            if (!(*grid)[i][j].weights) { perror("malloc failed"); exit(EXIT_FAILURE); }
            for (int f = 0; f < num_features; f++) {
                double mag = ((double)rand() / RAND_MAX) * 0.1;
                double phase = ((double)rand() / RAND_MAX) * 2 * PI - PI;
                (*grid)[i][j].weights[f] = mag * cexp(I * phase);
            }
        }
    }
}

void free_som(Neuron ***grid, int grid_size) {
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            free((*grid)[i][j].weights);
        }
        free((*grid)[i]);
    }
    free(*grid);
}

double complex_distance(complex double *X, complex double *W, int n) {
    double d_m = 0.0;
    double d_phi = 0.0;
    int count_phi = 0;
    for (int k = 0; k < n; k++) {
        double mag_x = cabs(X[k]);
        double mag_w = cabs(W[k]);
        d_m += pow(mag_x - mag_w, 2);
        if (mag_x > 1e-10 && mag_w > 1e-10) {
            double phase_diff = carg(X[k]) - carg(W[k]);
            d_phi += 1.0 - cos(phase_diff);
            count_phi++;
        }
    }
    d_m = sqrt(d_m);
    if (count_phi > 0) d_phi /= (double)count_phi;
    return ALPHA * d_m + BETA * d_phi;
}

double neighborhood_function(double dist_sq, double radius_sq) {
    return exp(-dist_sq / (2 * radius_sq));
}

double get_grid_distance(int x1, int y1, int x2, int y2, int grid_size, int is_toroidal) {
    double dx = fabs(x1 - x2);
    double dy = fabs(y1 - y2);
    if (is_toroidal) {
        dx = fmin(dx, grid_size - dx);
        dy = fmin(dy, grid_size - dy);
    }
    return dx * dx + dy * dy;
}

void train_som(Dataset *dataset, Neuron ***grid, int grid_size, int iterations, double initial_eta, double initial_sigma, int is_toroidal) {
    unsigned int master_seed = time(NULL);

    for (int iter = 0; iter < iterations; iter++) {
        if (iter % 1000 == 0) fprintf(stderr, "Iteration: %d / %d\n", iter, iterations);

        double eta = initial_eta * pow(MIN_ETA / initial_eta, (double)iter / iterations);
        double sigma = initial_sigma * pow(MIN_SIGMA / initial_sigma, (double)iter / iterations);
        double radius_sq = sigma * sigma;  // Note: in Gaussian, effective radius is sigma, but for cutoff, perhaps 3*sigma, but here use sigma^2 for consistency

        int global_bmu_x = 0, global_bmu_y = 0;
        double global_min_dist = INFINITY;
        complex double *sample_dft;

#pragma omp parallel
        {
#pragma omp single
            {
                int idx = rand_r(&master_seed) % dataset->num_samples;
                sample_dft = dataset->samples[idx].dft;
            }

            int local_bmu_x = 0, local_bmu_y = 0;
            double local_min_dist = INFINITY;

#pragma omp for collapse(2) nowait
            for (int i = 0; i < grid_size; i++) {
                for (int j = 0; j < grid_size; j++) {
                    double dist = complex_distance(sample_dft, (*grid)[i][j].weights, dataset->num_features);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_bmu_x = i;
                        local_bmu_y = j;
                    }
                }
            }

#pragma omp critical
            {
                if (local_min_dist < global_min_dist) {
                    global_min_dist = local_min_dist;
                    global_bmu_x = local_bmu_x;
                    global_bmu_y = local_bmu_y;
                }
            }

#pragma omp barrier

            int bmu_x = global_bmu_x;
            int bmu_y = global_bmu_y;

#pragma omp for collapse(2)
            for (int i = 0; i < grid_size; i++) {
                for (int j = 0; j < grid_size; j++) {
                    double dist_sq = get_grid_distance(bmu_x, bmu_y, i, j, grid_size, is_toroidal);
                    if (dist_sq <= radius_sq * 4) {  // Cutoff at 2*sigma for efficiency, since exp(-2) ~0.13, adjust if needed
                        double h = neighborhood_function(dist_sq, radius_sq);
                        for (int f = 0; f < dataset->num_features; f++) {
                            double mag_x = cabs(sample_dft[f]);
                            double mag_w = cabs((*grid)[i][j].weights[f]);
                            double new_mag = mag_w + eta * h * (mag_x - mag_w);

                            double theta_x = carg(sample_dft[f]);
                            double theta_w = carg((*grid)[i][j].weights[f]);
                            complex double weighted_avg = (1 - eta * h) * cexp(I * theta_w) + eta * h * cexp(I * theta_x);
                            double theta_new = carg(weighted_avg);

                            (*grid)[i][j].weights[f] = new_mag * cexp(I * theta_new);
                        }
                    }
                }
            }
        }
    }
}

void calculate_u_matrix(Neuron ***grid, double ***u_matrix, int grid_size, int num_features, int is_toroidal) {
    *u_matrix = malloc(grid_size * sizeof(double *));
    if (!*u_matrix) { perror("malloc failed"); exit(EXIT_FAILURE); }
    for (int i = 0; i < grid_size; i++) {
        (*u_matrix)[i] = calloc(grid_size, sizeof(double));
        if (!(*u_matrix)[i]) { perror("calloc failed"); exit(EXIT_FAILURE); }
    }

    double max_dist_val = 0.0;
    double min_dist_val = INFINITY;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            double max_neighbor_dist = 0.0;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di;
                    int nj = j + dj;
                    if (is_toroidal) {
                        ni = (ni + grid_size) % grid_size;
                        nj = (nj + grid_size) % grid_size;
                    } else if (ni < 0 || ni >= grid_size || nj < 0 || nj >= grid_size) {
                        continue;
                    }
                    double d = complex_distance((*grid)[i][j].weights, (*grid)[ni][nj].weights, num_features);
                    if (d > max_neighbor_dist) max_neighbor_dist = d;
                }
            }
            (*u_matrix)[i][j] = max_neighbor_dist;
#pragma omp critical
            {
                if ((*u_matrix)[i][j] > max_dist_val) max_dist_val = (*u_matrix)[i][j];
                if ((*u_matrix)[i][j] < min_dist_val) min_dist_val = (*u_matrix)[i][j];
            }
        }
    }

    double range = max_dist_val - min_dist_val;
    if (range < 1e-9) range = 1.0;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            (*u_matrix)[i][j] = ((*u_matrix)[i][j] - min_dist_val) / range;
        }
    }
}

void label_som_grid(Neuron ***grid, int ***label_grid, char ***unique_labels, int *num_unique_labels, int grid_size, int num_features) {
    int max_unique = MAX_UNIQUE_LABELS;
    *unique_labels = malloc(max_unique * sizeof(char *));
    if (!*unique_labels) { perror("malloc failed"); exit(EXIT_FAILURE); }
    *num_unique_labels = 0;

    for (int i = 0; i < g_dataset.num_samples; i++) {
        if (g_dataset.samples[i].label) {
            int found = 0;
            for (int j = 0; j < *num_unique_labels; j++) {
                if (strcmp(g_dataset.samples[i].label, (*unique_labels)[j]) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (*num_unique_labels >= max_unique) {
                    max_unique *= 2;
                    *unique_labels = realloc(*unique_labels, max_unique * sizeof(char *));
                    if (!*unique_labels) { perror("realloc failed"); exit(EXIT_FAILURE); }
                }
                (*unique_labels)[(*num_unique_labels)++] = strdup(g_dataset.samples[i].label);
            }
        }
    }

    if (*num_unique_labels == 0) return;

    long long total_cells = (long long)grid_size * grid_size * *num_unique_labels;
    if (total_cells > 100000000LL) {
        fprintf(stderr, "Skipping labeling due to high memory requirement (%lld cells)\n", total_cells);
        *num_unique_labels = 0;
        return;
    }

    int ***label_counts = malloc(grid_size * sizeof(int **));
    if (!label_counts) { perror("malloc failed"); exit(EXIT_FAILURE); }
    for (int i = 0; i < grid_size; i++) {
        label_counts[i] = malloc(grid_size * sizeof(int *));
        if (!label_counts[i]) { perror("malloc failed"); exit(EXIT_FAILURE); }
        for (int j = 0; j < grid_size; j++) {
            label_counts[i][j] = calloc(*num_unique_labels, sizeof(int));
            if (!label_counts[i][j]) { perror("calloc failed"); exit(EXIT_FAILURE); }
        }
    }

    // Assign samples to BMUs
#pragma omp parallel for
    for (int s = 0; s < g_dataset.num_samples; s++) {
        double min_dist = INFINITY;
        int bmu_x = 0, bmu_y = 0;
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                double d = complex_distance(g_dataset.samples[s].dft, (*grid)[i][j].weights, num_features);
                if (d < min_dist) {
                    min_dist = d;
                    bmu_x = i;
                    bmu_y = j;
                }
            }
        }
        if (g_dataset.samples[s].label) {
            for (int l = 0; l < *num_unique_labels; l++) {
                if (strcmp(g_dataset.samples[s].label, (*unique_labels)[l]) == 0) {
#pragma omp atomic
                    label_counts[bmu_x][bmu_y][l]++;
                    break;
                }
            }
        }
    }

    // Assign winning labels
#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int max_count = -1;
            int winning_label = -1;
            for (int l = 0; l < *num_unique_labels; l++) {
                if (label_counts[i][j][l] > max_count) {
                    max_count = label_counts[i][j][l];
                    winning_label = l;
                }
            }
            (*label_grid)[i][j] = winning_label;
        }
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) free(label_counts[i][j]);
        free(label_counts[i]);
    }
    free(label_counts);
}

// --- JSON Output ---
void output_json(Neuron ***grid, double ***u_matrix, int ***label_grid, char **unique_labels, int num_unique_labels, int grid_size, int num_features) {
    printf("{\n");
    printf("  \"grid_real\": [\n");
    for (int i = 0; i < grid_size; i++) {
        printf("    [\n");
        for (int j = 0; j < grid_size; j++) {
            printf("      [");
            for (int f = 0; f < num_features; f++) {
                printf("%.15f", creal((*grid)[i][j].weights[f]));
                if (f < num_features - 1) printf(",");
            }
            printf("]");
            if (j < grid_size - 1) printf(",");
            printf("\n");
        }
        printf("    ]");
        if (i < grid_size - 1) printf(",");
        printf("\n");
    }
    printf("  ],\n");

    printf("  \"grid_imag\": [\n");
    for (int i = 0; i < grid_size; i++) {
        printf("    [\n");
        for (int j = 0; j < grid_size; j++) {
            printf("      [");
            for (int f = 0; f < num_features; f++) {
                printf("%.15f", cimag((*grid)[i][j].weights[f]));
                if (f < num_features - 1) printf(",");
            }
            printf("]");
            if (j < grid_size - 1) printf(",");
            printf("\n");
        }
        printf("    ]");
        if (i < grid_size - 1) printf(",");
        printf("\n");
    }
    printf("  ],\n");

    printf("  \"u_matrix\": [\n");
    for (int i = 0; i < grid_size; i++) {
        printf("    [");
        for (int j = 0; j < grid_size; j++) {
            printf("%.15f", (*u_matrix)[i][j]);
            if (j < grid_size - 1) printf(",");
        }
        printf("]");
        if (i < grid_size - 1) printf(",");
        printf("\n");
    }
    printf("  ],\n");

    printf("  \"label_grid\": [\n");
    for (int i = 0; i < grid_size; i++) {
        printf("    [");
        for (int j = 0; j < grid_size; j++) {
            printf("%d", (*label_grid)[i][j]);
            if (j < grid_size - 1) printf(",");
        }
        printf("]");
        if (i < grid_size - 1) printf(",");
        printf("\n");
    }
    printf("  ],\n");

    printf("  \"unique_labels\": [");
    for (int l = 0; l < num_unique_labels; l++) {
        printf("\"");
        for (char *c = unique_labels[l]; *c; c++) {
            if (*c == '"') printf("\\\"");
            else printf("%c", *c);
        }
        printf("\"");
        if (l < num_unique_labels - 1) printf(",");
    }
    printf("],\n");
    printf("  \"grid_size\": %d,\n", grid_size);
    printf("  \"num_features\": %d,\n", num_features);

    // Means
    printf("  \"means\": [");
    for (int f = 0; f < num_features; f++) {
        printf("%.15f", g_means[f]);
        if (f < num_features - 1) printf(",");
    }
    printf("],\n");

    // Std devs
    printf("  \"std_devs\": [");
    for (int f = 0; f < num_features; f++) {
        printf("%.15f", g_std_devs[f]);
        if (f < num_features - 1) printf(",");
    }
    printf("]\n");

    printf("}\n");
}

int main(int argc, char *argv[]) {
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <csv> <grid_size> <iterations> <initial_eta> <initial_sigma> <is_toroidal> <calculate_labels>\n", argv[0]);
        fprintf(stderr, "is_toroidal and calculate_labels: 0 or 1\n");
        return 1;
    }

    const char *filename = argv[1];
    int grid_size = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    double initial_eta = atof(argv[4]);
    double initial_sigma = atof(argv[5]);
    int is_toroidal = atoi(argv[6]);
    int calculate_labels = atoi(argv[7]);

    g_dataset = load_csv(filename);
    normalize_dataset(&g_dataset, &g_means, &g_std_devs);
    compute_dfts(&g_dataset);

    Neuron **grid = NULL;
    init_som(&grid, grid_size, g_dataset.num_features);

    train_som(&g_dataset, &grid, grid_size, iterations, initial_eta, initial_sigma, is_toroidal);

    double **u_matrix = NULL;
    calculate_u_matrix(&grid, &u_matrix, grid_size, g_dataset.num_features, is_toroidal);

    int **label_grid = malloc(grid_size * sizeof(int *));
    if (!label_grid) { perror("malloc failed"); exit(EXIT_FAILURE); }
    for (int i = 0; i < grid_size; i++) {
        label_grid[i] = malloc(grid_size * sizeof(int));
        if (!label_grid[i]) { perror("malloc failed"); exit(EXIT_FAILURE); }
        for (int j = 0; j < grid_size; j++) label_grid[i][j] = -1;
    }
    char **unique_labels = NULL;
    int num_unique_labels = 0;
    if (calculate_labels) {
        label_som_grid(&grid, &label_grid, &unique_labels, &num_unique_labels, grid_size, g_dataset.num_features);
    }

    output_json(&grid, &u_matrix, &label_grid, unique_labels, num_unique_labels, grid_size, g_dataset.num_features);

    for (int i = 0; i < grid_size; i++) {
        free(u_matrix[i]);
        free(label_grid[i]);
    }
    free(u_matrix);
    free(label_grid);
    for (int i = 0; i < num_unique_labels; i++) free(unique_labels[i]);
    free(unique_labels);
    free_som(&grid, grid_size);
    free_dataset(&g_dataset);
    free(g_means);
    free(g_std_devs);

    return 0;
}

