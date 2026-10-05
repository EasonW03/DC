#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#define N 64
#define CP 16
#define LENGTH (N + CP)
#ifndef TOTAL_BITS
#define TOTAL_BITS (1UL << 25)
#endif
#define OFFSET 23
#define SYNC_SYMBOLS 256

static const double PI = 3.14159265358979323846;
static const int LEVEL[4] = {-3, -1, 3, 1};

void fft(double complex x[], int n, int inverse);
uint32_t random_word(uint64_t *state);
double uniform(uint64_t *state);
double complex noise(uint64_t *state, double sigma);
double complex map(const unsigned char *bits, int order);
void demap(double complex value, unsigned char *bits, int order);
FILE *open_output(const char *name);
int synchronize(const double complex *rx, int count, double *metric);
void simulate(int order, FILE *ber);
void self_test(void);

int main(void)
{
    FILE *ber;

    self_test();
    ber = open_output("ber.csv");
    fprintf(ber, "modulation,snr_db,bits,errors,ber,theory,cp_start,fft_start,signal_power,noise_power\n");
    simulate(2, ber);
    simulate(4, ber);
    if (fclose(ber) != 0) {
        perror("ber.csv");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/* PCG-XSH-RR: explicit unsigned arithmetic makes the sequence reproducible. */
uint32_t random_word(uint64_t *state)
{
    uint64_t old;
    uint32_t value, rotation;

    old = *state;
    *state = old * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    value = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    rotation = (uint32_t)(old >> 59u);
    return (value >> rotation) | (value << ((0u - rotation) & 31u));
}

double uniform(uint64_t *state)
{
    return ((double)random_word(state) + 0.5) / 4294967296.0;
}

/* Box-Muller: independent N(0, sigma^2) real and imaginary components. */
double complex noise(uint64_t *state, double sigma)
{
    double radius, angle;

    radius = sigma * sqrt(-2.0 * log(uniform(state)));
    angle = 2.0 * PI * uniform(state);
    return radius * (cos(angle) + I * sin(angle));
}

double complex map(const unsigned char *bits, int order)
{
    if (order == 2) {
        return ((1.0 - 2.0 * bits[0]) + I * (1.0 - 2.0 * bits[1])) / sqrt(2.0);
    }
    return (LEVEL[2 * bits[0] + bits[1]] + I * LEVEL[2 * bits[2] + bits[3]]) / sqrt(10.0);
}

void demap(double complex value, unsigned char *bits, int order)
{
    double real, imag;

    real = creal(value);
    imag = cimag(value);
    if (order == 2) {
        bits[0] = real < 0.0;
        bits[1] = imag < 0.0;
    } else {
        bits[0] = real >= 0.0;
        bits[1] = fabs(real) < 2.0 / sqrt(10.0);
        bits[2] = imag >= 0.0;
        bits[3] = fabs(imag) < 2.0 / sqrt(10.0);
    }
}

FILE *open_output(const char *name)
{
    FILE *file;

    file = fopen(name, "w");
    if (file == NULL) {
        perror(name);
        exit(EXIT_FAILURE);
    }
    return file;
}

/* Search one complete symbol period. Receiver never reads OFFSET. */
int synchronize(const double complex *rx, int count, double *metric)
{
    int d, m, i, best;
    size_t index;
    double complex correlation;
    double energy_a, energy_b;

    best = 0;
    for (d = 0; d < LENGTH; d++) {
        correlation = 0.0;
        energy_a = 0.0;
        energy_b = 0.0;
        for (m = 0; m < count; m++) {
            for (i = 0; i < CP; i++) {
                index = (size_t)m * LENGTH + d + i;
                correlation += conj(rx[index]) * rx[index + N];
                energy_a += creal(rx[index] * conj(rx[index]));
                energy_b += creal(rx[index + N] * conj(rx[index + N]));
            }
        }
        metric[d] = pow(cabs(correlation), 2.0) / (energy_a * energy_b + 1e-300);
        if (metric[d] > metric[best]) {
            best = d;
        }
    }
    return best;
}

void simulate(int order, FILE *ber)
{
    unsigned char *bits, decoded[4];
    double complex *rx, block[N], sample, perturbation;
    double metric[LENGTH], gamma, sigma, signal_energy, noise_energy, theory, a;
    uint64_t bit_state, noise_state;
    size_t i, m, k, b, symbols, samples, index, errors;
    int snr, start, count, trial, counts[5];
    char filename[100];
    const char *name;
    FILE *wave, *constellation, *sync;

    name = order == 2 ? "QPSK" : "16QAM";
    symbols = TOTAL_BITS / (N * order);
    samples = symbols * LENGTH + 2 * LENGTH;
    if (TOTAL_BITS % (N * order) != 0 || symbols <= SYNC_SYMBOLS) {
        fprintf(stderr, "TOTAL_BITS must contain more than 256 complete OFDM symbols.\n");
        exit(EXIT_FAILURE);
    }
    bits = malloc(TOTAL_BITS * sizeof(*bits));
    rx = malloc(samples * sizeof(*rx));
    if (bits == NULL || rx == NULL) {
        fprintf(stderr, "Cannot allocate simulation buffers.\n");
        exit(EXIT_FAILURE);
    }
    bit_state = UINT64_C(10000);
    for (i = 0; i < TOTAL_BITS; i++) {
        bits[i] = (unsigned char)(random_word(&bit_state) >> 31u);
    }
    counts[0] = 1;
    counts[1] = 4;
    counts[2] = 16;
    counts[3] = 64;
    counts[4] = SYNC_SYMBOLS;

    for (snr = 0; snr <= 24; snr += 3) {
        gamma = pow(10.0, snr / 10.0);
        sigma = sqrt(1.0 / (2.0 * N * gamma));
        noise_state = UINT64_C(987654321) + (uint64_t)(snr * 100 + order);
        wave = NULL;
        constellation = NULL;
        if (snr == 3 || snr == 15) {
            snprintf(filename, sizeof(filename), "wave_%s_%02d.csv", name, snr);
            wave = open_output(filename);
            fprintf(wave, "sample,tx_re,tx_im,rx_re,rx_im\n");
            snprintf(filename, sizeof(filename), "constellation_%s_%02d.csv", name, snr);
            constellation = open_output(filename);
            fprintf(constellation, "symbol,carrier,tx_re,tx_im,rx_re,rx_im\n");
        }
        for (i = 0; i < samples; i++) {
            rx[i] = 0.0;
        }
        for (i = 0; i < OFFSET; i++) {
            rx[i] = noise(&noise_state, sigma);
        }
        signal_energy = 0.0;
        noise_energy = 0.0;
        for (m = 0; m < symbols; m++) {
            for (k = 0; k < N; k++) {
                block[k] = map(bits + (m * N + k) * order, order);
            }
            fft(block, N, 1);
            for (k = 0; k < LENGTH; k++) {
                sample = block[(k + N - CP) % N];
                perturbation = noise(&noise_state, sigma);
                index = OFFSET + m * LENGTH + k;
                rx[index] = sample + perturbation;
                signal_energy += creal(sample * conj(sample));
                noise_energy += creal(perturbation * conj(perturbation));
                if (wave != NULL && m < 3) {
                    fprintf(wave, "%zu,%.17g,%.17g,%.17g,%.17g\n", m * LENGTH + k,
                        creal(sample), cimag(sample), creal(rx[index]), cimag(rx[index]));
                }
            }
        }
        for (i = OFFSET + symbols * LENGTH; i < samples; i++) {
            rx[i] = noise(&noise_state, sigma);
        }
        snprintf(filename, sizeof(filename), "sync_%s_%02d.csv", name, snr);
        sync = open_output(filename);
        fprintf(sync, "averaged_symbols,candidate,metric,detected_cp\n");
        start = 0;
        for (trial = 0; trial < 5; trial++) {
            count = counts[trial];
            start = synchronize(rx, count, metric);
            for (i = 0; i < LENGTH; i++) {
                fprintf(sync, "%d,%zu,%.17g,%d\n", count, i, metric[i], start);
            }
        }
        fclose(sync);
        errors = 0;
        for (m = 0; m < symbols; m++) {
            for (k = 0; k < N; k++) {
                block[k] = rx[start + CP + m * LENGTH + k];
            }
            fft(block, N, 0);
            for (k = 0; k < N; k++) {
                index = (m * N + k) * order;
                demap(block[k], decoded, order);
                for (b = 0; b < (size_t)order; b++) {
                    errors += decoded[b] != bits[index + b];
                }
                if (constellation != NULL && m < 3) {
                    sample = map(bits + index, order);
                    fprintf(constellation, "%zu,%zu,%.17g,%.17g,%.17g,%.17g\n",
                        m, k, creal(sample), cimag(sample), creal(block[k]), cimag(block[k]));
                }
            }
        }
        if (wave != NULL) {
            fclose(wave);
            fclose(constellation);
        }
        a = sqrt(gamma / 5.0);
        theory = order == 2 ? 0.5 * erfc(sqrt(gamma / 2.0)) :
            0.375 * erfc(a / sqrt(2.0)) + 0.25 * erfc(3.0 * a / sqrt(2.0)) - 0.125 * erfc(5.0 * a / sqrt(2.0));
        fprintf(ber, "%s,%d,%lu,%zu,%.17g,%.17g,%d,%d,%.17g,%.17g\n",
            name, snr, (unsigned long)TOTAL_BITS, errors, (double)errors / TOTAL_BITS, theory,
            start, start + CP, signal_energy / (symbols * LENGTH), noise_energy / (symbols * LENGTH));
        fflush(ber);
        printf("%s %2d dB: %zu / %lu errors, BER %.6g, detected CP %d (true %d)\n",
            name, snr, errors, (unsigned long)TOTAL_BITS, (double)errors / TOTAL_BITS, start, OFFSET);
        fflush(stdout);
    }
    free(rx);
    free(bits);
}

void self_test(void)
{
    double complex original[N], block[N], rx[4 * LENGTH];
    double metric[LENGTH], error;
    unsigned char bits[4], result[4];
    int order, value, i, k, d, detected;

    error = 0.0;
    for (order = 2; order <= 4; order += 2) {
        for (value = 0; value < (1 << order); value++) {
            for (i = 0; i < order; i++) {
                bits[i] = (value >> i) & 1;
            }
            demap(map(bits, order), result, order);
            if (memcmp(bits, result, (size_t)order) != 0) {
                fprintf(stderr, "Mapping self-test failed.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    for (k = 0; k < N; k++) {
        original[k] = cos(k * 1.23) + I * sin(k * 0.71);
        block[k] = original[k];
    }
    fft(block, N, 1);
    for (d = 0; d < LENGTH; d++) {
        memset(rx, 0, sizeof(rx));
        for (i = 0; i < 3 * LENGTH; i++) {
            /* Reset the sample index at each OFDM boundary. */
            rx[d + i] = block[((i % LENGTH) + N - CP) % N];
        }
        detected = synchronize(rx, 2, metric);
        if (detected != d) {
            fprintf(stderr, "Timing self-test failed at %d: detected %d.\n", d, detected);
            exit(EXIT_FAILURE);
        }
    }
    fft(block, N, 0);
    for (k = 0; k < N; k++) {
        error = fmax(error, cabs(block[k] - original[k]));
    }
    if (error > 1e-10) {
        fprintf(stderr, "FFT round-trip failed.\n");
        exit(EXIT_FAILURE);
    }
    printf("Self-tests passed: all mapping labels, 80 noiseless timing offsets, FFT error %.3g\n", error);
}

void fft(double complex x[], int n, int inverse)
{
    int bit;
    double complex temp;
    double angle;
    double complex step;
    double complex w, a, b;
    int i, j, k;


    // change sample to bit-reversed order
    j = 0;
    for (i = 1; i < n; i++) {
        bit = n / 2;
        while (j & bit) {
            j ^= bit;
            bit /= 2;
        }
        j ^= bit;
        if (i < j) {
            temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }

    // butterfly stages
    for (i = 2; i <= n; i *= 2) {
        angle = ((inverse ? 2.0 : -2.0) * PI) / i;
        step = cos(angle) + I * sin(angle);

        for (j = 0; j < n; j += i) {
            w = 1.0;
            for (k = 0; k < (i / 2); k++) {
                a = x[j + k];
                b = w * x[j + k + (i / 2)];
                x[j + k] = a + b;
                x[j + k + (i / 2)] = a - b;
                w *= step;
            }
        }
    }

    if (inverse) {
        for (i = 0; i < n; i++) {
            x[i] /= n;
        }
    }
}

