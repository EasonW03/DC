#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define N 64
#define CP 16
#define LENGTH (N + CP)
#ifndef TOTAL_BITS
#define TOTAL_BITS 33554432
#endif
#define OFFSET 23
#define SYNC_SYMBOLS 256

static const double PI = 3.14159265358979323846;
static const int LEVEL[4] = {-3, -1, 3, 1};

void fft(double complex x[], int n, int inverse);
double complex noise(double sigma);
double complex map(const int *bits, int bits_per_symbol);
void demap(double complex value, int *bits, int bits_per_symbol);
FILE *open_output(const char *name);
int synchronize(const double complex *rx, int count, double *metric);
void simulate(int bits_per_symbol, FILE *ber);

int main(void)
{
    FILE *ber;

    ber = open_output("ber.csv");
    fprintf(ber, "modulation,snr_db,bits,errors,ber,theory,cp_start,fft_start,signal_power,noise_power\n");
    simulate(2, ber);  /* Required: QPSK, 2 bits per symbol. */
    simulate(4, ber);  /* Bonus: 16-QAM, 4 bits per symbol. */
    if (fclose(ber) != 0) {
        perror("ber.csv");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/* rand() is uniform; Box-Muller converts two draws into Gaussian I/Q noise. */
double complex noise(double sigma)
{
    double u1, u2, radius, angle;

    /* Keep u1 strictly between 0 and 1 so log(u1) is defined. */
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    radius = sigma * sqrt(-2.0 * log(u1));
    angle = 2.0 * PI * u2;
    return radius * (cos(angle) + I * sin(angle));
}

double complex map(const int *bits, int bits_per_symbol)
{
    if (bits_per_symbol == 2) {
        return ((1.0 - 2.0 * bits[0]) + I * (1.0 - 2.0 * bits[1])) / sqrt(2.0);
    }
    return (LEVEL[2 * bits[0] + bits[1]] + I * LEVEL[2 * bits[2] + bits[3]]) / sqrt(10.0);
}

void demap(double complex value, int *bits, int bits_per_symbol)
{
    double real, imag;

    real = creal(value);
    imag = cimag(value);
    if (bits_per_symbol == 2) {
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

// Search one complete symbol period. Receiver never reads OFFSET.
int synchronize(const double complex *rx, int count, double *metric)
{
    int d, m, i, best;
    int index;
    double complex correlation;
    double energy_a, energy_b;

    best = 0;
    for (d = 0; d < LENGTH; d++) {
        correlation = 0.0;
        energy_a = 0.0;
        energy_b = 0.0;
        for (m = 0; m < count; m++) {
            for (i = 0; i < CP; i++) {
                index = m * LENGTH + d + i;
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

void simulate(int bits_per_symbol, FILE *ber)
{
    int *bits, decoded[4];
    double complex *rx, block[N], sample, perturbation;
    double metric[LENGTH], gamma, sigma, signal_energy, noise_energy, theory, a;
    int i, m, k, b, symbols, samples, index, errors;
    int snr, start, count, trial, counts[5];
    char filename[100];
    const char *name;
    FILE *wave, *constellation, *sync;

    name = bits_per_symbol == 2 ? "QPSK" : "16QAM";
    symbols = TOTAL_BITS / (N * bits_per_symbol);
    /* One extra period covers the leading delay and the last FFT window. */
    samples = symbols * LENGTH + LENGTH;
    if (TOTAL_BITS <= 0 || TOTAL_BITS % (N * bits_per_symbol) != 0 ||
        symbols <= SYNC_SYMBOLS || symbols <= 64 || SYNC_SYMBOLS <= 0 ||
        OFFSET < 0 || OFFSET >= LENGTH) {
        fprintf(stderr, "Check TOTAL_BITS, SYNC_SYMBOLS, and OFFSET settings.\n");
        exit(EXIT_FAILURE);
    }
    bits = malloc(TOTAL_BITS * sizeof(*bits));
    rx = malloc(samples * sizeof(*rx));
    if (bits == NULL || rx == NULL) {
        fprintf(stderr, "Cannot allocate simulation buffers.\n");
        exit(EXIT_FAILURE);
    }
    /* Reuse the same input bits at every SNR. */
    srand(10000);
    for (i = 0; i < TOTAL_BITS; i++) {
        bits[i] = rand() % 2;
    }
    counts[0] = 1;
    counts[1] = 4;
    counts[2] = 16;
    counts[3] = 64;
    counts[4] = SYNC_SYMBOLS;

    for (snr = 0; snr <= 24; snr += 3) {
        gamma = pow(10.0, snr / 10.0);
        sigma = sqrt(1.0 / (2.0 * N * gamma));
        srand(20000 + snr * 100 + bits_per_symbol);
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
        for (i = 0; i < OFFSET; i++) {
            rx[i] = noise(sigma);
        }
        signal_energy = 0.0;
        noise_energy = 0.0;
        /* Map bits, perform IFFT, add CP, and pass samples through AWGN. */
        for (m = 0; m < symbols; m++) {
            for (k = 0; k < N; k++) {
                block[k] = map(bits + (m * N + k) * bits_per_symbol, bits_per_symbol);
            }
            fft(block, N, 1);
            for (k = 0; k < LENGTH; k++) {
                if (k < CP) {
                    sample = block[N - CP + k];  /* Copy the last 16 samples. */
                } else {
                    sample = block[k - CP];     /* Then send all 64 samples. */
                }
                perturbation = noise(sigma);
                index = OFFSET + m * LENGTH + k;
                rx[index] = sample + perturbation;
                signal_energy += creal(sample * conj(sample));
                noise_energy += creal(perturbation * conj(perturbation));
                if (wave != NULL && m < 3) {
                    fprintf(wave, "%d,%.17g,%.17g,%.17g,%.17g\n", m * LENGTH + k,
                        creal(sample), cimag(sample), creal(rx[index]), cimag(rx[index]));
                }
            }
        }
        for (i = OFFSET + symbols * LENGTH; i < samples; i++) {
            rx[i] = noise(sigma);
        }
        snprintf(filename, sizeof(filename), "sync_%s_%02d.csv", name, snr);
        sync = open_output(filename);
        fprintf(sync, "averaged_symbols,candidate,metric,detected_cp\n");
        start = 0;
        for (trial = 0; trial < 5; trial++) {
            count = counts[trial];
            start = synchronize(rx, count, metric);
            for (i = 0; i < LENGTH; i++) {
                fprintf(sync, "%d,%d,%.17g,%d\n", count, i, metric[i], start);
            }
        }
        fclose(sync);
        /* Use the detected boundary, remove CP, FFT, and compare bits. */
        errors = 0;
        for (m = 0; m < symbols; m++) {
            for (k = 0; k < N; k++) {
                block[k] = rx[start + CP + m * LENGTH + k];
            }
            fft(block, N, 0);
            for (k = 0; k < N; k++) {
                index = (m * N + k) * bits_per_symbol;
                demap(block[k], decoded, bits_per_symbol);
                for (b = 0; b < bits_per_symbol; b++) {
                    errors += decoded[b] != bits[index + b];
                }
                if (constellation != NULL && m < 3) {
                    sample = map(bits + index, bits_per_symbol);
                    fprintf(constellation, "%d,%d,%.17g,%.17g,%.17g,%.17g\n",
                        m, k, creal(sample), cimag(sample), creal(block[k]), cimag(block[k]));
                }
            }
        }
        if (wave != NULL) {
            fclose(wave);
            fclose(constellation);
        }
        /* Reference BER for the report; not used by the receiver. */
        if (bits_per_symbol == 2) {
            theory = 0.5 * erfc(sqrt(gamma / 2.0));
        } else {
            a = sqrt(gamma / 10.0);
            theory = 0.375 * erfc(a) + 0.25 * erfc(3.0 * a) - 0.125 * erfc(5.0 * a);
        }
        fprintf(ber, "%s,%d,%d,%d,%.17g,%.17g,%d,%d,%.17g,%.17g\n",
            name, snr, TOTAL_BITS, errors, (double)errors / TOTAL_BITS, theory,
            start, start + CP, signal_energy / (symbols * LENGTH), noise_energy / (symbols * LENGTH));
        fflush(ber);
        printf("%s %2d dB: %d / %d errors, BER %.6g, detected CP %d (true %d)\n",
            name, snr, errors, TOTAL_BITS, (double)errors / TOTAL_BITS, start, OFFSET);
        fflush(stdout);
    }
    free(rx);
    free(bits);
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
