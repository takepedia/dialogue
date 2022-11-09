#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void die(char *log);
void FFT(double *xr, double *xi, double *Xr, double *Xi, int N);
void IFFT(double *Xr, double *Xi, double *xr, double *xi, int N);
long now();

#define SAMPLING_RATE 16000
#define NANO_SEC      1000000000L
#define DIR_ASSET     "../asset/speech_sample/"
#define DIR_OUTPUT    "./output/"
#define EXT_DAT       ".dat"
#define SUF_FFT       "_fft"

/**
 * @brief プログラムを終了させる
 * 
 * @param log error に出す項目
 */
void die(char *log) {
  perror(log);
  exit(EXIT_FAILURE);
}

/**
 * @brief 高速 Fourier 変換
 * 
 * @param xr  時間空間実部
 * @param xi  時間空間虚部
 * @param Xr  周波数空間実部
 * @param Xi  周波数空間虚部
 * @param N   データのサイズ
 */
void FFT(double *xr, double *xi, double *Xr, double *Xi, int N) {
  static double *rbuf, *ibuf;
  static int bufsize = 0;

  /* memory allocation for buffers */
  if (bufsize != N) {
    bufsize = N;
    rbuf = (double *) malloc(bufsize * sizeof(double));
    ibuf = (double *) malloc(bufsize * sizeof(double));
    for (int i = 0; i < bufsize; ++i) {
      rbuf[i] = 0.0;
      ibuf[i] = 0.0;
    }
  }

  /* bit reverse of xr[] & xi[] --> store to rbuf[] and ibuf[] */
  int i = 0;
  int j = 0;
  rbuf[0] = xr[0];
  ibuf[0] = xi[0];
  for (j = 1; j < N - 1; ++j) {
    int k = 0;
    for (k = N / 2; k <= i; k /= 2) {
      i -= k;
    }
    i += k;
    rbuf[j] = xr[i];
    ibuf[j] = xi[i];
  }
  rbuf[j] = xr[j];
  ibuf[j] = xi[j];

  /* butterfly calculation */
  double theta = -2.0 * M_PI;
  for (int n = 1; n * 2 <= N; n *= 2) {
    theta *= 0.5;
    for (int i = 0; i < n; i++) {
      double wr = cos(theta * i);
      double wi = sin(theta * i);
      for (j = i; j < N; j += n * 2) {
        int k = j + n;
        /**
         * Re{W * buf[k]} = wr * rbuf[k] - wi * ibuf[k]
         * Im{W * buf[k]} = wr * ibuf[k] + wi * rbuf[k]
         **/
        Xr[j] = 1 * rbuf[j] + wr * rbuf[k] - wi * ibuf[k];
        Xi[j] = 1 * ibuf[j] + wr * ibuf[k] + wi * rbuf[k];
        Xr[k] = 1 * rbuf[j] - wr * rbuf[k] + wi * ibuf[k];
        Xi[k] = 1 * ibuf[j] - wr * ibuf[k] - wi * rbuf[k];
      }
    }
    for (i = 0; i < N; i++) {
      rbuf[i] = Xr[i];
      ibuf[i] = Xi[i];
    }
  }
  return;
}

/**
 * @brief 逆高速 Fourier 変換
 * 
 * @param Xr  周波数空間実部
 * @param Xi  周波数空間虚部
 * @param xr  時間空間実部
 * @param xi  時間空間虚部
 * @param N   データのサイズ
 */
void IFFT(double *Xr, double *Xi, double *xr, double *xi, int N) {
  int i, j, k, n, n2;
  double theta, wr, wi;

  static double *rbuf, *ibuf;
  static int bufsize = 0;

  /* memory allocation for buffers */
  if (bufsize != N) {
    bufsize = N;
    rbuf = (double *) calloc(sizeof(double), bufsize);
    ibuf = (double *) calloc(sizeof(double), bufsize);
  }

  /* bit reverse of Xr[] & Xi[] --> store to rbuf[] and ibuf[] */
  i = j = 0;
  rbuf[j] = Xr[j] / N;
  ibuf[j] = Xi[j] / N;
  for (j = 1; j < N - 1; j++) {
    for (k = N / 2; k <= i; k /= 2) i -= k;
    i += k;
    rbuf[j] = Xr[i] / N;
    ibuf[j] = Xi[i] / N;
  }
  rbuf[j] = Xr[j] / N;
  ibuf[j] = Xi[j] / N;

  /* butterfly calculation */
  theta = 2.0 * M_PI; /* not -2.0*M_PI !!! */
  for (n = 1; (n2 = n * 2) <= N; n = n2) {
    theta *= 0.5;
    for (i = 0; i < n; i++) {
      wr = cos(theta * i);
      wi = sin(theta * i);
      for (j = i; j < N; j += n2) {
        k = j + n;
        xr[j] = rbuf[j] + wr * rbuf[k] - wi * ibuf[k];
        xi[j] = ibuf[j] + wr * ibuf[k] + wi * rbuf[k];
        xr[k] = rbuf[j] - wr * rbuf[k] + wi * ibuf[k];
        xi[k] = ibuf[j] - wr * ibuf[k] - wi * rbuf[k];
      }
    }

    for (i = 0; i < N; i++) {
      rbuf[i] = xr[i];
      ibuf[i] = xi[i];
    }
  }
  return;
}

/**
 * @brief 現在時刻をナノ秒単位で返す
 * 
 * @return long 現在時刻（ナノ秒）
 */
long now() {
  struct timespec current_ts[1];
  clock_gettime(CLOCK_REALTIME, current_ts);
  return current_ts->tv_sec * NANO_SEC + current_ts->tv_nsec;
}

int main(int argc, char **argv) {
  /* check the format of input */
  if (argc != 4) {
    fprintf(stderr, "Usage: %s DATfile skip[sample] frame length[sample]\n",
            argv[0]);
    exit(EXIT_FAILURE);
  }
 
  char *filename_input = (char *) malloc(100 * sizeof(char));
  char *filename_output = (char *) malloc(100 * sizeof(char));
  sprintf(filename_input, "%s%s%s", DIR_ASSET, argv[1], EXT_DAT);
  sprintf(filename_output, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_FFT, EXT_DAT);

  FILE *fp_input = fopen(filename_input, "r");
  FILE *fp_output = fopen(filename_output, "w");
  int nskip = atoi(argv[2]);
  int framelen = atoi(argv[3]);

  /* check the validity of input */
  if (fp_input == NULL || fp_output == NULL) {
    die("open");
  }
  if (nskip < 0) {
    fprintf(stderr, "# of skip must be positive\n");
    exit(EXIT_FAILURE);
  }
  if (framelen < 0) {
    fprintf(stderr, "frame length must be positive\n");
    exit(EXIT_FAILURE);
  }

  /* memory allocation & initilization */
  short *sdata = (short *) malloc(framelen * sizeof(short));
  double *xr = (double *) malloc(framelen * sizeof(double));
  double *xi = (double *) malloc(framelen * sizeof(double));
  double *Xr = (double *) malloc(framelen * sizeof(double));
  double *Xi = (double *) malloc(framelen * sizeof(double));
  for (int i = 0; i < framelen; ++i) {
    sdata[i] = 0.0;
    xr[i] = 0.0;
    xi[i] = 0.0;
    Xr[i] = 0.0;
    Xi[i] = 0.0;
  }
  if (sdata == NULL || xr == NULL || xi == NULL || Xr == NULL || Xi == NULL) {
    exit(1);
  }

  fseek(fp_input, nskip * sizeof(short), SEEK_SET);
  fread(sdata, sizeof(short), framelen, fp_input);

  /* windowing */
  for (i = 0; i < framelen; i++) {
    xr[i] = (0.54 - 0.46 * cos(2 * M_PI * i / (framelen - 1))) * sdata[i];
    xi[i] = 0.0;
  }

/* FFT */
#ifdef STRESSTEST
  int repeat = 1000;
  long time_begin = now();
  for (int i = 0; i < repeat; ++i) {
    FFT(xr, xi, Xr, Xi, framelen);
  }
  long time_end = now();
  printf("FFT: %ld nsec / proc\n", (time_end - time_begin) / repeat);
#else
  FFT(xr, xi, Xr, Xi, framelen);
#endif

/* plot the result */
#ifndef INVERSE
  for (int i = 0; i < framelen; i++) {
    double spec = log10((Xr[i] * Xr[i] + Xi[i] * Xi[i]) / framelen);
    fprintf(fp_output, "%d %lf\n", i * SAMPLING_RATE / framelen, spec);
  }

#else
  /* IFFT trial */
  IFFT(Xr, Xi, xr, xi, framelen);

  for (int i = 0; i < framelen; i++) printf("%d %lf\n", i, xr[i]);
#endif

  fclose(fp_input);
  fclose(fp_output);
  free(sdata);
  free(xr);
  free(xi);
  free(Xr);
  free(Xi);
  return EXIT_SUCCESS;
}