#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void die(char *log);
void DFT(double *xr, double *xi, double *Xr, double *Xi, int N);
void IDFT(double *Xr, double *Xi, double *xr, double *xi, int N);
long now();

#define SAMPLING_RATE 16000
#define NANO_SEC      1000000000L
#define DIR_ASSET     "../asset/speech_sample/"
#define DIR_OUTPUT    "./output/"
#define EXT_DAT       ".dat"
#define SUF_DFT       "_dft"

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
 * @brief 離散 Fourier 変換
 * 
 * @param xr  時間空間実部
 * @param xi  時間空間虚部
 * @param Xr  周波数空間実部
 * @param Xi  周波数空間虚部
 * @param N   データのサイズ
 */
void DFT(double *xr, double *xi, double *Xr, double *Xi, int N) {
  for (int k = 0; k < N; ++k) {
    Xr[k] = 0.0;
    Xi[k] = 0.0;
  }
  for (int k = 0; k < N; ++k) {
    for (int n = 0; n < N; ++n) {
      double wr = cos(2.0 * M_PI * n * k / N);
      double wi = sin(2.0 * M_PI * n * k / N);
      Xr[k] += xr[n] * wr - xi[n] * wi;
      Xi[k] += xr[n] * wi + xi[n] * wr;
    }
  }
  return;
}

/**
 * @brief 逆離散 Fourier 変換
 * 
 * @param Xr  周波数空間実部
 * @param Xi  周波数空間虚部
 * @param xr  時間空間実部
 * @param xi  時間空間虚部
 * @param N   データのサイズ
 */
void IDFT(double *Xr, double *Xi, double *xr, double *xi, int N) {
  for (int n = 0; n < N; ++n) {
    xr[n] = 0.0;
    xi[n] = 0.0;
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < N; ++k) {
      double wr = cos(2.0 * M_PI * n * k / N);
      double wi = sin(2.0 * M_PI * n * k / N);
      xr[n] += Xr[k] * wr - Xi[k] * wi;
      xi[n] += Xr[k] * wi + Xi[k] * wr;
    }
    xr[n] /= N;
    xi[n] /= N;
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
    fprintf(stderr, "Usage: %s [voice name] [skip] [frame length]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char *filename_input = (char *) malloc(100 * sizeof(char));
  char *filename_output = (char *) malloc(100 * sizeof(char));
  sprintf(filename_input, "%s%s%s", DIR_ASSET, argv[1], EXT_DAT);
  sprintf(filename_output, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_DFT, EXT_DAT);

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
    exit(EXIT_FAILURE);
  }

  fseek(fp_input, nskip * sizeof(short), SEEK_SET);
  fread(sdata, sizeof(short), framelen, fp_input);

  /* windowing */
  for (int i = 0; i < framelen; ++i) {
    xr[i] = (0.54 - 0.46 * cos(2 * M_PI * i / (framelen - 1))) * sdata[i];
    xi[i] = 0.0;
  }

/* DFT */
#ifdef STRESSTEST
  int repeat = 1000;
  long time_begin = now();
  for (int i = 0; i < repeat; ++i) {
    DFT(xr, xi, Xr, Xi, framelen);
  }
  long time_end = now();
  printf("DFT: %ld nsec / proc\n", (time_end - time_begin) / repeat);
#else
  DFT(xr, xi, Xr, Xi, framelen);
#endif
  

/* plot the result */
#ifndef INVERSE
  for (int i = 0; i < framelen; ++i) {
    double spec = log10((Xr[i] * Xr[i] + Xi[i] * Xi[i]) / framelen);
    fprintf(fp_output, "%d %lf\n", i * SAMPLING_RATE / framelen, spec);
  }

#else
  /* IFFT trial */
  DFFT(Xr, Xi, xr, xi, framelen);

  for (int i = 0; i < framelen; ++i) printf("%d %lf\n", i, xr[i]);
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