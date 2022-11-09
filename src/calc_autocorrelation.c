#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WINDOW_SIZE   512
#define SAMPLING_RATE 16000
#define DIR_ASSET     "../asset/speech_sample/"
#define DIR_OUTPUT    "./output/"
#define EXT_DAT       ".dat"
#define SUF_COR       "_autocorrelation"

/**
 * @brief プログラムを終了させる
 * 
 * @param log error に出す項目
 */
void die(char* log) {
  perror(log);
  exit(EXIT_FAILURE);
}

/**
 * @brief hamming 窓を掛ける
 * 
 * @param raw_data    元々のデータ
 * @param window_data 窓を掛けた後のデータを格納する
 * @param size        データサイズ
 */
void window(short int* raw_data, double* window_data, int size) {
  for (int i = 0; i < size; ++i) {
    window_data[i] =
      (double) (0.54 - 0.46 * cos(2 * M_PI * i / (size - 1))) * raw_data[i];
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: %s [filename]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /* generate file names */
  char* filename_input = (char*) malloc(100 * sizeof(char));
  char* filename_output = (char*) malloc(100 * sizeof(char));
  sprintf(filename_input, "%s%s%s", DIR_ASSET, argv[1], EXT_DAT);
  sprintf(filename_output, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_COR, EXT_DAT);

  /* open file */
  int fd_read = open(filename_input, O_RDONLY);
  FILE* fp_write = fopen(filename_output, "w");
  if (fd_read == -1 || fp_write == NULL) {
    die("open");
  }

  /* read input file */
  short int* data = (short int*) malloc(WINDOW_SIZE * sizeof(short int));
  int read_byte = read(fd_read, data, WINDOW_SIZE * sizeof(short int));
  if (read_byte == -1) {
    die("read");
  }

  /* windowing */
  double* data_windowed = (double*) malloc(WINDOW_SIZE * sizeof(double));
  window(data, data_windowed, WINDOW_SIZE);

  double* autocorrelation_list = (double*) malloc(WINDOW_SIZE * sizeof(double));
  for (int tau = 0; tau < WINDOW_SIZE; ++tau) {
    autocorrelation_list[tau] = 0.0;
    for (int i = 0; i < WINDOW_SIZE - tau; ++i) {
      autocorrelation_list[tau] += data_windowed[i] * data_windowed[i + tau];
    }
    if (tau > 0) {
      autocorrelation_list[tau] /= autocorrelation_list[0];
    }
  }

  autocorrelation_list[0] = 1.0;

  for (int tau = 0; tau < WINDOW_SIZE; ++tau) {
    int written_byte =
      fprintf(fp_write, "%f %lf\n", (double) tau / SAMPLING_RATE,
              autocorrelation_list[tau]);
    if (written_byte == -1) {
      die("write");
    }
  }

  close(fd_read);
  fclose(fp_write);
  free(data);
  free(data_windowed);
  free(autocorrelation_list);
  return EXIT_SUCCESS;
}
