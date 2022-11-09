#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define WINDOW_SIZE   512
#define SAMPLING_RATE 16000
#define DIR_ASSET     "../asset/speech_sample/"
#define DIR_OUTPUT    "./output/"
#define EXT_DAT       ".dat"
#define SUF_COR       "_autocor_all"

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
void window(short int* raw_data, double* window_data, int window_size) {
  for (int i = 0; i < window_size; ++i) {
    window_data[i] =
      (double) (0.54 - 0.46 * cos(2 * M_PI * i / (window_size - 1))) *
      raw_data[i];
  }
}

/**
 * @brief 最小値を求める
 * 
 * @param value1 値１
 * @param value2 値２
 * @return int value1 と value2 のうち小さい方
 */
int min(int value1, int value2) {
  if (value1 < value2) {
    return value1;
  } else {
    return value2;
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
  FILE* fp_read = fopen(filename_input, "r");
  FILE* fp_write = fopen(filename_output, "w");
  short int* data = (short int*) malloc(100000 * sizeof(short int));
  short int* buf = (short int*) malloc(WINDOW_SIZE * sizeof(short int));
  if (fp_read == NULL) {
    die("open");
  }

  /* データを data に読み込む */
  int counter = 0;
  int data_size = 0;
  while (true) {
    int read_byte = fread(buf, sizeof(short int), WINDOW_SIZE, fp_read);
    if (read_byte == -1) {
      die("read");
    }
    if (read_byte == 0) {
      break;
    }
    for (int i = 0; i < read_byte; ++i) {
      data[counter * WINDOW_SIZE + i] = buf[i];
      data_size += 1;
    }
    counter += 1;
  }

  for (int num_skip = 0; WINDOW_SIZE / 2 * num_skip + WINDOW_SIZE < data_size;
       ++num_skip) {
    int index_cur = WINDOW_SIZE / 2 * num_skip;

    double* window_data = (double*) malloc(WINDOW_SIZE * sizeof(double));
    for (int i = 0; i < WINDOW_SIZE; ++i) {
      window_data[i] =
        (double) (0.54 - 0.46 * cos(2 * M_PI * i / (WINDOW_SIZE - 1))) *
        data[index_cur + i];
    }

    double* cor_list = (double*) malloc(WINDOW_SIZE * sizeof(double));
    for (int i = 0; i < WINDOW_SIZE; ++i) {
      cor_list[i] = 0.0;
    }

    double r0 = 0.0;
    for (int i = 0; i < WINDOW_SIZE; ++i) {
      r0 += window_data[i] * window_data[i];
    }

    for (int tau = 0; tau < WINDOW_SIZE; ++tau) {
      for (int i = 0; i < WINDOW_SIZE - tau; ++i) {
        cor_list[tau] += (double) window_data[i] * window_data[i + tau];
      }
      cor_list[tau] /= r0;
    }

    double* cor_peaks_value = (double*) malloc(WINDOW_SIZE * sizeof(double));
    int* cor_peaks_tau = (int*) malloc(WINDOW_SIZE * sizeof(int));
    int cor_peaks_count = 0;
    for (int tau = 1; tau < WINDOW_SIZE - 1; ++tau) {
      if (cor_list[tau - 1] < cor_list[tau] &&
          cor_list[tau] < cor_list[tau + 1]) {
        cor_peaks_value[cor_peaks_count] = cor_list[tau];
        cor_peaks_tau[cor_peaks_count] = tau;
        cor_peaks_count += 1;
      }
    }

    double cor_peaks_max_value = 0.0;
    int cor_peaks_max_tau = 0;
    for (int i = 0; i < cor_peaks_count; ++i) {
      if (cor_peaks_value[i] > cor_peaks_max_value) {
        cor_peaks_max_value = cor_peaks_value[i];
        cor_peaks_max_tau = cor_peaks_tau[i];
      }
    }

    fprintf(fp_write, "%d %f\n", num_skip,
            log10((double) SAMPLING_RATE / cor_peaks_max_tau));
    free(window_data);
    free(cor_list);
    free(cor_peaks_value);
    free(cor_peaks_tau);
  }

  free(data);
  free(buf);
  fclose(fp_read);
  fclose(fp_write);

  return EXIT_SUCCESS;
}
