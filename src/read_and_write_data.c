#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INPUT_SIZE  512
#define DIR_ASSET   "../asset/speech_sample/"
#define DIR_OUTPUT  "./output/"
#define EXT_DAT     ".dat"
#define SUF_RAW     "_raw"
#define SUF_HAMMING "_hamming"
#define SUF_HANNING "_hanning"

void die(char* log) {
  perror(log);
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: %s [filename]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /* generate file names */
  char* filename_input = (char*) malloc(100 * sizeof(char));
  char* filename_output_raw = (char*) malloc(100 * sizeof(char));
  char* filename_output_hamming = (char*) malloc(100 * sizeof(char));
  char* filename_output_hanning = (char*) malloc(100 * sizeof(char));

  sprintf(filename_input, "%s%s%s", DIR_ASSET, argv[1], EXT_DAT);
  sprintf(filename_output_raw, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_RAW,
          EXT_DAT);
  sprintf(filename_output_hamming, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_HAMMING,
          EXT_DAT);
  sprintf(filename_output_hanning, "%s%s%s%s", DIR_OUTPUT, argv[1], SUF_HANNING,
          EXT_DAT);

  /* open file */
  int fd_read = open(filename_input, O_RDONLY);
  FILE* fp_write_raw = fopen(filename_output_raw, "w");
  FILE* fp_write_hamming = fopen(filename_output_hamming, "w");
  FILE* fp_write_hanning = fopen(filename_output_hanning, "w");
  if (fd_read == -1 || fp_write_raw == NULL || fp_write_hamming == NULL ||
      fp_write_hanning == NULL) {
    die("open");
  }

  /* read input file */
  short int* data = (short int*) malloc(INPUT_SIZE * sizeof(short int));
  int read_byte = read(fd_read, data, INPUT_SIZE * sizeof(short int));
  if (read_byte == -1) {
    die("read");
  }

  /* calculate and write in output file */
  for (int i = 0; i < INPUT_SIZE; ++i) {
    int written_byte_raw = fprintf(fp_write_raw, "%d %d\n", i, data[i]);
    int written_byte_hamming = fprintf(
      fp_write_hamming, "%d %f\n", i,
      (double) (0.54 - 0.46 * cos(2 * M_PI * i / (INPUT_SIZE - 1))) * data[i]);
    int written_byte_haing = fprintf(
      fp_write_hanning, "%d %f\n", i,
      (double) (0.5 - 0.5 * cos(2 * M_PI * i / (INPUT_SIZE - 1))) * data[i]);
    if (written_byte_raw == -1 || written_byte_hamming == -1) {
      die("write");
    }
  }

  close(fd_read);
  fclose(fp_write_raw);
  fclose(fp_write_hamming);
  fclose(fp_write_hanning);
  return EXIT_SUCCESS;
}