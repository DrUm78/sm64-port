#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(TARGET_OD) || defined(TARGET_LINUX)
#define FULLPATH_BUFFER_SIZE 2024
#include <sys/stat.h>
#include <errno.h>
#endif
#include "unistd.h"
#include "fsutils.h"

#if defined(TARGET_OD) || defined(TARGET_LINUX)
void get_home_filename(const char* filename, char* outbuffer) {
    struct stat info;

    snprintf(outbuffer, FULLPATH_BUFFER_SIZE, "%s/.sm64-port/", getenv("HOME"));
    if (stat(outbuffer, &info) != 0) {
        fprintf(stderr, "Creating '%s' for the first time...\n", outbuffer);
        mkdir(outbuffer, 0700);
        if (stat(outbuffer, &info) != 0) {
            fprintf(stderr, "Unable to create '%s': %s\n", outbuffer, strerror(errno));
            abort();
        }
    }

    snprintf(outbuffer, FULLPATH_BUFFER_SIZE, "%s/.sm64-port/%s", getenv("HOME"), filename);
}
#endif


FILE *fopen_home(const char *filename, const char *mode)
{
#if defined(TARGET_OD) || defined(TARGET_LINUX)
    char fnamepath[FULLPATH_BUFFER_SIZE];
    get_home_filename(filename, fnamepath);
    return fopen(fnamepath, mode);
#else
    return fopen(filename, mode);
#endif
} 

int file_exists_home(const char *filename) {
#if defined(TARGET_OD) || defined(TARGET_LINUX)
    char fnamepath[FULLPATH_BUFFER_SIZE];
    get_home_filename(filename, fnamepath);
    return access(fnamepath, F_OK) == 0;
#else
    return access(filename, F_OK) == 0;
#endif
}