#include "savestate.h"

#include "PR/ultratypes.h"

#include <unistd.h>

extern char __data_start[];
//extern char etext[];
//extern char edata[];
extern char end[];

extern uint8_t *texcache;
extern uint32_t texcache_size;

static void* bss_save=0;
static size_t bss_size=0;

static uint8_t *texcache_save=0;
//uint32_t texcache_size_save=0; no need because it's restored via bss!

static int savestate_request=0;

// found using pmap <pid>
// everything seems constant, maybe not depending on the build
// memstart is the first rw--- chunk of memory
// memstop looks like end[], but I can't find a way to get memstart in a similar, more reliable way
//static void* memstart = 0xeae000;
//static void* memstop = 0x1655000;

// edit : found a more precise, linker-time reliable way
static void* memstart = __data_start;
static void* memstop = end;

void print_stats() {
    //printf("       etext  %10p\n", etext);
    //printf("       edata  %10p\n", edata);
    printf("data start    %10p\n",__data_start);
    printf("       end    %10p\n", end);
}

void savestate_check() {
    if (savestate_request) {
        // Positive value : save request
        if (savestate_request > 0) {
            printf("Saving state\n");
            print_stats();

            if (bss_save) {
                free(bss_save);
                bss_save = NULL;
            }
            if (texcache_save) {
                free(texcache_save);
                texcache_save = NULL;
            }

            texcache_save = malloc(texcache_size);
            memcpy(texcache_save,texcache,texcache_size);

            bss_size = memstop - memstart;
            bss_save = malloc(bss_size);
            SDL_LockAudio();
            memcpy(bss_save,memstart,bss_size);
            SDL_UnlockAudio();
        }
        // Negative value : load request
        else {
            savestate_request = -savestate_request;
            if (bss_save) {
                printf("Loading state\n");
                print_stats();
                free(texcache);
                SDL_LockAudio();
                memcpy(memstart,bss_save,bss_size);
                SDL_UnlockAudio();
                texcache=malloc(texcache_size);
                memcpy(texcache,texcache_save,texcache_size);
            }
        }
        savestate_request = 0;
    }
}

void savestate_request_save(int slot) {
    savestate_request = 1;
}

void savestate_request_load(int slot) {
    savestate_request = -1;
}
