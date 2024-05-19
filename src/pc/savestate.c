#include "savestate.h"

#include "PR/ultratypes.h"

#include <stdlib.h>
#include <unistd.h>

#include "SDL/SDL.h"

#include "fsutils.h"

#define SAVESTATE_FILENAME "sm64_savestate"
#define SAVESTATE_VERSION 1

extern char __data_start[];
//extern char etext[];
//extern char edata[];
extern char end[];

extern uint8_t *texcache;
extern uint32_t texcache_size;

static uint8_t *texcache_save=0;

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

FILE *fopen_slot(int slot, const char* mode) {
    char* name;
    FILE* ret;

    name = malloc(sizeof(SAVESTATE_FILENAME) + 5);
	sprintf(name, "%s.%02d", SAVESTATE_FILENAME, slot);

    ret = fopen_home(name,mode);
    if (!ret) {
        printf("Failed to open file %s\n", name);
    }
    free(name);
    return ret;
}

void trysave(int slot) {
    const size_t bss_size = memstop - memstart;
    FILE* f;
    unsigned int version = SAVESTATE_VERSION;

    printf("Saving state %d...\n", slot);

    f = fopen_slot(slot,"w");
    if (!f) {
        printf("Failed to save savestate %d\n", slot);
        return;
    }

    // first entry : version
    fwrite(&version, sizeof version, 1, f);
    // second entry : data size
    fwrite(&bss_size, sizeof bss_size, 1, f);
    // third entry : texture cache size
    fwrite(&texcache_size, sizeof texcache_size, 1 , f);

    // write actual contents
    SDL_LockAudio();
    fwrite(memstart, sizeof(char), bss_size, f);
    SDL_UnlockAudio();
    fwrite(texcache, sizeof(char), texcache_size, f);

    // end it all
    if (fclose(f)==0) {
        printf("State %d saved successfully!\n", slot);
    }
    else {
        printf("Unexpected error while saving state %d\n", slot);
    }
}

void tryload(int slot) {
    const size_t bss_size = memstop - memstart;
    FILE* f;

    unsigned int version;
    size_t bss_size_read;
    uint32_t texcache_size_read;

    printf("Loading state %d...\n", slot);

    f = fopen_slot(slot,"r");
    if (!f) {
        printf("Failed to load savestate %d\n", slot);
        return;
    }

    // first check the version is correct
    fread(&version, sizeof version, 1, f);
    if (version != SAVESTATE_VERSION) {
        printf("Loading of savestate %d aborted : expected version %d but the file is version %d\n", slot, SAVESTATE_VERSION, version);
        fclose(f);
        return;
    }
    // second we check if the bss size is the same as ours, we don't want unmatched sizes
    fread(&bss_size_read, sizeof bss_size_read, 1, f);
    if (bss_size_read != bss_size) {
        printf("Loading of savestate %d aborted : mismatch size of the bss section. Expected %ld but found %ld\n", slot, bss_size, bss_size_read);
        fclose(f);
        return;
    }
    // I can't think of any sanity check for the texture cache
    fread(&texcache_size_read, sizeof texcache_size_read, 1, f);

    // finally some important reading
    free(texcache); // free the current texcache before its pointer becomes corrupted
    SDL_LockAudio();
    fread(memstart, sizeof(char), bss_size, f);
    SDL_UnlockAudio();
    texcache_size = texcache_size_read;
    texcache = malloc(texcache_size);
    fread(texcache,sizeof(char),texcache_size,f);

    // end it all
    if (fclose(f)==0) {
        printf("State %d loaded successfully!\n", slot);
    }
    else {
        printf("Unexpected error while loading state %d\n", slot);
    }
}

void savestate_check() {
    if (savestate_request) {
        // Positive value : save request
        if (savestate_request > 0) {
            trysave(savestate_request);
        }
        // Negative value : load request
        else {
            tryload(-savestate_request);
        }
        savestate_request = 0;
    }
}

void savestate_request_save(int slot) {
    savestate_request = slot;
}

void savestate_request_load(int slot) {
    savestate_request = -slot;
}
