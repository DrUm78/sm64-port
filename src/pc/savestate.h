#ifndef _SAVESTATE_H
#define _SAVESTATE_H

#define QUICKSAVE_SLOT 0xFFFF
#define SAVESTATE_EXCLUDE __attribute__((section("dontsave")))

void savestate_check();

void savestate_request_save(int slot);
void savestate_request_load(int slot);

void savestate_get_name(int slot, char* namebuffer);

#endif
