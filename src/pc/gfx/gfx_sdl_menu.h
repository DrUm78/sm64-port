#ifndef _MENU_H_
#define _MENU_H_


typedef enum{
    MENU_TYPE_VOLUME,
    MENU_TYPE_BRIGHTNESS,
    MENU_TYPE_SAVE,
    MENU_TYPE_LOAD,
    MENU_TYPE_ASPECT_RATIO,
    MENU_TYPE_EXIT,
    MENU_TYPE_POWERDOWN,
    NB_MENU_TYPES,
} ENUM_MENU_TYPE;


///------ Definition of the different aspect ratios 
#define ASPECT_RATIOS \
    X(ASPECT_RATIOS_TYPE_CROPPED, "CROPPED") \
    X(ASPECT_RATIOS_TYPE_STRETCHED, "STRETCHED") \
    X(NB_ASPECT_RATIOS_TYPES, "") 

////------ Enumeration of the different aspect ratios ------
#undef X
#define X(a, b) a,
typedef enum {ASPECT_RATIOS} ENUM_ASPECT_RATIOS_TYPES;

////------ Defines to be shared -------
#define RES_HW_SCREEN_HORIZONTAL    240
#define RES_HW_SCREEN_VERTICAL      240
#define STEP_CHANGE_VOLUME          10
#define STEP_CHANGE_BRIGHTNESS      10
#define NOTIF_SECONDS_DISP			2

////------ Menu commands -------
#define SHELL_CMD_VOLUME_GET                "volume_get"
#define SHELL_CMD_VOLUME_SET                "volume_set"
#define SHELL_CMD_BRIGHTNESS_GET            "brightness_get"
#define SHELL_CMD_BRIGHTNESS_SET            "brightness_set"
#define SHELL_CMD_NOTIF                     "notif_set"
#define SHELL_CMD_WRITE_QUICK_LOAD_CMD      "write_args_quick_load_file"
#define SHELL_CMD_TURN_AMPLI_ON             "start_audio_amp 1"
#define SHELL_CMD_TURN_AMPLI_OFF            "start_audio_amp 0"
#define SHELL_CMD_CANCEL_SCHED_POWERDOWN    "cancel_sched_powerdown"
#define SHELL_CMD_INSTANT_PLAY              "instant_play"
#define SHELL_CMD_SHUTDOWN_FUNKEY           "shutdown_funkey"

////------ Global variables -------
extern int volume_percentage;
extern int brightness_percentage;

extern const char *aspect_ratio_name[];
extern int aspect_ratio;
extern int aspect_ratio_factor_percent;
extern int aspect_ratio_factor_step;
extern int stop_menu_loop;

//####################################
//# Functions
//####################################
void init_menu_SDL();
void deinit_menu_SDL();
void init_menu_zones();
void init_menu_system_values();
void run_menu_loop();


#endif /* _MENU_H_ */
