#pragma once

extern int CURR_BATCH;
extern bool scheduleExit;
extern void *getGpuTasks(void *argc);
extern void *finishGpuTasks(void *argc);