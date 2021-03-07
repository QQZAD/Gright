#pragma once
#define _G4C_AC_TEST_
#ifdef _G4C_AC_TEST_
#include <stdio.h>
#include <stdlib.h>
#endif

#include <stddef.h>

#include <errno.h>
#include "g4c.h"
#include "g4c_ac.h"

#include <cstdio>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>

using namespace std;

class ACState
{
public:
	int id;
	ACState *prev;
	map<char, int> go;
	set<int> output;
	int failure;

	int transition[AC_ALPHABET_SIZE];

	ACState() : id(0), prev(0), failure(-1) {}
	ACState(int sid) : id(sid), prev(0), failure(-1) {}
	ACState(int sid, ACState *sprev) : id(sid), prev(sprev), failure(-1) {}
	~ACState() {}
};

class ACMachine
{
public:
	vector<ACState *> states;
	char **patterns;
	int npatterns;

	ACMachine() {}
	~ACMachine()
	{
		for (int i = 0; i < states.size(); i++)
			delete states[i];
		states.clear();
	}
};

extern "C" void
ac_build_goto(char *kws[], int n, ACMachine *acm);

extern "C" void
ac_build_failure(ACMachine *acm);

extern "C" void
ac_build_transition(ACMachine *acm);

extern "C"
{
#include <stdlib.h>
#include <string.h>
}

extern "C" int
g4c_cpu_acm_match(g4c_acm_t *acm, uint8_t *data, int len);

extern "C" int
ac_build_machine(ac_machine_t *acm, char **patterns,
				 int npatterns, unsigned int memflags);

extern "C" void
ac_release_machine(ac_machine_t *acm);

extern "C" int
ac_match(char *str, int len, unsigned int *res, int once, ac_machine_t *acm);
#ifdef _G4C_AC_TEST_

static void
dump_state(ACState *s, char *kws[]);

static void
dump_c_state(ac_state_t *s, ac_machine_t *acm);

extern "C" void
dump_c_acm(ac_machine_t *acm);

#endif
