#include "ac.hpp"

extern "C" void
ac_build_goto(char *kws[], int n, ACMachine *acm)
{
    int sid = 1;
    int i;
    vector<ACState *> &states = acm->states;

    states.push_back(new ACState(0, 0));
    for (i = 0; i < n; i++)
    {
        char *kw = kws[i];
        int nsid = 0, j = 0;
        ACState *s = states[0];

        while (kw[j] && s->go.find(kw[j]) != s->go.end())
        {
            nsid = s->go[kw[j]];
            s = states[nsid];
            j++;
        }

        while (kw[j])
        {
            ACState *ns = new ACState(sid++, s);
            states.push_back(ns);
            s->go[kw[j]] = ns->id;
            s = ns;
            j++;
        }

        s->output.insert(i);
    }

    // setup initial state goto function
    for (i = 0; i < AC_ALPHABET_SIZE; i++)
    {
        if (states[0]->go.find(i) == states[0]->go.end())
            states[0]->go[i] = 0;
    }
}

extern "C" void
ac_build_failure(ACMachine *acm)
{
    queue<int> q;
    ACState *s = acm->states[0];

    for (map<char, int>::iterator ite = s->go.begin();
         ite != s->go.end(); ++ite)
    {
        if (ite->second != s->id)
            q.push(ite->second);
        acm->states[ite->second]->failure = s->id; //0
    }

    while (!q.empty())
    {
        int rid = q.front();
        ACState *r = acm->states[rid];

        q.pop();
        for (map<char, int>::iterator ite = r->go.begin();
             ite != r->go.end(); ++ite)
        {
            q.push(ite->second);

            int fid = r->failure;
            ACState *f = acm->states[fid];
            while (f->go.find(ite->first) == f->go.end())
            {
                fid = f->failure;
                f = acm->states[fid];
            }
            acm->states[ite->second]->failure = f->go[ite->first];
            f = acm->states[f->go[ite->first]];
            acm->states[ite->second]->output.insert(
                f->output.begin(),
                f->output.end());
        }
    }
}

extern "C" void
ac_build_transition(ACMachine *acm)
{
    queue<int> q;
    ACState *s = acm->states[0];

    for (int i = 0; i < AC_ALPHABET_SIZE; i++)
    {
        s->transition[i] = s->go[i];
        if (s->go[i] != s->id)
            q.push(s->go[i]);
    }

    while (!q.empty())
    {
        int rid = q.front();
        ACState *r = acm->states[rid];
        q.pop();

        for (int i = 0; i < AC_ALPHABET_SIZE; i++)
        {
            if (r->go.find(i) != r->go.end())
            {
                q.push(r->go[i]);
                r->transition[i] = r->go[i];
            }
            else
                r->transition[i] = acm->states[r->failure]->transition[i];
        }
    }
}

extern "C" int
g4c_cpu_acm_match(g4c_acm_t *acm, uint8_t *data, int len)
{
    int ret = 0, nid;
    uint8_t c;

    int cid = 0;
    for (int i = 0; i < len; i++)
    {
        c = data[i];
        nid = g4c_acm_htransitions(acm, cid)[c];
        if (*g4c_acm_houtput(acm, cid))
        {
            if (!ret || ret > *g4c_acm_houtput(acm, cid))
            {
                ret = *g4c_acm_houtput(acm, cid);
            }
        }
        cid = nid;
    }

    return ret;
}

extern "C" int
ac_build_machine(ac_machine_t *acm, char **patterns,
                 int npatterns, unsigned int memflags)
{
    size_t
        psz = 0,   // total size of all pattern strings, including '\0'
        ppsz = 0,  // total size of all pointers of pattern strings
        stsz = 0,  // total size of all states
        trsz = 0,  // total size of all transition table
        outsz = 0; // total size of all output function table
    int i;
    ACMachine cppacm;

    // Build C++ ACMachine
    ac_build_goto(patterns, npatterns, &cppacm);
    ac_build_failure(&cppacm);
    ac_build_transition(&cppacm);

    memset(acm, 0, sizeof(ac_machine_t));

    // easy settings
    acm->nstates = static_cast<int>(cppacm.states.size());
    acm->npatterns = npatterns;
    acm->memflags = memflags;

    // calculate all sizes
    for (i = 0; i < npatterns; i++)
        psz += strlen(patterns[i]) + 1;
    ppsz = sizeof(char *) * npatterns;

    stsz = acm->nstates * sizeof(ac_state_t);
    stsz = g4c_round_up(stsz, G4C_PAGE_SIZE);

    trsz = acm->nstates * AC_ALPHABET_SIZE * sizeof(int);
    trsz = g4c_round_up(trsz, G4C_PAGE_SIZE);

    for (i = 0; i < acm->nstates; i++)
    {
        acm->noutputs += cppacm.states[i]->output.size();
    }
    outsz = acm->noutputs * sizeof(int);
    outsz = g4c_round_up(outsz, G4C_PAGE_SIZE);

    acm->memsz = psz + ppsz + stsz + trsz + outsz;
    acm->memsz = g4c_round_up(acm->memsz, G4C_PAGE_SIZE);

    // memory allocation and assignment
    acm->mem = malloc(acm->memsz);
    if (acm->mem)
    {
        int *tmpoutput;
        char *ptn;
        int output_offset = 0;

        // default layout:
        //  --------------------
        //  | states ...       |
        //  +------------------+
        //  | transitions ...  |
        //  +------------------+
        //  | outputs ...      |
        //  +------------------+
        //  | pattern ptrs ... |
        //  +------------------+
        //  | patterns ...     |
        //  --------------------
        //
        acm->states = (ac_state_t *)acm->mem;
        acm->transitions = (int *)g4c_ptr_add(acm->mem, stsz);
        acm->outputs = (int *)g4c_ptr_add(acm->transitions, trsz);
        acm->patterns = (char **)g4c_ptr_add(acm->outputs, outsz);

        // copy each state's information
        tmpoutput = acm->outputs;
        for (i = 0; i < acm->nstates; i++)
        {
            ACState *cpps = cppacm.states[i];
            ac_state_t *acs = acm->states + i;

            acs->id = cpps->id;
            acs->prev = (cpps->prev ? cpps->prev->id : -1);
            acs->noutput = (int)(cpps->output.size());
            acs->output = (acs->noutput ? output_offset : -1);

            // copy output table
            for (set<int>::iterator ite = cpps->output.begin();
                 ite != cpps->output.end();
                 ++ite)
            {
                *tmpoutput = *ite;
                ++tmpoutput;
                ++output_offset;
            }

            // copy transition table
            memcpy(acm_state_transitions(acm, i),
                   cpps->transition,
                   sizeof(int) * AC_ALPHABET_SIZE);
        }

        ptn = (char *)(acm->patterns + npatterns);
        for (i = 0; i < npatterns; i++)
        {
            strcpy(ptn, patterns[i]);
            acm->patterns[i] = ptn;
            ptn += strlen(patterns[i]) + 1;
        }

        // OK
        return 0;
    }

    return -ENOMEM;
}

extern "C" void
ac_release_machine(ac_machine_t *acm)
{
    free(acm->mem);
}

extern "C" int
ac_match(char *str, int len, unsigned int *res, int once, ac_machine_t *acm)
{
    int nm = 0;

    char c;
    ac_state_t *st = acm_state(acm, 0);
    for (int i = 0; i < len; i++)
    {
        c = str[i];
        if (c < 0)
            return -1;
        int nid = acm_state_transitions(acm, st->id)[(int)c];
        st = acm_state(acm, nid);
        if (st->noutput > 0)
        {
            if (res)
            {
                for (int j = 0; j < st->noutput; j++)
                {
                    int ot = acm_state_output(acm, st->output)[j];
                    res[ot] = 0x80000000 |
                              (i + 1 - strlen(acm_pattern(acm, ot)));
                }
            }
            if (!nm && once)
            {
                return st->noutput;
            }
            nm += st->noutput;
        }
    }
    return nm;
}

#ifdef _G4C_AC_TEST_

static void
dump_state(ACState *s, char *kws[])
{
    map<char, int>::iterator ite;
    set<int>::iterator oite;

    printf("S %3d, previous: %3d, failure: %3d\n",
           s->id, (s->prev ? s->prev->id : -1), s->failure);

    printf("\t%4lu Gotos: ", s->go.size());
    for (ite = s->go.begin(); ite != s->go.end(); ++ite)
    {
        printf("(%4d --%c--> %-4d) ", s->id, ite->first, ite->second);
    }
    printf("\n");

    printf("\t%4lu Outputs: \n", s->output.size());
    for (oite = s->output.begin(); oite != s->output.end(); ++oite)
    {
        printf("\t\t%s\n", kws[*oite]);
    }
    printf("\n");

    printf("\tTransitions: \n\t");
    for (int i = 0; i < 128; i++)
    {
        printf(" (%3d,%3d)", i, s->transition[i]);
    }
    printf("\n");
}

static void
dump_c_state(ac_state_t *s, ac_machine_t *acm)
{
    printf("State %d, previous: %d, #outputs: %d\n", s->id, s->prev, s->noutput);
    printf("\tOutputs:\n");
    for (int i = 0; i < s->noutput; i++)
    {
        int ot = acm_state_output(acm, s->output)[i];
        printf("\t\t%3d: %s\n", ot, acm->patterns[ot]);
    }

    int *tr = acm_state_transitions(acm, s->id);
    printf("\tTransitions:\n");
    for (int i = 0; i < AC_ALPHABET_SIZE; i++)
    {
        if (tr[i] != 0)
        {
            printf("\t\t%d--%c-->%d\n", s->id, i, tr[i]);
        }
    }
}

extern "C" void
dump_c_acm(ac_machine_t *acm)
{
    printf("C version ACM:\n"
           "\tmem=%p, size=%lu, flags=0X%08X\n"
           "\tstates=%p, nstates=%d, sizeof(ac_state_t)=%lu\n"
           "\ttransitions=%p, outputs=%p, noutputs=%d\n"
           "\tpatterns=%p, npatterns=%d\n",
           acm->mem, acm->memsz, acm->memflags,
           acm->states, acm->nstates, sizeof(ac_state_t),
           acm->transitions, acm->outputs, acm->noutputs,
           acm->patterns, acm->npatterns);
    printf("States\n");
    for (int i = 0; i < acm->nstates; i++)
        dump_c_state(acm->states + i, acm);
}

#endif