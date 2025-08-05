#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/heuristics.h"
#include "scip/pub_misc.h"
#include "scip/cons_linear.h"
#include "scip/type_timing.h"
// #include "scip/heur_directionalrounding.h"

#define HEUR_NAME              "directionalrounding"
#define HEUR_DESC              "fixes fractional vars to floor or ceil and resolves LP"
#define HEUR_DISPCHAR          'D'
#define HEUR_PRIORITY          10000
#define HEUR_FREQ              -1
#define HEUR_FREQOFS           0
#define HEUR_MAXDEPTH          -1
// #define HEUR_TIMING            SCIP_HEURTIMING_AFTERNODE
#define HEUR_TIMING SCIP_HEURTIMING_AFTERLPNODE

#define HEUR_USESSUBSCIP       FALSE

#define DEFAULT_ROUNDDOWN      TRUE

/** heuristic execution method */
static
SCIP_DECL_HEUREXEC(heurExecDirectionalRounding)
{
    SCIP_VAR** vars;
    int nvars = SCIPgetNOrigVars(scip);
    vars = SCIPgetOrigVars(scip);

    SCIP_Bool down;
    SCIP_CALL(SCIPgetBoolParam(scip, "heuristics/" HEUR_NAME "/down", &down));

    SCIP_VAR** fixvars;
    SCIP_Real* fixvals;
    int nfix = 0;

    SCIP_CALL(SCIPallocBufferArray(scip, &fixvars, nvars));
    SCIP_CALL(SCIPallocBufferArray(scip, &fixvals, nvars));

    if (!SCIPisRelaxSolValid(scip)) {
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
    }

    for (int i = 0; i < nvars; ++i)
    {
        SCIP_VAR* var = vars[i];

        if (!SCIPvarIsIntegral(var))
            continue;

        SCIP_Real val;
        if (SCIPisRelEQ(scip, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)))
            continue; // variable already fixed

        val = SCIPgetRelaxSolVal(scip, var);

        if (!SCIPisFeasIntegral(scip, val))
        {
            SCIP_Real fixedval = down ? SCIPfloor(scip, val) : SCIPceil(scip, val);
            fixvars[nfix] = var;
            fixvals[nfix] = fixedval;
            ++nfix;
        }
    }

    if (nfix == 0)
    {
        *result = SCIP_DIDNOTFIND;
        SCIPfreeBufferArray(scip, &fixvals);
        SCIPfreeBufferArray(scip, &fixvars);
        return SCIP_OKAY;
    }

    SCIP_CALL(SCIPstartProbing(scip));
    for (int i = 0; i < nfix; ++i)
    {
        SCIP_CALL(SCIPfixVarProbing(scip, fixvars[i], fixvals[i]));
    }

    SCIP_Bool *Cutoff = NULL;
    SCIP_CALL(SCIPsolveProbingLP(scip, FALSE, FALSE, Cutoff));

    if (SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL)
    {
        SCIP_SOL* sol;
        SCIP_CALL(SCIPcreateSol(scip, &sol, heur));

        for (int i = 0; i < nvars; ++i)
        {
            SCIP_Real lpval = SCIPgetSolVal(scip, NULL, vars[i]);
            SCIP_CALL(SCIPsetSolVal(scip, sol, vars[i], lpval));
        }

        SCIP_Bool stored;
        SCIP_CALL(SCIPtrySolFree(scip, &sol, TRUE, TRUE, TRUE, TRUE, TRUE, &stored));
        if (stored)
            *result = SCIP_FOUNDSOL;
    }

    SCIP_CALL(SCIPendProbing(scip));

    SCIPfreeBufferArray(scip, &fixvals);
    SCIPfreeBufferArray(scip, &fixvars);

    return SCIP_OKAY;
}

/** includes the directional rounding heuristic */
SCIP_RETCODE SCIPincludeHeurDirectionalRounding(SCIP* scip)
{
    SCIP_HEUR* heur = NULL;

    SCIP_CALL(SCIPincludeHeurBasic(scip, &heur,
        HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ,
        HEUR_FREQOFS, HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
        heurExecDirectionalRounding, NULL));

    SCIP_CALL(SCIPaddBoolParam(scip,
        "heuristics/" HEUR_NAME "/down",
        "round fractional variables down (TRUE) or up (FALSE)",
        NULL, FALSE, DEFAULT_ROUNDDOWN, NULL, NULL));

    return SCIP_OKAY;
}
