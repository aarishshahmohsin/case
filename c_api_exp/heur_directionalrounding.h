#ifndef __SCIP_HEUR_DIRECTIONALROUNDING_H__
#define __SCIP_HEUR_DIRECTIONALROUNDING_H__

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the directional rounding primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurDirectionalRounding(SCIP* scip);

#ifdef __cplusplus
}
#endif

#endif
