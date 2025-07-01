#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <stdio.h>
#include <string.h>

#define MAX_HEURISTICS 50
#define MAX_PARAM_NAME_LENGTH 128
#define MAX_PARAMS_PER_HEUR 100

/** Structure to store parameter information */
typedef struct {
    char name[MAX_PARAM_NAME_LENGTH];
    int count;
} ParameterInfo;

/** List of heuristics to check */
const char* target_heuristics[] = {
    "rens",
    // "rounding",
    // "fracdiving",
    // "intshifting",
    "octane",
    // "feaspump",
    // "localbranching",
    // "rins",
    // "dins",
    // "crossover"
};
const int num_target_heuristics = 2;

/** Main function */
int main(int argc, char** argv) {
    SCIP* scip = NULL;
    SCIP_HEUR** heuristics = NULL;
    int nheurs = 0;
    ParameterInfo* common_params = NULL;
    int ncommon_params = 0;
    int i, j, k;
    
    /* Initialize SCIP */
    SCIP_CALL( SCIPcreate(&scip) );
    SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
    
    /* Get all primal heuristics */
    heuristics = SCIPgetHeurs(scip);
    nheurs = SCIPgetNHeurs(scip);
    
    if(nheurs == 0) {
        printf("No primal heuristics found!\n");
        SCIPfree(&scip);
        return 1;
    }
    
    /* Find and list our target heuristics */
    SCIP_HEUR* target_heur_ptrs[num_target_heuristics];
    int found_heuristics = 0;
    
    printf("Looking for specific heuristics:\n");
    for(i = 0; i < num_target_heuristics; i++) {
        SCIP_Bool found = FALSE;
        for(j = 0; j < nheurs; j++) {
            if(strcmp(target_heuristics[i], SCIPheurGetName(heuristics[j])) == 0) {
                target_heur_ptrs[found_heuristics] = heuristics[j];
                found_heuristics++;
                found = TRUE;
                printf("  Found %s\n", target_heuristics[i]);
                break;
            }
        }
        if(!found) {
            printf("  Warning: %s not found\n", target_heuristics[i]);
        }
    }
    
    if(found_heuristics == 0) {
        printf("None of the target heuristics were found!\n");
        SCIPfree(&scip);
        return 1;
    }
    
    printf("\nChecking %d heuristics for common parameters...\n", found_heuristics);
    
    /* Allocate memory for parameter tracking */
    common_params = (ParameterInfo*)malloc(MAX_PARAMS_PER_HEUR * sizeof(ParameterInfo));
    if(common_params == NULL) {
        printf("Memory allocation failed!\n");
        SCIPfree(&scip);
        return 1;
    }
    
    /* Initialize with parameters from the first target heuristic */
    char heurname[MAX_PARAM_NAME_LENGTH];
    snprintf(heurname, MAX_PARAM_NAME_LENGTH, "heuristics/%s/", SCIPheurGetName(target_heur_ptrs[0]));
    
    SCIP_PARAM** params = SCIPgetParams(scip);
    int nparams = SCIPgetNParams(scip);
    
    int first_heur_params = 0;
    for(i = 0; i < nparams; ++i) {
        const char* paramname = SCIPparamGetName(params[i]);
        if(strncmp(paramname, heurname, strlen(heurname)) == 0) {
            strncpy(common_params[first_heur_params].name, 
                   paramname + strlen(heurname), 
                   MAX_PARAM_NAME_LENGTH - 1);
            common_params[first_heur_params].count = 1;
            first_heur_params++;
        }
    }
    ncommon_params = first_heur_params;
    
    /* Compare with parameters from other target heuristics */
    for(i = 1; i < found_heuristics; ++i) {
        snprintf(heurname, MAX_PARAM_NAME_LENGTH, "heuristics/%s/", SCIPheurGetName(target_heur_ptrs[i]));
        
        /* Check which common parameters exist in this heuristic */
        for(j = 0; j < ncommon_params; ) {
            SCIP_Bool found = FALSE;
            char fullparamname[MAX_PARAM_NAME_LENGTH];
            
            /* Construct full parameter name */
            snprintf(fullparamname, MAX_PARAM_NAME_LENGTH, "%s%s", heurname, common_params[j].name);
            
            /* Search for this parameter */
            for(k = 0; k < nparams; ++k) {
                if(strcmp(fullparamname, SCIPparamGetName(params[k])) == 0) {
                    found = TRUE;
                    break;
                }
            }
            
            if(found) {
                common_params[j].count++;
                j++;
            } else {
                /* Remove this parameter from common list */
                if(j < ncommon_params - 1) {
                    memmove(&common_params[j], &common_params[j+1], 
                           (ncommon_params - j - 1) * sizeof(ParameterInfo));
                }
                ncommon_params--;
            }
        }
        
        if(ncommon_params == 0) {
            break;  // No common parameters left
        }
    }
    
    /* Print results */
    if(ncommon_params == 0) {
        printf("No common parameters found among the specified heuristics.\n");
    } else {
        printf("Parameters common to all %d heuristics:\n", found_heuristics);
        for(i = 0; i < ncommon_params; ++i) {
            if(common_params[i].count == found_heuristics) {
                printf("  %s\n", common_params[i].name);
            }
        }
    }
    
    /* Clean up */
    free(common_params);
    SCIPfree(&scip);
    
    return 0;
}

/** Error handling wrapper */
#undef SCIP_CALL
#define SCIP_CALL(x) do {                      \
    SCIP_RETCODE _restat_;                     \
    if( (_restat_ = (x)) != SCIP_OKAY ) {      \
        printf("Error <%d> in line %d\n",       \
            _restat_, __LINE__);                \
        SCIPfree(&scip);                       \
        return _restat_;                       \
    }                                          \
} while( FALSE )