#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "scip/scip.h"
#include "scip/scipdefplugins.h"

typedef struct
{
    double *data;
    int rows;
    int cols;
} Matrix;

typedef struct
{
    double initial_reach;
    double reach;
    double *hyperplane_w;
    double bias_c;
    double *x_vals;
    double *y_vals;
    double precision_violation_v;
    long node_count;
    double time_taken;
    int status; // 0 = optimal, 1 = timelimit, -1 = error
    char error_msg[256];
} SolverResults;

// Function to create matrix
Matrix *create_matrix(int rows, int cols)
{
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    mat->data = (double *)calloc(rows * cols, sizeof(double));
    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

// Function to free matrix
void free_matrix(Matrix *mat)
{
    if (mat)
    {
        free(mat->data);
        free(mat);
    }
}

// Function to get matrix element
double get_element(Matrix *mat, int row, int col)
{
    return mat->data[row * mat->cols + col];
}

// Function to set matrix element
void set_element(Matrix *mat, int row, int col, double value)
{
    mat->data[row * mat->cols + col] = value;
}

// Function to compute dot product of matrix row with vector
double dot_product_row(Matrix *mat, int row, double *vec)
{
    double result = 0.0;
    for (int j = 0; j < mat->cols; j++)
    {
        result += get_element(mat, row, j) * vec[j];
    }
    return result;
}

SolverResults *scip_solver(
    double theta,
    double theta0,
    double theta1,
    Matrix *P,
    Matrix *N,
    double epsilon_P,
    double epsilon_N,
    double epsilon_R,
    double lambda_param,
    char *dataset_name,
    int run,
    double *init_w,
    double init_c,
    int print_output,
    double time_limit)
{
    SCIP *scip = NULL;
    SCIP_RETCODE retcode;
    SolverResults *results = (SolverResults *)malloc(sizeof(SolverResults));

    // Initialize results structure
    memset(results, 0, sizeof(SolverResults));
    results->status = -1;

    // Combine P and N matrices into X
    int num_positive = P->rows;
    int num_negative = N->rows;
    int total_samples = num_positive + num_negative;
    int n_features = P->cols;

    Matrix *X = create_matrix(total_samples, n_features);

    // Copy P samples
    for (int i = 0; i < num_positive; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            set_element(X, i, j, get_element(P, i, j));
        }
    }

    // Copy N samples
    for (int i = 0; i < num_negative; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            set_element(X, num_positive + i, j, get_element(N, i, j));
        }
    }

    // Set lambda parameter if not provided
    if (lambda_param <= 0)
    {
        lambda_param = (num_positive + 1) * theta1;
    }

    // Calculate initial reach if init_w and init_c are provided
    double initial_reach = 0.0;
    int *initial_xs = NULL;
    int *initial_ys = NULL;

    if (init_w != NULL)
    {
        initial_xs = (int *)malloc(num_positive * sizeof(int));
        initial_ys = (int *)malloc(num_negative * sizeof(int));

        // Calculate distances and initial solution
        for (int i = 0; i < num_positive; i++)
        {
            double distance = dot_product_row(P, i, init_w) - init_c;
            initial_xs[i] = (distance >= epsilon_P) ? 1 : 0;
            initial_reach += initial_xs[i];
        }

        for (int i = 0; i < num_negative; i++)
        {
            double distance = dot_product_row(N, i, init_w) - init_c;
            initial_ys[i] = (distance > -epsilon_N) ? 1 : 0;
        }

        printf("Initial reach = %.2f\n", initial_reach);
    }

    results->initial_reach = initial_reach;

    // Create SCIP instance
    retcode = SCIPcreate(&scip);
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to create SCIP instance");
        goto CLEANUP;
    }

    // retcode = SCIPsetRealParam(scip, "numerics/feastol", 1e-9);
    // retcode = SCIPsetRealParam(scip, "numerics/epsilon", 1e-12);

    // retcode = SCIPsetParam(scip, "heuristics/emphasis", "off");
    // retcode = SCIPsetRealParam(scip, "limits/gap", 0.01);
    // retcode = SCIPsetRealParam(scip, "constraints/initialsol/partial/unknrate", 1);

    // retcode = SCIPsetBoolParam(scip, "presolving/usesubsol", FALSE);
    // retcode = SCIPsetIntParam(scip, "emphasis/optimality", 1);

    // Include default plugins
    retcode = SCIPincludeDefaultPlugins(scip);
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to include default plugins");
        goto CLEANUP;
    }

    // Create problem
    retcode = SCIPcreateProbBasic(scip, "Wide-Reach Classification");
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to create problem");
        goto CLEANUP;
    }

    printf("reached the numerical stability part\n");

    // Set output verbosity
    // if (!print_output)
    // {
    //     retcode = SCIPsetIntParam(scip, "display/verblevel", 0);
    // }

    retcode = SCIPsetRealParam(scip, "limits/time", 120);

    // SCIPsolveConcurrent(scip);
    // retcode = SCIPsetIntParam(scip, "parallel/mode", 0);

    // Set additional parameters
    // retcode = SCIPsetParam(scip, "node");
    // retcode = SCIPsetIntParam(scip, "nodeselection/bfs/stdpriority", 250000);
    // retcode = SCIPsetIntParam(scip, "separating/maxrounds", -1);

    // Create decision variables
    SCIP_VAR **x_vars = (SCIP_VAR **)malloc(num_positive * sizeof(SCIP_VAR *));
    SCIP_VAR **y_vars = (SCIP_VAR **)malloc(num_negative * sizeof(SCIP_VAR *));
    SCIP_VAR **w_vars = (SCIP_VAR **)malloc(n_features * sizeof(SCIP_VAR *));
    SCIP_VAR *c_var;
    SCIP_VAR *V_var;

    // Create binary variables x_i for positive samples
    for (int i = 0; i < num_positive; i++)
    {
        char var_name[32];
        sprintf(var_name, "x_%d", i);
        retcode = SCIPcreateVarBasic(scip, &x_vars[i], var_name, 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY);
        if (retcode != SCIP_OKAY)
        {
            strcpy(results->error_msg, "Failed to create x variables");
            goto CLEANUP;
        }
        retcode = SCIPaddVar(scip, x_vars[i]);
    }

    // Create binary variables y_j for negative samples
    for (int j = 0; j < num_negative; j++)
    {
        char var_name[32];
        sprintf(var_name, "y_%d", j);
        retcode = SCIPcreateVarBasic(scip, &y_vars[j], var_name, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY);
        if (retcode != SCIP_OKAY)
        {
            strcpy(results->error_msg, "Failed to create y variables");
            goto CLEANUP;
        }
        retcode = SCIPaddVar(scip, y_vars[j]);
    }

    // Create continuous variables w_d for hyperplane
    for (int d = 0; d < n_features; d++)
    {
        char var_name[32];
        sprintf(var_name, "w_%d", d);
        retcode = SCIPcreateVarBasic(scip, &w_vars[d], var_name, -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
        if (retcode != SCIP_OKAY)
        {
            strcpy(results->error_msg, "Failed to create w variables");
            goto CLEANUP;
        }
        retcode = SCIPaddVar(scip, w_vars[d]);
    }

    // Create continuous variable c for bias
    retcode = SCIPcreateVarBasic(scip, &c_var, "c", -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to create c variable");
        goto CLEANUP;
    }
    retcode = SCIPaddVar(scip, c_var);

    // Create continuous variable V for precision violation
    retcode = SCIPcreateVarBasic(scip, &V_var, "V", 0.0, SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS);
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to create V variable");
        goto CLEANUP;
    }
    retcode = SCIPaddVar(scip, V_var);

    // Create precision constraint: V >= (theta-1)*sum(x_i) + theta*sum(y_j) + theta*epsilon_R
    SCIP_CONS *precision_cons;
    retcode = SCIPcreateConsBasicLinear(scip, &precision_cons, "PrecisionConstraint", 0, NULL, NULL, -SCIPinfinity(scip), SCIPinfinity(scip));

    // Add V with coefficient -1 (since we want V >= ...)
    retcode = SCIPaddCoefLinear(scip, precision_cons, V_var, -1.0);

    // Add x variables with coefficient (theta-1)
    for (int i = 0; i < num_positive; i++)
    {
        retcode = SCIPaddCoefLinear(scip, precision_cons, x_vars[i], theta - 1.0);
    }

    // Add y variables with coefficient theta
    for (int j = 0; j < num_negative; j++)
    {
        retcode = SCIPaddCoefLinear(scip, precision_cons, y_vars[j], theta);
    }

    // Set RHS to -theta*epsilon_R (since we moved V to LHS with -1 coefficient)
    SCIPchgRhsLinear(scip, precision_cons, -theta * epsilon_R);

    retcode = SCIPaddCons(scip, precision_cons);
    if (retcode != SCIP_OKAY)
    {
        strcpy(results->error_msg, "Failed to add precision constraint");
        goto CLEANUP;
    }

    // Create classification constraints for positive samples
    // x_i <= 1 + sum(w_d * X[i,d]) - c - epsilon_P
    for (int i = 0; i < num_positive; i++)
    {
        SCIP_CONS *pos_cons;
        char cons_name[32];
        sprintf(cons_name, "Positive_%d", i);

        retcode = SCIPcreateConsBasicLinear(scip, &pos_cons, cons_name, 0, NULL, NULL, -SCIPinfinity(scip), SCIPinfinity(scip));

        // Add x_i with coefficient 1
        retcode = SCIPaddCoefLinear(scip, pos_cons, x_vars[i], 1.0);

        // Add w_d variables with coefficients -X[i,d]
        for (int d = 0; d < n_features; d++)
        {
            retcode = SCIPaddCoefLinear(scip, pos_cons, w_vars[d], -get_element(X, i, d));
        }

        // Add c with coefficient 1
        retcode = SCIPaddCoefLinear(scip, pos_cons, c_var, 1.0);

        // Set RHS to 1 - epsilon_P
        SCIPchgRhsLinear(scip, pos_cons, 1.0 - epsilon_P);

        retcode = SCIPaddCons(scip, pos_cons);
        SCIPreleaseCons(scip, &pos_cons);
    }

    // Create classification constraints for negative samples
    // y_j >= sum(w_d * X[j,d]) - c + epsilon_N
    for (int j = 0; j < num_negative; j++)
    {
        SCIP_CONS *neg_cons;
        char cons_name[32];
        sprintf(cons_name, "Negative_%d", j);

        retcode = SCIPcreateConsBasicLinear(scip, &neg_cons, cons_name, 0, NULL, NULL, -SCIPinfinity(scip), SCIPinfinity(scip));

        // Add y_j with coefficient -1 (since we want y_j >= ...)
        retcode = SCIPaddCoefLinear(scip, neg_cons, y_vars[j], -1.0);

        // Add w_d variables with coefficients X[num_positive+j,d]
        for (int d = 0; d < n_features; d++)
        {
            retcode = SCIPaddCoefLinear(scip, neg_cons, w_vars[d], get_element(X, num_positive + j, d));
        }

        // Add c with coefficient -1
        retcode = SCIPaddCoefLinear(scip, neg_cons, c_var, -1.0);

        // Set RHS to -epsilon_N
        SCIPchgRhsLinear(scip, neg_cons, -epsilon_N);

        retcode = SCIPaddCons(scip, neg_cons);
        SCIPreleaseCons(scip, &neg_cons);
    }

    // Set objective: maximize sum(x_i) - lambda_param * V
    retcode = SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE);

    // Set objective coefficients
    for (int i = 0; i < num_positive; i++)
    {
        retcode = SCIPchgVarObj(scip, x_vars[i], 1.0);
    }
    retcode = SCIPchgVarObj(scip, V_var, -lambda_param);

    // Write problem to file
    char filename[256];
    sprintf(filename, "%s.lp", dataset_name);
    retcode = SCIPwriteOrigProblem(scip, filename, "lp", FALSE);

    const char *SCIP_HEURISTICS[] = {
        "rens",
        "rounding",
        "fracdiving",
        "octane",
        "feaspump",
        "localbranching",
        "rins",
        "dins",
        "crossover",
        "intshifting",
        NULL // Sentinel value to mark end of array
    };

    // const char* important_heuristics[] = {
    //     "alns",
    //     "completesol",
    //     "conflictdiving",
    //     "dualval",
    //     "locks",
    //     "lpface",
    //     "mutation",
    //     // "intshifting",
    //     "objpscostdiving",
    //     "proximity",
    //     "randrounding",
    //     "repair",
    // "scheduler",
    //     "shifting",
    //     "trivial",
    //     "trustregion",
    //     "zeroobj",
    //     "zirounding",
    //     NULL
    // };

    const char *important_heuristics[] = {
        // "adaptivediving",
        // "alns",
        // "intshifting",
        // "clique",
        "completesol",
        // "conflictdiving",
        // // "crossover",
        // "distributiondivi",
        // // "feaspump",
        // // "fracdiving",
        // "guideddiving",
        // "intshifting",
        // "linesearchdiving",
        // "locks",
        // "lpface",
        // "objpscostdiving",
        // "oneopt",
        // "pscostdiving",
        // "randrounding",
        // // "rens",
        // // "rins",
        // "rootsoldiving",
        // "rounding",
        // "shifting",
        // "veclendiving",
        // "scheduler",
        // "zirounding",
        NULL};

    if (run)
    {

        // disable other heuristics
        int disable = 1;

        if (disable)
        {
            // SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, 0);
            SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, 0);
            // Get all heuristic plugins from SCIP
            SCIP_HEUR **heurs = SCIPgetHeurs(scip);
            int nheurs = SCIPgetNHeurs(scip);

            for (int i = 0; i < nheurs; ++i)
            {
                const char *heur_name = SCIPheurGetName(heurs[i]);

                //     int allow = 0;
                //     for (int j = 0; SCIP_HEURISTICS[j] != NULL; ++j)
                //     {
                //         if (strcmp(SCIP_HEURISTICS[j], heur_name) == 0)
                //         {
                //             allow = 1;
                //             break;
                //         }
                //     }
                //     int allow_important = 0;

                
                //     for (int j = 0; important_heuristics[j] != NULL; ++j)
                //     {
                //         if (strcmp(important_heuristics[j], heur_name) == 0)
                //         {
                //             allow_important = 1;
                //             break;
                //         }
                //     }

                // if (!allow && allow_important)
                // {
                char paramname[SCIP_MAXSTRLEN];

                // frequency
                snprintf(paramname, sizeof(paramname), "heuristics/%s/freq", heur_name);
                retcode = SCIPsetIntParam(scip, paramname, -1);

                // snprintf(paramname, sizeof(paramname), "heuristics/%s/priority", heur_name);
                // retcode = SCIPsetIntParam(scip, paramname, -100000000);

                // }
            }

            // int target_heuristic_index = 4;

            // char paramname[SCIP_MAXSTRLEN];
            // snprintf(paramname, sizeof(paramname), "heuristics/%s/freq", SCIP_HEURISTICS[target_heuristic_index]);
            // retcode = SCIPsetIntParam(scip, paramname, 1);

            // snprintf(paramname, sizeof(paramname), "heuristics/%s/priority", SCIP_HEURISTICS[target_heuristic_index]);
            // retcode = SCIPsetIntParam(scip, paramname, 1);

            // snprintf(paramname, sizeof(paramname), "heuristics/%s/freqofs", SCIP_HEURISTICS[target_heuristic_index]);
            // retcode = SCIPsetIntParam(scip, paramname, 0);

            // snprintf(paramname, sizeof(paramname), "heuristics/%s/maxdepth", SCIP_HEURISTICS[target_heuristic_index]);
            // retcode = SCIPsetIntParam(scip, paramname, -1);

            // SCIP_HEUR *heur = SCIPfindHeur(scip, SCIP_HEURISTICS[target_heuristic_index]);
            // SCIPheurSetTimingmask(heur, SCIP_HEURTIMING_BEFORENODE + SCIP_HEURTIMING_DURINGLPLOOP + SCIP_HEURTIMING_AFTERLPLOOP + SCIP_HEURTIMING_AFTERLPNODE + SCIP_HEURTIMING_AFTERPSEUDONODE + SCIP_HEURTIMING_AFTERLPPLUNGE + SCIP_HEURTIMING_AFTERPSEUDOPLUNGE);
        }
        else
        {
            SCIPsetHeuristics(scip, SCIP_PARAMSETTING_AGGRESSIVE, 0);
        }

        // retcode = SCIPsetIntParam(scip, "heuristics/scheduler/freq", 1);
        // retcode = SCIPsetIntParam(scip, "heuristics/intshifting/freq", 1);
        // retcode = SCIPsetIntParam(scip, "heuristics/alns/freq", 1);
        retcode = SCIPsetIntParam(scip, "heuristics/intshifting/freq", 1);
        // retcode = SCIPsetIntParam(scip, "heuristics/completesol/freq", 1);
        // retcode = SCIPsetBoolParam(scip, "heuristics/feaspump/usefp20", TRUE);
        // retcode = SCIPsetRealParam(scip, "heuristics/completesol/maxunknownrate", 1);
        // retcode = SCIPsetBoolParam(scip, "heuristics/completesol/addallsols", TRUE);

        // Create and set partial solution if initial values are provided
        int USE_SOLUTION = 1;
        if (init_w != NULL && initial_xs != NULL && initial_ys != NULL && USE_SOLUTION)
        {
            SCIP_SOL *partial_sol;
            SCIP_Bool partial;

            // printf("Applying initial solution with reach=%.2f\n", initial_reach);
            // printf("hello aarish\n");

            // Create partial solution
            // retcode = SCIPcreatePartialSol(scip, &partial_sol, NULL);
            // printf("check 1\n");
            // retcode = SCIPcreateSol(scip, &partial_sol, NULL);
            retcode = SCIPcreatePartialSol(scip, &partial_sol, NULL);
            // retcode = SCIPcreateSol(scip, &partial_sol, NULL);
            // retcode = SCIPcreatePartialSol(scip, &partial_sol, NULL);
            // printf("check 2\n");

            SCIP_Bool stored;

            if (retcode != SCIP_OKAY)
            {
                printf("Warning: Could not create partial solution\n");
            }
            else
            {
                // // Set x variables in partial solution
                for (int i = 0; i < num_positive; i++)
                {
                    retcode = SCIPsetSolVal(scip, partial_sol, x_vars[i], (double)initial_xs[i]);
                    if (retcode != SCIP_OKAY)
                    {
                        printf("Warning: Could not set x[%d] in partial solution\n", i);
                    }
                }

                // Set y variables in partial solution
                for (int j = 0; j < num_negative; j++)
                {
                    retcode = SCIPsetSolVal(scip, partial_sol, y_vars[j], (double)initial_ys[j]);
                    if (retcode != SCIP_OKAY)
                    {
                        printf("Warning: Could not set y[%d] in partial solution\n", j);
                    }
                }

                // // Set w variables in partial solution
                // for (int d = 0; d < n_features; d++)
                // {
                //     retcode = SCIPsetSolVal(scip, partial_sol, w_vars[d], init_w[d]);
                //     if (retcode != SCIP_OKAY)
                //     {
                //         printf("Warning: Could not set w[%d] in partial solution\n", d);
                //     }
                // }

                // // Set c variable in partial solution
                // retcode = SCIPsetSolVal(scip, partial_sol, c_var, init_c);
                // if (retcode != SCIP_OKAY)
                // {
                //     printf("Warning: Could not set c in partial solution\n");
                // }

                // // // Calculate and set V variable in partial solution
                // double v_val = 0.0;
                // double temp_sum = 0.0;
                // for (int i = 0; i < num_positive; i++)
                // {
                //     temp_sum += initial_xs[i];
                // }
                // temp_sum *= (theta - 1.0);

                // for (int j = 0; j < num_negative; j++)
                // {
                //     temp_sum += theta * initial_ys[j];
                // }
                // temp_sum += theta * epsilon_R;

                // v_val = fmax(0.0, temp_sum);
                // retcode = SCIPsetSolVal(scip, partial_sol, V_var, v_val);
                // if (retcode != SCIP_OKAY)
                // {
                //     printf("Warning: Could not set V in partial solution\n");
                // }

                // Try to add the partial solution
                // SCIP_Bool stored;
                // retcode = SCIPaddSolFree(scip, &partial_sol, &stored);
                // SCIPtrySolFree(scip, SCIP_SOL **sol, unsigned int printreason, unsigned int completely, unsigned int checkbounds, unsigned int checkintegrality, unsigned int checklprows, unsigned int *stored)
                // retcode = SCIPtrySolFree(scip, &partial_sol, TRUE, TRUE,
                // FALSE, FALSE, FALSE, &stored);

                retcode = SCIPaddSolFree(scip, &partial_sol, &stored);
                // retcode = SCIPaddSol(scip, partial_sol, 0);

                // retcode = SCIPtrySol(scip, partial_sol,
                //                          TRUE,  // not check feasibility
                //                          FALSE,   // check bounds
                //                          FALSE,   // check domain
                //                          FALSE,   // print reason
                //                         //  SCIPprintTreeStatistics(SCIP *scip, FILE *file)
                //                         FALSE,
                //                          &stored);

                // retcode = SCIPaddSol(scip, partial_sol, 0);
            }
        }

        // Solve the problem
        clock_t start_time = clock();
        retcode = SCIPsolve(scip);
        clock_t end_time = clock();

        results->time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        if (retcode != SCIP_OKAY)
        {
            strcpy(results->error_msg, "Optimization failed");
            goto CLEANUP;
        }

        SCIP_STATUS status = SCIPgetStatus(scip);

        if (status == SCIP_STATUS_OPTIMAL || status == SCIP_STATUS_TIMELIMIT)
        {
            SCIP_SOL *sol = SCIPgetBestSol(scip);

            SCIPprintStatistics(scip, NULL);

            if (sol != NULL)
            {
                results->status = (status == SCIP_STATUS_OPTIMAL) ? 0 : 1;

                // Extract reach
                results->reach = 0.0;
                for (int i = 0; i < num_positive; i++)
                {
                    results->reach += SCIPgetSolVal(scip, sol, x_vars[i]);
                }

                // Extract hyperplane parameters
                results->hyperplane_w = (double *)malloc(n_features * sizeof(double));
                for (int d = 0; d < n_features; d++)
                {
                    results->hyperplane_w[d] = SCIPgetSolVal(scip, sol, w_vars[d]);
                }

                // Extract bias
                results->bias_c = SCIPgetSolVal(scip, sol, c_var);

                // Extract x values
                results->x_vals = (double *)malloc(num_positive * sizeof(double));
                for (int i = 0; i < num_positive; i++)
                {
                    results->x_vals[i] = SCIPgetSolVal(scip, sol, x_vars[i]);
                }

                // Extract y values
                results->y_vals = (double *)malloc(num_negative * sizeof(double));
                for (int j = 0; j < num_negative; j++)
                {
                    results->y_vals[j] = SCIPgetSolVal(scip, sol, y_vars[j]);
                }

                // Extract precision violation
                results->precision_violation_v = SCIPgetSolVal(scip, sol, V_var);

                // Extract node count
                results->node_count = SCIPgetNNodes(scip);
            }
            else
            {
                strcpy(results->error_msg, "No solution found");
                results->status = -1;
            }
        }
        else
        {
            sprintf(results->error_msg, "No optimal solution found. Status: %d", status);
            results->status = -1;
        }
    }

CLEANUP:
    // Release constraints
    SCIPreleaseCons(scip, &precision_cons);

    // Release variables
    if (x_vars)
    {
        for (int i = 0; i < num_positive; i++)
        {
            if (x_vars[i])
                SCIPreleaseVar(scip, &x_vars[i]);
        }
        free(x_vars);
    }

    if (y_vars)
    {
        for (int j = 0; j < num_negative; j++)
        {
            if (y_vars[j])
                SCIPreleaseVar(scip, &y_vars[j]);
        }
        free(y_vars);
    }

    if (w_vars)
    {
        for (int d = 0; d < n_features; d++)
        {
            if (w_vars[d])
                SCIPreleaseVar(scip, &w_vars[d]);
        }
        free(w_vars);
    }

    if (c_var)
        SCIPreleaseVar(scip, &c_var);
    if (V_var)
        SCIPreleaseVar(scip, &V_var);

    // Free SCIP instance
    if (scip)
    {
        SCIPfree(&scip);
    }

    // Free temporary data
    free_matrix(X);
    if (initial_xs)
        free(initial_xs);
    if (initial_ys)
        free(initial_ys);

    return results;
}

// Function to free solver results
void free_solver_results(SolverResults *results)
{
    if (results)
    {
        if (results->hyperplane_w)
            free(results->hyperplane_w);
        if (results->x_vals)
            free(results->x_vals);
        if (results->y_vals)
            free(results->y_vals);
        free(results);
    }
}

// Example usage function
int main()
{
    // Example usage - you would populate these with your actual data
    Matrix *P = create_matrix(10, 5); // 10 positive samples, 5 features
    Matrix *N = create_matrix(15, 5); // 15 negative samples, 5 features

    // Initialize with random data (replace with your actual data)
    srand(42);
    for (int i = 0; i < P->rows * P->cols; i++)
    {
        P->data[i] = ((rand() % 1000) / 1000.0) * 2.0 - 1.0;
    }
    for (int i = 0; i < N->rows * N->cols; i++)
    {
        N->data[i] = ((rand() % 1000) / 1000.0) * 2.0 - 1.0;
    }

    // Initial hyperplane parameters
    double init_w[] = {0.1, 0.2, -0.1, 0.3, -0.2};
    double init_c = 0.5;

    // Solve
    SolverResults *results = scip_solver(
        0.8,    // theta
        0.1,    // theta0
        0.2,    // theta1
        P,      // positive samples
        N,      // negative samples
        0.01,   // epsilon_P
        0.01,   // epsilon_N
        0.001,  // epsilon_R
        -1.0,   // lambda_param (will be auto-calculated)
        "test", // dataset_name
        1,      // run
        init_w, // initial w
        init_c, // initial c
        1,      // print_output
        300.0   // time_limit
    );

    // Print results
    if (results->status >= 0)
    {
        printf("Optimization completed successfully!\n");
        printf("Initial reach: %.2f\n", results->initial_reach);
        printf("Final reach: %.2f\n", results->reach);
        printf("Bias c: %.6f\n", results->bias_c);
        printf("Precision violation V: %.6f\n", results->precision_violation_v);
        printf("Node count: %ld\n", results->node_count);
        printf("Time taken: %.2f seconds\n", results->time_taken);
        printf("Status: %s\n", results->status == 0 ? "Optimal" : "Time limit");
    }
    else
    {
        printf("Error: %s\n", results->error_msg);
    }

    // Cleanup
    free_solver_results(results);
    free_matrix(P);
    free_matrix(N);

    return 0;
}
