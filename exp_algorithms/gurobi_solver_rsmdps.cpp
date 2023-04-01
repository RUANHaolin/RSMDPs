//
//  gurobi_solver_rsmdps.cpp
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#include <stdio.h>
#include "definitions.h"
#include "gurobi_c++.h"

#include <iomanip>
#include <chrono>
#include <random>
#include <sstream>

/**
 * Solve the rsmdps using gurobi linear solver, problem (14) for the version on 27-04-2022
 */
pair<vector<numvec>, double> srect_solve_gurobi_rsmdps(const RSMDPs& prob) {

    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();

    const vector<vector<numvec>>& P = prob.P;
    const vector<numvec>& r = prob.r;
    const numvec&         d = prob.d;
    const prec_t&     gamma = prob.gamma;
    const numvec&         w = prob.w;
    const prec_t&       tau = prob.tau;

    const size_t nStates  = P.size();
    const size_t nActions = P[0].size();

    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);


    /*********************************************************************
             Create varables
             see https://support.gurobi.com/hc/en-us/community/posts/4414331624849-Create-multi-dimensional-variables-read-data-file-in-C-
    *********************************************************************/

    GRBVar** u;
    u = new GRBVar * [nStates];
    for (size_t s = 0; s < nStates; s++) {
        u[s] = model.addVars(nActions, GRB_CONTINUOUS);
        for (size_t a = 0; a < nActions; a++) {
            u[s][a].set(GRB_DoubleAttr_LB, 0.0);
            u[s][a].set(GRB_DoubleAttr_UB, inf);
        }
    }

    GRBVar*** alpha;
    alpha = new GRBVar * *[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        alpha[s1] = new GRBVar * [nStates];
        for (size_t s2 = 0; s2 < nStates; s2++) {
            alpha[s1][s2] = model.addVars(nActions, GRB_CONTINUOUS);
            for (size_t a = 0; a < nActions; a++) {
                alpha[s1][s2][a].set(GRB_DoubleAttr_LB, -inf);
                alpha[s1][s2][a].set(GRB_DoubleAttr_UB, inf);
            }
        }
    }

    GRBVar**** beta;
    GRBVar**** y;
    GRBVar**** y_abs;
    beta = new GRBVar * **[nStates];
    y = new GRBVar * **[nStates];
    y_abs = new GRBVar * **[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        beta[s1] = new GRBVar * *[nStates];
        y[s1] = new GRBVar * *[nStates];
        y_abs[s1] = new GRBVar * *[nStates];
        for (size_t s2 = 0; s2 < nStates; s2++) {
            beta[s1][s2] = new GRBVar * [nActions];
            y[s1][s2] = new GRBVar * [nActions];
            y_abs[s1][s2] = new GRBVar * [nActions];
            for (size_t a = 0; a < nActions; a++) {
                beta[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);
                y[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);
                y_abs[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);

                for (size_t s3 = 0; s3 < nStates; s3++) {
                    beta[s1][s2][a][s3].set(GRB_DoubleAttr_LB, 0.0);
                    beta[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                    y[s1][s2][a][s3].set(GRB_DoubleAttr_LB, -inf);
                    y[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                    y_abs[s1][s2][a][s3].set(GRB_DoubleAttr_LB, 0.0);
                    y_abs[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                }
            }
        }
    }

    auto y_norm1 = unique_ptr<GRBVar[]>(model.addVars(numvec(nStates, 0.0).data(), nullptr,
        nullptr,
        vector<char>(nStates, GRB_CONTINUOUS).data(),
        nullptr, int(nStates)));


    /*********************************************************************
             Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    GRBLinExpr sum_r_u;
    vector<GRBLinExpr> sum_u_s(nStates);             // e^\top u_s
    vector<GRBLinExpr> sum_alpha_s(nStates);         // e^\top alpha_s
    vector<GRBLinExpr> sum_p_hat_y_s(nStates);       // hat{p}^\top y_s
    vector<GRBLinExpr> sum_y_abs_s(nStates);         // sum of y_abs

    // 1st for-loop to handle the "for every s constraint"
    for (size_t s = 0; s < nStates; s++) {

        for (size_t a = 0; a < nActions; a++) {
            sum_r_u += r[s][a] * u[s][a];
            sum_u_s[s] += u[s][a];
        }

        for (size_t s2 = 0; s2 < nStates; s2++) {
            for (size_t a = 0; a < nActions; a++) {
                sum_alpha_s[s] += alpha[s][s2][a];
                for (size_t s3 = 0; s3 < nStates; s3++) {
                    sum_p_hat_y_s[s] += P[s2][a][s3] * y[s][s2][a][s3];

                    if (s == s3) {
                        model.addConstr(y[s][s2][a][s3] == beta[s][s2][a][s3] - (gamma * u[s2][a]) - alpha[s][s2][a]);
                    }
                    else {
                        model.addConstr(y[s][s2][a][s3] == beta[s][s2][a][s3] - alpha[s][s2][a]);
                    }

                    model.addConstr( y[s][s2][a][s3] <= y_abs[s][s2][a][s3]);
                    model.addConstr(-y[s][s2][a][s3] <= y_abs[s][s2][a][s3]);

                    sum_y_abs_s[s] += y_abs[s][s2][a][s3];
                }
            }
        }

        model.addConstr(sum_u_s[s] - d[s] <= - sum_alpha_s[s] - sum_p_hat_y_s[s]);
        model.addConstr(sum_y_abs_s[s] <= y_norm1[s]);
        objective += w[s] * y_norm1[s];
    }

    model.addConstr(sum_r_u >= tau);

    // set objective
    model.setObjective(objective, GRB_MINIMIZE);
    
    // run optimization
    model.optimize();

    // retrieve policy values
    vector<numvec> u_star;
    for (size_t s = 0; s < nStates; s++) {
        numvec pi_s;
        for (size_t a = 0; a < nActions; a++) {
            pi_s.push_back(u[s][a].get(GRB_DoubleAttr_X));
        }
        u_star.push_back(pi_s);
    }


    // retrieve the worst-case response values
    return { u_star, model.get(GRB_DoubleAttr_ObjVal) };
}



/**
 * Solve the mdps using gurobi linear solver
 */
pair<vector<numvec>, double> srect_solve_gurobi_mdps(const RSMDPs& prob) {

    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();

    const vector<vector<numvec>>& P = prob.P;
    const vector<numvec>& r = prob.r;
    const numvec& d = prob.d;
    const prec_t& gamma = prob.gamma;

    const size_t nStates = P.size();
    const size_t nActions = P[0].size();

    // //GRBEnv env = GRBEnv(true);
    //GRBEnv env = GRBEnv();
    //GRBEnv* env = 0;
    //env = new GRBEnv();
    // GRBEnv env = GRBEnv();
    //env.set(GRB_IntParam_OutputFlag, 0);
    //env.set(GRB_IntParam_Threads, 1);
    //GRBModel model = GRBModel(env);
    
    // make sure it is run in a single thread for a fair comparison
    //env.set(GRB_IntParam_OutputFlag, 0);
    //env.set(GRB_IntParam_Threads, 1);
    //env.start();

    // construct the LP model
    //GRBModel model = GRBModel(env);
    //GRBModel model = GRBModel(*env);
    //model.getEnv().set(GRB_IntParam_OutputFlag, 0);
    //model.getEnv().set(GRB_IntParam_Threads, 1);

    GRBEnv env = GRBEnv(true);
    //GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    
    env.start();
    
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    /*********************************************************************
             Create varables
             see https://support.gurobi.com/hc/en-us/community/posts/4414331624849-Create-multi-dimensional-variables-read-data-file-in-C-
    *********************************************************************/

    GRBVar** u;
    u = new GRBVar * [nStates];
    for (size_t s = 0; s < nStates; s++) {
        u[s] = model.addVars(nActions, GRB_CONTINUOUS);
        for (size_t a = 0; a < nActions; a++) {
            u[s][a].set(GRB_DoubleAttr_LB, 0.0);
            u[s][a].set(GRB_DoubleAttr_UB, inf);
        }
    }

    /*********************************************************************
                          Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    vector<GRBLinExpr> sum_u_s (nStates);             // e^\top u_s
    vector<GRBLinExpr> sum_p_hat_y_s (nStates);       // hat{p}^\top u

    for (size_t s = 0; s < nStates; s++) {

        for (size_t a = 0; a < nActions; a++) {
            objective  += r[s][a] * u[s][a];
            sum_u_s[s] += u[s][a];

            for (size_t s2 = 0; s2 < nStates; s2++) {
                sum_p_hat_y_s[s2] += gamma * P[s][a][s2] * u[s][a];
            }
        }
    }

    for (size_t s = 0; s < nStates; s++) {
        model.addConstr(sum_u_s[s] - sum_p_hat_y_s[s] <= d[s]);
    }

    // set objective
    model.setObjective(objective, GRB_MAXIMIZE);

    // run optimization
    model.optimize();

    // retrieve policy values
    vector<numvec> policy;
    for (size_t s = 0; s < nStates; s++) {
        numvec pi_s;
        for (size_t a = 0; a < nActions; a++) {
            pi_s.push_back(u[s][a].get(GRB_DoubleAttr_X));
        }
        policy.push_back(pi_s);
    }


    // retrieve the worst-case response values
    return { policy, model.get(GRB_DoubleAttr_ObjVal) };
}


