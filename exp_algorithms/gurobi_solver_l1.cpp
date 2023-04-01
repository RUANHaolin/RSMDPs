//
//  gurobi_solver_l1.cpp
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#include <stdio.h>
#include "definitions.h"
#include "gurobi_c++.h"
#include "gurobi_solver_l1.h"

#include <iomanip>
#include <chrono>
#include <random>
#include <sstream>


pair<numvec, double> srect_solve_gurobi_l1(const BellmanEq_s& instanceS) {
    const vector<numvec>& z     = instanceS.b;
    const vector<numvec>& pbar  = instanceS.p_bar;
    const prec_t&         kappa = instanceS.kappa;
    const vector<numvec>& w     = instanceS.sigma;
    
    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();
    
    assert(pbar.size() == z.size());
    assert(w.empty() || w.size() == z.size());
    
    // helpful numbers of actions
    const size_t nAction = pbar.size();
    // number of transition states for each action
    vector<size_t> statecounts(nAction);
    transform(pbar.cbegin(), pbar.cend(), statecounts.begin(), [](const numvec& v) {return v.size();});
    // the number of states per action does not need to be the same
    // (when transitions are sparse)
    const size_t nstateactions = accumulate(statecounts.cbegin(), statecounts.cend(), size_t(0));
    
    /*
    // This is used to debug the problem with Gurboi, it seems to be a license problem
    try {
        GRBModel model = GRBModel(env);
        //GRBModel model = GRBModel(env, argv[1]);
    } catch (GRBException e) {std::cout << e.getMessage(); // information from length_error is lost
    }
    return pair<numvec, double>(instanceS.b[0],0);
    */
    
    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();
    
    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    // Create varables: duals of the nature problem
    auto x = unique_ptr<GRBVar[]>(model.addVars(numvec(nAction, -inf).data(), nullptr,
                                                nullptr,
                                                vector<char>(nAction, GRB_CONTINUOUS).data(),
                                                nullptr, int(nAction)));
    //  outer loop: actions, inner loop: next state
    auto yp = unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
                                                 nullptr,
                                                 vector<char>(nstateactions, GRB_CONTINUOUS).data(),
                                                 nullptr, int(nstateactions)));
    auto yn = unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
                                                 nullptr,
                                                 vector<char>(nstateactions, GRB_CONTINUOUS).data(),
                                                 nullptr, int(nstateactions)));
    
    auto lambda = model.addVar(0, inf, -kappa, GRB_CONTINUOUS, "lambda");
    
    // primal variables for the nature
    auto d = unique_ptr<GRBVar[]>(model.addVars(numvec(nAction, 0).data(), nullptr,
                                                numvec(nAction, 0).data(),
                                                vector<char>(nAction, GRB_CONTINUOUS).data(),
                                                nullptr, int(nAction)));
    // objective
    GRBLinExpr objective;
    
    size_t i = 0;
    // constraints dual to variables of the inner problem
    for (size_t actionid = 0; actionid < nAction; actionid++) {
        objective += x[actionid];
        for (size_t stateid = 0; stateid < statecounts[actionid]; stateid++) {
            // objective
            objective += -pbar[actionid][stateid] * yp[i];
            objective += pbar[actionid][stateid] * yn[i];
            // dual for p
            model.addConstr(x[actionid] - yp[i] + yn[i] <= d[actionid] * z[actionid][stateid]);
            // dual for z
            double weight = w.size() > 0 ? w[actionid][stateid] : 1.0;
            model.addConstr(-lambda * weight + yp[i] + yn[i] <= 0);
            // update the counter (an absolute index for each variable)
            i++;
        }
    }
    objective += -lambda * kappa;
    
    // constraint on the policy pi
    GRBLinExpr ones;
    ones.addTerms(numvec(nAction, 1.0).data(), d.get(), int(nAction));
    model.addConstr(ones, GRB_EQUAL, 1);
    
    // set objective
    model.setObjective(objective, GRB_MAXIMIZE);
    
    // run optimization
    model.optimize();
    
    // retrieve policy values
    numvec policy(nAction);
    for (size_t i = 0; i < nAction; i++) {
        policy[i] = d[i].get(GRB_DoubleAttr_X);
    }
    
    // retrieve the worst-case response values
    return {policy, model.get(GRB_DoubleAttr_ObjVal)};
}


