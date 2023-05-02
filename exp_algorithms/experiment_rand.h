//
//  experiment_rand.h
//  PDA_submit
//
//  Created by Haolin on 31/3/2023
//

#ifndef experiment_rand_h
#define experiment_rand_h
#include "definitions.h"

RSMDPs GenRandInstance_rsmdps(size_t nStates, size_t nActions);


tuple<prec_t,prec_t,prec_t, prec_t,prec_t,prec_t, prec_t,prec_t,prec_t, prec_t> get_speed_gurobi(RSMDPs& prob, const size_t nStates, const size_t instance_num);


void runsave_rsmdps_speed(const function<RSMDPs(size_t nStates, size_t nActions)>& prob_gen, const sizvec nStates_ls, const size_t repetitions);


prec_t get_speed_gurobi_RMDP(RSMDPs& prob, const size_t nStates);


void run_rmdps_speed(const function<RSMDPs(size_t nStates, size_t nActions)>& prob_gen, const sizvec nStates_ls, const size_t repetitions);





#endif /* experiment_rand_h */


