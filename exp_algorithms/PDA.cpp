//
//  PDA.cpp
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#include <stdio.h>
#include "definitions.h"
#include "gurobi_solver_rsmdps.h"
#include "PDA.h"

#include "gurobi_c++.h"
#include <random>

#include <chrono> // modification


// this is a function that samples s1,s2,a1, and a RV from Uni(0,1)
tuple<size_t,size_t,size_t,prec_t> sampling_SSA_RV(const RSMDPs& prob){
    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());   // comment this if you need the same instance all the time.
    uniform_int_distribution<size_t> distributionS(0, prob.nStates-1);
    uniform_int_distribution<size_t> distributionA(0, prob.nActions-1);
    uniform_real_distribution<double> distributionR(0.0, 1.0);

    size_t s_sample1 = distributionS(generator);
    size_t s_sample2 = distributionS(generator);
    size_t a_sample  = distributionA(generator);

    prec_t probility = distributionR(generator);


    return {s_sample1,s_sample2,a_sample,probility};
}


// This is a solver that solve the following problem, given z and other parameters
// min   z^T u + (1/2 nu) * || u - u_hat ||^2
// s.t   r^T u >= tau
//           u >= 0
void PDA_primal_update(const RSMDPs& prob, PDA& pda) {
    const prec_t smallEps = 1e-14;
    const prec_t Infty    = numeric_limits<prec_t>::infinity();

    // update z_p based on current dual variables
    pda.update_coeff_primal(prob);
    const vector<numvec>& z = pda.z_primal;
    
    const vector<numvec>& u_hat = pda.u0;
    const prec_t& nu            = pda.nu;
    const prec_t& tau           = prob.tau;

    const size_t SA = prob.nStates * prob.nActions;

    assert(prob.nStates  == z.size());
    assert(prob.nActions == z[0].size());
    assert(prob.nStates  == pda.u0.size());
    assert(prob.nActions == pda.u0[0].size());

    // check if r^T u >= tau is necessary by checking the solution without this constraint
    // compute the breakpoints for alpha, for future use
    auto u_soln = pda.get_zeros_SA();
    numvec alpha_bkpt_ls; alpha_bkpt_ls.reserve(SA);
    vector<sizvec> SA_pos_ls; SA_pos_ls.reserve(SA);
    prec_t ru_temp = 0.0;
    for (size_t s1 = 0; s1 < prob.nStates; s1++) {
        for (size_t a = 0; a < prob.nActions; a++) {
            prec_t r_sa = prob.r[s1][a];

            prec_t u_i  = max( u_hat[s1][a] - (nu * z[s1][a]) , 0.0);
            u_soln[s1][a] = u_i;
            ru_temp += r_sa * u_i;
            
            // following: computing breakpoints for next step
            prec_t alpha_i = (r_sa > smallEps) ? (1.0 / r_sa) * (z[s1][a] - (u_hat[s1][a] / nu)) : Infty;
            alpha_bkpt_ls.push_back(alpha_i);
            SA_pos_ls.push_back({s1,a});
        }
    }
    if (ru_temp >= tau) {
        pda.u = u_soln;
        return;
    }

    // we now assume r^T u == tau, first sort alpha_i
    sizvec idx        = sort_indexes_ascending(alpha_bkpt_ls);
    prec_t alpha_star = Infty;

    // check piece by peice
    prec_t sum_r_u    = 0.0;
    prec_t sum_nu_r2  = 0.0;
    prec_t sum_r_nu_z = 0.0;
    bool find_alpha = false;
    for (size_t i = 0; i < SA; i++) {
        size_t idx_t = idx[i];
        size_t idx_s = SA_pos_ls[idx_t][0];
        size_t idx_a = SA_pos_ls[idx_t][1];
        //size_t idx_s = idx_t / prob.nActions;
        //size_t idx_a = idx_t % prob.nActions;

        // set alpha interval
        prec_t alpha_low = alpha_bkpt_ls[idx_t];
        prec_t alpha_up = (i < SA - 1) ? alpha_bkpt_ls[idx[i + 1]] : Infty;
        
        // update the terms
        prec_t r_sa = prob.r[idx_s][idx_a];
        assert(r_sa > smallEps);
        prec_t z_sa = z[idx_s][idx_a];
        prec_t u_hat_sa = u_hat[idx_s][idx_a];

        sum_r_u    += r_sa * u_hat_sa;
        sum_nu_r2  += nu * r_sa * r_sa;
        sum_r_nu_z += r_sa * nu * z_sa;
        prec_t alpha_temp = (tau + sum_r_nu_z - sum_r_u) / sum_nu_r2;

        if (alpha_low - smallEps <= alpha_temp && alpha_temp <= alpha_up + smallEps) {
            alpha_star = alpha_temp;
            find_alpha = true;
            break;
        }
    }
    assert(find_alpha == true);

    // compute the solution with alpha_star
    for (size_t s1 = 0; s1 < prob.nStates; s1++) {
        for (size_t a = 0; a < prob.nActions; a++) {
            prec_t r_sa = prob.r[s1][a];
            prec_t z_sa = z[s1][a];

            prec_t u1 = max(u_hat[s1][a] + nu * ((alpha_star * r_sa) - z_sa), 0.0);
            prec_t u2 = max(u_hat[s1][a] - (nu * z_sa), 0.0);
            prec_t u_i = (r_sa > smallEps) ? u1 : u2;

            u_soln[s1][a] = u_i;
        }
    }
    
    pda.u = u_soln;
    return;
}


// This is a solver that solve the following problem
// min   z^T theta + (1/2 sigma) * || theta - theta_hat ||^2
// s.t   e^T theta = lambda_p
//    theta_low <= theta <= theta_up
prec_t solver_BoxCon_QP(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, const size_t& s_hat, const size_t& s_prime, const size_t& a_prime, bool update_theta) {

    const prec_t smallEps = 1e-13;
    const prec_t Infty = numeric_limits<prec_t>::infinity();

    // update parameters if they haven't been updated.
    if (pda.algType == PDA_block_plus){
        pda.update_coeff_dual_z_ssa(prob, s_hat, s_prime, a_prime);
    }
    // compute theta_low and theta_up
    const auto [theta_low, theta_up] = pda.compute_theta_bound_ssa(prob, s_hat, s_prime, a_prime, lambda_p);
    
    const numvec& z          = pda.z_dual[s_hat][s_prime][a_prime];
    const prec_t& sigma      = pda.sigma;
    const numvec& theta_hat  = pda.theta0[s_hat][s_prime][a_prime];
    const size_t nStates     = z.size();
    
    // check feasibility, and if there is easy case
    prec_t sum_theta_low = 0.0;
    prec_t sum_theta_up = 0.0;
    prec_t f_low = 0.0;
    prec_t f_up = 0.0;
    
    for (size_t s = 0; s < nStates; s++) {
        sum_theta_low += theta_low[s];
        sum_theta_up  += theta_up[s];

        f_low += (theta_low[s] * z[s]) + ((0.5 / sigma) * (theta_low[s] - theta_hat[s]) * (theta_low[s] - theta_hat[s]));
        f_up  += (theta_up[s]  * z[s]) + ((0.5 / sigma) * (theta_up[s]  - theta_hat[s]) * (theta_up[s]  - theta_hat[s]));
    }
    
    assert(sum_theta_low <= lambda_p + smallEps && lambda_p <= sum_theta_up + smallEps);
    if (abs(sum_theta_low - lambda_p) < smallEps) {
        if (update_theta) { pda.theta[s_hat][s_prime][a_prime] = theta_low; }
        cout << "HEHEHE" << endl;
        return f_low;
    }
    else if (abs(sum_theta_up - lambda_p) < smallEps) {
        if (update_theta) { pda.theta[s_hat][s_prime][a_prime] = theta_up; }
        cout << "HAHAHA" << endl;
        return f_up;
    }
    
    // compute the breakpoints for alpha, for both cases, and sort
    numvec alpha_bkpt_ls;                   // for the breakpoints
    sizvec alpha_pos_ls;                    // the position of the breakpoints
    sizvec alpha_i_cnt(nStates, 0);         // count the hitted breakpoints for each s
    alpha_bkpt_ls.reserve(2 * nStates);
    alpha_pos_ls.reserve(2 * nStates);

    for (size_t s = 0; s < nStates; s++) {
        prec_t alpha_i_low = ((theta_low[s] - theta_hat[s]) / sigma) + z[s];
        prec_t alpha_i_up  = ((theta_up[s]  - theta_hat[s]) / sigma) + z[s];

        alpha_bkpt_ls.push_back(alpha_i_low);
        alpha_bkpt_ls.push_back(alpha_i_up);

        alpha_pos_ls.push_back(s);
        alpha_pos_ls.push_back(s);
    }
    const sizvec idx = sort_indexes_ascending(alpha_bkpt_ls);
    prec_t alpha_star = Infty;

    // check piece by peice
    prec_t sum_theta_hat = 0.0;
    prec_t sum_sigma     = 0.0;
    prec_t sum_sigma_z   = 0.0;
    sum_theta_up  = 0.0;

    bool find_alpha = false;
    for (size_t i = 0; i < 2 * nStates; i++) {
        size_t idx_t = idx[i];

        // set alpha interval
        prec_t alpha_low = alpha_bkpt_ls[idx_t];
        prec_t alpha_up = (i < (2 * nStates) - 1) ? alpha_bkpt_ls[idx[i + 1]] : Infty;

        // find out which s for this breakpoint
        // update the terms based on whether its alpha_low or alpha_up
        size_t s_idx = alpha_pos_ls[idx_t];
        if (alpha_i_cnt[s_idx] == 0) { // alpha_i_low
            sum_theta_low -= theta_low[s_idx];

            sum_theta_hat += theta_hat[s_idx];
            sum_sigma     += sigma;
            sum_sigma_z   += sigma * z[s_idx];

            alpha_i_cnt[s_idx] += 1;
        }
        else { // alpha_i_up
            sum_theta_hat -= theta_hat[s_idx];
            sum_sigma     -= sigma;
            sum_sigma_z   -= sigma * z[s_idx];

            sum_theta_up += theta_up[s_idx];

            alpha_i_cnt[s_idx] += 1;
            assert(alpha_i_cnt[s_idx] <= 2);
            assert(i > 0);
        }

        prec_t alpha_temp = (lambda_p - sum_theta_low - sum_theta_hat - sum_theta_up + sum_sigma_z) / sum_sigma;

        if (alpha_low - smallEps <= alpha_temp && alpha_temp <= alpha_up + smallEps) {
            alpha_star = alpha_temp;
            find_alpha = true;
            break;
        }
    }
    if (find_alpha == false) {
        cout << sum_sigma << endl;
    }
    assert(find_alpha == true);

    // compute the solution with alpha_star
    numvec theta_star; theta_star.reserve(nStates);
    prec_t f_star = 0.0;
    for (size_t s = 0; s < nStates; s++) {
        prec_t u_i = theta_hat[s] + (sigma * (alpha_star - z[s]));
        u_i = max(u_i, theta_low[s]);
        u_i = min(u_i, theta_up[s]);

        theta_star.push_back(u_i);
        f_star += (u_i * z[s]) + ( (0.5 / sigma) * (u_i - theta_hat[s]) * (u_i - theta_hat[s]) );
    }
    if (update_theta) { pda.theta[s_hat][s_prime][a_prime] = theta_star; }
    
    //------------------------------------------------------
    //                  Testing below
    //------------------------------------------------------
    //prec_t f_gurobi_test = solver_BoxCon_QP_Gurobi(prob, pda, lambda_p, s_hat, s_prime, a_prime, update_theta);
    //cout << "This is subprobles error: " << f_gurobi_test << "  " << f_star << "  " << abs(f_gurobi_test - f_star) << endl;
    //assert( abs(f_star - f_gurobi_test) < 1e-5 );
    
    return f_star;
}



// dual update for a given s in S, with a fixed lambda_s
// for a fixed lambda_s, solve SA subproblems, which are QPs with box constraints
prec_t solver_dual_fixed_lambda_s(const RSMDPs& prob, PDA& pda, const prec_t lambda_p, size_t& s_hat, bool update_theta) {
    //construct solution
    prec_t f_star = pda.c_dual[s_hat] * lambda_p;
    f_star += (0.5 / pda.sigma ) * (lambda_p - pda.lambda0[s_hat]) * (lambda_p - pda.lambda0[s_hat]);
    
    // solve all subproblems
    for (size_t s = 0; s < prob.nStates; s++) {
        for (size_t a = 0; a < prob.nActions; a++) {
            f_star += solver_BoxCon_QP(prob, pda, lambda_p, s_hat, s, a, update_theta);
        }
    }

    //------------------------------------------------------
    //                  Testing below
    //------------------------------------------------------
    //prec_t f_gurobi_test = solver_dual_fixed_lambda_s_Gurobi_check(prob, pda, lambda_p, s_hat, update_theta);
    //cout << "This is sum of subproblems error: " << f_gurobi_test << "  " << f_star << "  " << abs(f_gurobi_test - f_star) << endl;
    //assert( abs(f_gurobi_test - f_star) < 1e-5 );
    
    return f_star;
}


// dual update for a given s_hat in S
// trisection on lambda_s
void PDA_dual_update_fixed_s(const RSMDPs& prob, PDA& pda, size_t& s_hat) {
    // ratio for golden section search
    const prec_t rho = (3.0 - sqrt(5.0)) / 2.0;
    const prec_t smallEps = 1e-12;

    // update dual coeffs, we need update for all (s_prime, a_prime)'s cases, as we need to compute lambda_UB
    pda.update_coeff_dual_c(prob, s_hat);
    for (size_t s = 0; s < prob.nStates; s++){
        for (size_t a = 0; a < prob.nActions; a++){
            pda.update_coeff_dual_z_ssa(prob, s_hat, s, a);
        }
    }
    /**********************************************************************************
                         Trisection
    **********************************************************************************/
    bool update_theta = false;
    
    prec_t lambda_LB = 0.0;
    prec_t lambda_UB = pda.compute_lambda_UB(prob, s_hat);
    
    prec_t lambda_range = lambda_UB - lambda_LB;
    prec_t lambda_u = lambda_LB + (rho * lambda_range);
    prec_t lambda_v = lambda_UB - (rho * lambda_range);
    
    prec_t f_u = solver_dual_fixed_lambda_s(prob, pda, lambda_u, s_hat, update_theta);
    prec_t f_v = solver_dual_fixed_lambda_s(prob, pda, lambda_v, s_hat, update_theta);

    //cout << "lambda_UB is " << lambda_UB << endl;
    
    while (true) {
        if (abs(f_u-f_v) <= 1e-15){
            lambda_LB = lambda_u;
            lambda_UB = lambda_v;
            
            lambda_range = lambda_UB - lambda_LB;
            lambda_u = lambda_LB + (rho * lambda_range);
            lambda_v = lambda_UB - (rho * lambda_range);
            
            f_u = solver_dual_fixed_lambda_s(prob, pda, lambda_u, s_hat, update_theta);
            f_v = solver_dual_fixed_lambda_s(prob, pda, lambda_v, s_hat, update_theta);
        }
        else if (f_u > f_v) {
            lambda_LB = lambda_u;
            lambda_u  = lambda_v;
            f_u = f_v;

            lambda_range = lambda_UB - lambda_LB;
            lambda_v = lambda_UB - (rho * lambda_range);
            
            assert(abs(lambda_u - lambda_LB - (rho * lambda_range)) < 1e-8);
            
            f_v = solver_dual_fixed_lambda_s(prob, pda, lambda_v, s_hat, update_theta);
        }
        else {
            lambda_UB = lambda_v;
            lambda_v  = lambda_u;
            f_v = f_u;

            lambda_range = lambda_UB - lambda_LB;
            lambda_u = lambda_LB + (rho * lambda_range);
            
            assert(abs(lambda_v - lambda_UB + (rho * lambda_range)) < 1e-8);
                
            f_u = solver_dual_fixed_lambda_s(prob, pda, lambda_u, s_hat, update_theta);
        }
        
        if (abs(f_u-f_v) <= 1e-15 && abs(lambda_u-lambda_v) <= 1e-15){
            update_theta = true;
            f_u = solver_dual_fixed_lambda_s(prob, pda, lambda_u, s_hat, update_theta);

            // update pda solution
            pda.lambda[s_hat] = lambda_u;
            return;
        }
    
        if (lambda_range >= 1e-10){
            assert(lambda_range > smallEps);
            if (lambda_LB >= lambda_u + smallEps){
                cout << lambda_LB << "  " << lambda_u << endl;
            }
            assert(lambda_LB < lambda_u + smallEps);
            if (lambda_u >= lambda_v + smallEps){
                cout << lambda_u << "  " << lambda_v << endl;
                cout << lambda_u - lambda_v << endl;
                cout << lambda_LB << endl;
                cout << lambda_UB - lambda_LB << endl;
                cout << f_u - f_v << endl;
            }
            assert(lambda_u < lambda_v + smallEps);
            assert(lambda_v < lambda_UB + smallEps);
        }
        
        // compute the solution
        if (lambda_range < 1e-10) {
            update_theta = true;
            if (f_u < f_v) {
                f_u = solver_dual_fixed_lambda_s(prob, pda, lambda_u, s_hat, update_theta);

                // update pda solution
                pda.lambda[s_hat] = lambda_u;
                return;
            }
            else {
                f_v = solver_dual_fixed_lambda_s(prob, pda, lambda_v, s_hat, update_theta);

                // update pda solution
                pda.lambda[s_hat] = lambda_v;
                return;
            }
        }
        
    }
    return;
}




// dual update for a given s in S
// trisection on lambda_s
void PDA_dual_update(const RSMDPs& prob, PDA& pda) {
    
    switch(pda.algType) {
        case PDA_org : {
            for (size_t s = 0; s < prob.nStates; s++) {
                PDA_dual_update_fixed_s(prob, pda, s);
                //PDA_dual_update_fixed_s_Gurobi_check(prob, pda, s);
            }
            break;
        }
        case PDA_block : {
            auto [s_sample,s2,a1,RV] = sampling_SSA_RV(prob);
            PDA_dual_update_fixed_s(prob, pda, s_sample);
            //PDA_dual_update_fixed_s_Gurobi_check(prob, pda, s_sample);
            break;
        }
        case PDA_block_plus : {
            auto [s_sample1,s_sample2,a_sample,probility] = sampling_SSA_RV(prob);
            
            if (probility < 1.0/ ( prob.nStates * prob.nActions ) ){
                PDA_dual_update_fixed_s(prob, pda, s_sample1);
                //PDA_dual_update_fixed_s_Gurobi_check(prob, pda, s_sample1);
            }
            else{
                solver_BoxCon_QP(prob, pda, pda.lambda0[s_sample1], s_sample1, s_sample2, a_sample, true);
            }

            break;
        }
        default : {
            invalid_argument("This type is not available.");
        }
   }
    return;
}


// the main process for PDA algorithm
pair<numvec,size_t> solver_PDA(const RSMDPs& prob, const PDA_type algType, const vector<numvec>& u_star, const prec_t& f_star) {
// paramters:

    // initialization
    PDA pda;
    pda.algType = algType; //??????
    pda.initialization(prob);

    
    // warm start
    //auto [pi, obj] = srect_solve_gurobi_mdps(prob);
    //pda.u0 = pi;
    //pda.u  = pi;
    //PDA_dual_update(prob, pda);
    
    /////////////////////////////////////////////////////////////
    //////////////////////Siyu_update////////////////////////////
    /////////////////////////////////////////////////////////////
    size_t nActions;      // get the number of actions
    size_t nStates;       // get the number of states
    nStates = prob.nStates;
    nActions = prob.nActions;
    prec_t X_norm = sqrt(1 + nStates * prob.gamma * prob.gamma + nActions - 1);

    size_t nIter = 2000;
    switch(pda.algType) {
        case PDA_org : {
            nIter = 2000;
            pda.sigma = sqrt(nStates) / X_norm;
            pda.nu = 1.0 / (sqrt(nStates) * X_norm);
            break;
        }
        case PDA_block : {
            nIter = 20000;
            //nIter = 50000;
            pda.sigma = nStates / X_norm;
            pda.nu = 1.0 / (nStates * X_norm);
            break;
        }
        case PDA_block_plus : {
            nIter = 400000;
            //nIter = 50000;
            pda.sigma = pow(nStates,1.25) / X_norm;
            pda.nu = 1.0 / (pow(nStates, 1.25) * X_norm);
            break;
        }
        default : {
            invalid_argument("This type is not available.");
        }
   }
    size_t iter_cnt = 0;
    /////////////////////////////////////////////////////////////
    //////////////////////Siyu_update////////////////////////////
    /////////////////////////////////////////////////////////////

    // 1st iteration
    PDA_primal_update(prob, pda);
    //PDA_primal_update_Gurobi_check(prob, pda);
    PDA_dual_update(prob, pda);
    iter_cnt++;
    
    pda.copy_new2old();
    pda.u_avg      = pda.u;
    pda.lambda_avg = pda.lambda;
    pda.theta_avg  = pda.theta;
    
    numvec obj_vec; obj_vec.reserve(nIter);
    //numvec err_vec; err_vec.reserve(Niter);
    
    for (size_t iter = 2; iter < nIter; iter++) {
        PDA_primal_update(prob, pda);
        //PDA_primal_update_Gurobi_check(prob, pda);
        PDA_dual_update(prob, pda);
        iter_cnt++;
        
        // update average
        pda.update_avg_soln_iter_k(iter);
        pda.copy_new2old();
        
        prec_t obj_current = pda.compute_obj(prob);
        obj_vec.push_back( obj_current );
        
        // 0.005, 0.001; original:0.05
        if (iter > 30 && abs(obj_current - f_star)/f_star < 0.05){
//            cout << "PDA stops." << endl;
            break;
        }
            
            //err_vec.push_back( pda.compute_error(u_star) );
    }
    return {obj_vec, iter_cnt};
    //return err_vec;
}



/*------------------------------------------------------------
                        Testing
------------------------------------------------------------*/

// This is a solver that solve the following problem, given z and other parameters
// min   z^T u + (1/2 nu) * || u - u_hat ||^2
// s.t   r^T u >= tau
//           u >= 0
void PDA_primal_update_Gurobi_check(const RSMDPs& prob, PDA& pda) {
    
    assert(1 < 0); // prevent using it
    
    const prec_t Infty    = numeric_limits<prec_t>::infinity();

    // update z_p based on current dual variables
    const vector<numvec>& z = pda.z_primal;
    
    const vector<numvec>& u_hat = pda.u0;
    const prec_t& nu            = pda.nu;
    const prec_t& tau           = prob.tau;
    
    const size_t& nStates  = prob.nStates;
    const size_t& nActions = prob.nActions;

    assert(prob.nStates  == z.size());
    assert(prob.nActions == z[0].size());
    assert(prob.nStates  == pda.u0.size());
    assert(prob.nActions == pda.u0[0].size());

    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    GRBVar** u;
    u = new GRBVar * [nStates];
    for (size_t s = 0; s < nStates; s++) {
        u[s] = model.addVars(nActions, GRB_CONTINUOUS);
        for (size_t a = 0; a < nActions; a++) {
            u[s][a].set(GRB_DoubleAttr_LB, 0.0);
            u[s][a].set(GRB_DoubleAttr_UB, Infty);
        }
    }
    
    // objective
    GRBQuadExpr objective;
    GRBLinExpr sum_ru;
    
    for (size_t s1 = 0; s1 < nStates; s1++) {
        for (size_t a = 0; a < nActions; a++) {
            objective += (z[s1][a] * u[s1][a]) + ( (0.5/nu) * (u[s1][a] - u_hat[s1][a]) * (u[s1][a] - u_hat[s1][a]) );
            
            sum_ru += prob.r[s1][a] * u[s1][a];
        }
    }
    
    model.addConstr(sum_ru >= tau);
    model.setObjective(objective, GRB_MINIMIZE);
    
    
    // run optimization
    model.optimize();
    
    // retrieve policy values
    vector<numvec> u_star = pda.get_zeros_SA();
    prec_t errrr = -Infty;
    for (size_t s = 0; s < nStates; s++) {
        for (size_t a = 0; a < nActions; a++){
            u_star[s][a] = u[s][a].get(GRB_DoubleAttr_X);
            errrr = max(errrr, u_star[s][a] - pda.u[s][a]);
        }
    }
    cout << errrr << endl;
    assert(errrr < 1e-4);
    
    return;
}


// Gurobi check dual subproblem
prec_t solver_BoxCon_QP_Gurobi(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, const size_t& s_hat, const size_t& s_prime, const size_t& a_prime, bool update_theta) {

    assert(1 < 0); // prevent using it

    
    const prec_t Infty = numeric_limits<prec_t>::infinity();

    // update parameters if they haven't been updated.
    if (pda.algType == PDA_block_plus){
        pda.update_coeff_dual_z_ssa(prob, s_hat, s_prime, a_prime);
    }
    // compute theta_low and theta_up
    const auto [theta_low, theta_up] = pda.compute_theta_bound_ssa(prob, s_hat, s_prime, a_prime, lambda_p);
    
    const numvec& z          = pda.z_dual[s_hat][s_prime][a_prime];
    const prec_t& sigma      = pda.sigma;
    const numvec& theta_hat  = pda.theta0[s_hat][s_prime][a_prime];
    const size_t nStates     = z.size();

    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    GRBVar* theta;
    theta = model.addVars(nStates, GRB_CONTINUOUS);
    for (size_t s = 0; s < nStates; s++) {
        theta[s].set(GRB_DoubleAttr_LB, 0.0);
        theta[s].set(GRB_DoubleAttr_UB, Infty);
    }
    
    // objective
    GRBQuadExpr objective;
    GRBLinExpr sum_theta;
    
    for (size_t s = 0; s < nStates; s++) {
        objective += ( z[s] * theta[s] ) + ( (0.5/sigma) * (theta[s] - theta_hat[s]) * (theta[s] - theta_hat[s]) );
        
        sum_theta += theta[s];
        
        model.addConstr(theta_low[s] <= theta[s]);
        model.addConstr(theta[s] <= theta_up[s]);
    }
    
    model.addConstr(sum_theta == lambda_p);
    
    model.setObjective(objective, GRB_MINIMIZE);
    
    // run optimization
    model.optimize();
    
    // retrieve policy values
    numvec theta_star(nStates, 0.0);
    for (size_t s = 0; s < nStates; s++) {
        theta_star[s] = theta[s].get(GRB_DoubleAttr_X);
    }
    if (update_theta) { pda.theta[s_hat][s_prime][a_prime] = theta_star; }
    
    return model.get(GRB_DoubleAttr_ObjVal);
}





// dual update for a given s in S, with a fixed lambda_s
// for a fixed lambda_s, solve SA subproblems, which are QPs with box constraints
prec_t solver_dual_fixed_lambda_s_Gurobi_check(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, size_t& s_hat, bool update_theta){
    
    assert(1 < 0); // prevent using it
    
    const prec_t Infty = numeric_limits<prec_t>::infinity();
    
    const size_t& nStates  = prob.nStates;
    const size_t& nActions = prob.nActions;
    
    
    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    GRBVar*** theta;
    theta = new GRBVar * *[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        theta[s1] = new GRBVar * [nActions];
        for (size_t a = 0; a < nActions; a++) {
            theta[s1][a] = model.addVars(nStates, GRB_CONTINUOUS);
            for (size_t s2 = 0; s2 < nStates; s2++) {
                theta[s1][a][s2].set(GRB_DoubleAttr_LB, 0.0);
                theta[s1][a][s2].set(GRB_DoubleAttr_UB, Infty);
            }
        }
    }
    
    // objective
    GRBQuadExpr objective;
    vector<GRBLinExpr> sum_u_sa(nStates * nActions);
    
    //objective += pda.c_dual[s_hat] * lambda_p;
    //objective += (0.5/pda.sigma) * (lambda_p - pda.lambda0[s_hat]) * (lambda_p - pda.lambda0[s_hat]);
    size_t cnt = 0;
    for (size_t s1 = 0; s1 < nStates; s1++) {
        for (size_t a = 0; a < nActions; a++) {
            for (size_t s2 = 0; s2 < nStates; s2++) {
                objective += theta[s1][a][s2] * pda.z_dual[s_hat][s1][a][s2];
                objective += (0.5/pda.sigma) * (theta[s1][a][s2] - pda.theta0[s_hat][s1][a][s2]) * (theta[s1][a][s2] - pda.theta0[s_hat][s1][a][s2]) ;
                
                
                model.addConstr( theta[s1][a][s2] - ( lambda_p * prob.P[s1][a][s2] ) <=  prob.w[s_hat] );
                model.addConstr( theta[s1][a][s2] - ( lambda_p * prob.P[s1][a][s2] ) >= -prob.w[s_hat] );
                
                sum_u_sa[cnt] += theta[s1][a][s2];
            }
            model.addConstr( lambda_p == sum_u_sa[cnt] );
            cnt++;
        }
    }
    
    // set objective
    model.setObjective(objective, GRB_MINIMIZE);
    
    // run optimization
    model.optimize();
    
    prec_t Obj = 0.0;
    Obj += pda.c_dual[s_hat] * lambda_p;
    Obj += (0.5/pda.sigma) * (lambda_p - pda.lambda0[s_hat]) * (lambda_p - pda.lambda0[s_hat]);
    Obj += model.get(GRB_DoubleAttr_ObjVal);
    
    return Obj;
}



// Gurobi test for dual_update with fixed s
void PDA_dual_update_fixed_s_Gurobi_check(const RSMDPs& prob, PDA& pda, size_t& s_hat){
    
    assert(1 < 0); // prevent using it
    
    const prec_t Infty = numeric_limits<prec_t>::infinity();
    
    const size_t& nStates  = prob.nStates;
    const size_t& nActions = prob.nActions;
    
    // update dual coeffs, we need update for all (s_prime, a_prime)'s cases, as we need to compute lambda_UB
    pda.update_coeff_dual_c(prob, s_hat);
    for (size_t s = 0; s < prob.nStates; s++){
        for (size_t a = 0; a < prob.nActions; a++){
            pda.update_coeff_dual_z_ssa(prob, s_hat, s, a);
        }
    }
    
    
    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);
    
    auto lambda = model.addVar(0, Infty, 1.0, GRB_CONTINUOUS);
    
    GRBVar*** theta;
    theta = new GRBVar * *[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        theta[s1] = new GRBVar * [nActions];
        for (size_t a = 0; a < nActions; a++) {
            theta[s1][a] = model.addVars(nStates, GRB_CONTINUOUS);
            for (size_t s2 = 0; s2 < nStates; s2++) {
                theta[s1][a][s2].set(GRB_DoubleAttr_LB, 0.0);
                theta[s1][a][s2].set(GRB_DoubleAttr_UB, Infty);
            }
        }
    }
    
    // objective
    GRBQuadExpr objective;
    vector<GRBLinExpr> sum_u_sa(nStates * nActions);
    
    objective += pda.c_dual[s_hat] * lambda;
    objective += (0.5/pda.sigma) * (lambda - pda.lambda0[s_hat]) * (lambda - pda.lambda0[s_hat]);
    size_t cnt = 0;
    for (size_t s1 = 0; s1 < nStates; s1++) {
        for (size_t a = 0; a < nActions; a++) {
            for (size_t s2 = 0; s2 < nStates; s2++) {
                objective += theta[s1][a][s2] * pda.z_dual[s_hat][s1][a][s2];
                objective += (0.5/pda.sigma) * (theta[s1][a][s2] - pda.theta0[s_hat][s1][a][s2]) * (theta[s1][a][s2] - pda.theta0[s_hat][s1][a][s2]) ;
                
                
                model.addConstr( theta[s1][a][s2] - ( lambda * prob.P[s1][a][s2] ) <=  prob.w[s_hat] );
                model.addConstr( theta[s1][a][s2] - ( lambda * prob.P[s1][a][s2] ) >= -prob.w[s_hat] );
                
                sum_u_sa[cnt] += theta[s1][a][s2];
            }
            model.addConstr( lambda == sum_u_sa[cnt] );
            cnt++;
        }
    }
    
    // set objective
    model.setObjective(objective, GRB_MINIMIZE);
    
    // run optimization
    model.optimize();
    
    prec_t lambda_star = lambda.get(GRB_DoubleAttr_X);
    
    // test correctness
    //cout << lambda_star << " " << abs(lambda_star - pda.lambda[s_hat]) << endl;
    if (abs(lambda_star - pda.lambda[s_hat]) > 1e-4){
        cout << lambda_star << " " << abs(lambda_star - pda.lambda[s_hat]) << endl;
    }
    assert(abs(lambda_star - pda.lambda[s_hat]) < 1e-4);
    
    
    // replace proposed fast algorithm for testing
    /*
    pda.lambda[s_hat] = lambda_star;
    
    auto theta_star = pda.get_zeros_SAS();
    for (size_t s1 = 0; s1 < nStates; s1++) {
        for (size_t a = 0; a < nActions; a++) {
            for (size_t s2 = 0; s2 < nStates; s2++) {
                theta_star[s1][a][s2] = theta[s1][a][s2].get(GRB_DoubleAttr_X);
                // ! ! !
                pda.theta[s_hat][s1][a][s2] = theta_star[s1][a][s2];
            }
        }
    }
     */
    return;
}









