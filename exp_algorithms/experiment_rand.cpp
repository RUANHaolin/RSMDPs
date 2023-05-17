//
//  experiment_rand.cpp
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#include <stdio.h>
#include "definitions.h"
#include "gurobi_solver_rsmdps.h"
#include "gurobi_solver_l1.h"
#include "PDA.h"

#include <random>
#include <fstream>
#include <functional>
//#include <experimental/filesystem>
#include <ctime>



// Generate a random instance for RSMDPs
RSMDPs GenRandInstance_rsmdps(size_t nStates, size_t nActions) {

    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.0, 1.0);

    prec_t gamma = 0.95;

    // initial distribution
    numvec d; d.reserve(nStates);
    for (size_t s = 0; s < nStates; s++) {
        d.push_back(1.0 / nStates);
    }

    // reward
    vector<numvec> r; r.reserve(nStates);
    for (size_t s = 0; s < nStates; s++) {
        numvec r_s; r_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++) {
            r_s.push_back( distribution(generator) );
        }
        r.push_back(r_s);
    }

    // transition kernel
    vector<vector<numvec>> P;
    P.reserve(nStates);
    for (size_t s = 0; s < nStates; s++) {
        vector<numvec> P_s;
        P_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++) {
            prec_t P_sa_sum = 0.0;
            numvec P_sa;
            P_sa.reserve(nStates);
            for (size_t s2 = 0; s2 < nStates; s2++) {
                prec_t p_temp = distribution(generator);
                P_sa.push_back(p_temp);
                P_sa_sum += p_temp;
            }

            for (size_t s2 = 0; s2 < nStates; s2++) {
                P_sa[s2] = P_sa[s2] / P_sa_sum;
            }
            P_s.push_back(P_sa);
        }
        P.push_back(P_s);
    }
    RSMDPs instance;
    instance.P = P;
    instance.r = r;
    instance.d = d;
    instance.gamma = gamma;
    instance.createAuxiliaryVar();

    auto [pi, obj] = srect_solve_gurobi_mdps(instance);
    instance.tau = 0.85 * obj;

    return instance;
}


// Compare ourself to Gurobi
tuple<prec_t,prec_t,prec_t, prec_t,prec_t,prec_t, prec_t,prec_t,prec_t, prec_t> get_speed_gurobi(RSMDPs& prob, const size_t nStates, const size_t instance_num) {

    // solve rsmdps
    auto start_gurobi_rs  = std::chrono::high_resolution_clock::now();
    auto [u_rs, obj_rs]   = srect_solve_gurobi_rsmdps(prob);
    auto finish_gurobi_rs = std::chrono::high_resolution_clock::now();

    prec_t dur_gurobi_rs = std::chrono::duration_cast<std::chrono::milliseconds> (finish_gurobi_rs - start_gurobi_rs).count();

    cout << "Gurobi (RSMDP) done, obj: "<<  obj_rs << endl;
    cout << "time used :" << dur_gurobi_rs << endl;
    
    // PDA
    auto start_pda   = std::chrono::high_resolution_clock::now();
    auto [obj_ls_pda, iter_pda1] = solver_PDA(prob, PDA_type (PDA_org), u_rs, obj_rs);
    auto finish_pda  = std::chrono::high_resolution_clock::now();

    prec_t dur_pda_rs    = std::chrono::duration_cast<std::chrono::milliseconds> (finish_pda - start_pda).count();
    prec_t err_rate_pda1 = abs(obj_rs - obj_ls_pda.back())/obj_rs;
    
    cout << "PDA done, obj: "<< obj_ls_pda.back() << " , rate : " << err_rate_pda1 <<  endl;
    cout << "nIter used :" << iter_pda1 << endl;
    cout << "time used :" << dur_pda_rs << endl;
    
    // PDA2
    auto start_pda2   = std::chrono::high_resolution_clock::now();
    auto [obj_ls_pda2, iter_pda2] = solver_PDA(prob, PDA_type (PDA_block), u_rs, obj_rs);
    auto finish_pda2  = std::chrono::high_resolution_clock::now();

    prec_t dur_pda2_rs   = std::chrono::duration_cast<std::chrono::milliseconds> (finish_pda2 - start_pda2).count();
    prec_t err_rate_pda2 = abs(obj_rs - obj_ls_pda2.back())/obj_rs;
    
    cout << "PDA_block done, obj: "<< obj_ls_pda2.back() << " , rate : " << err_rate_pda2 << endl;
    cout << "nIter used:" << iter_pda2 << endl;
    cout << "time used:" << dur_pda2_rs << endl;
    
    // PDA3
    auto start_pda3   = std::chrono::high_resolution_clock::now();
    auto [obj_ls_pda3, iter_pda3] = solver_PDA(prob, PDA_type (PDA_block_plus), u_rs, obj_rs);
    auto finish_pda3  = std::chrono::high_resolution_clock::now();

    prec_t dur_pda3_rs   = std::chrono::duration_cast<std::chrono::milliseconds> (finish_pda3 - start_pda3).count();
    prec_t err_rate_pda3 = abs(obj_rs - obj_ls_pda3.back())/obj_rs;
    
    cout << "PDA_block+ done, obj: "<< obj_ls_pda3.back() << " , rate : " << err_rate_pda3 << endl;
    cout << "nIter used:" << iter_pda3 << endl;
    cout << "time used:" << dur_pda3_rs << endl;
    
    
    
    string filename0 = "./pda_data/ind_data/RMDPS_" + to_string(nStates) + "_" + to_string(instance_num) + "_obj_time.csv";
    ofstream ofs0(filename0, ofstream::out);
    // write the header
    ofs0 << "states,instance_num,obj_rs,err_rate1,err_rate2,err_rate3,iter1,iter2,iter3" << endl;
    ofs0 << nStates   << "," <<
        instance_num  << "," <<
        obj_rs        << "," <<
        err_rate_pda1 << "," <<
        err_rate_pda2 << "," <<
        err_rate_pda3 << "," <<
            iter_pda1 << "," <<
            iter_pda2 << "," <<
            iter_pda3 << "," << endl;
    ofs0.close();

    string filename1 = "./pda_data/ind_data/RMDPS_" + to_string(nStates) + "_"+ to_string(instance_num) + "_pda1.csv";
    ofstream ofs1(filename1, ofstream::out);
    // write the header
    ofs1 << "obj" << endl;
    for (size_t i  = 0; i < obj_ls_pda.size(); i++){ ofs1 << obj_ls_pda[i] << endl; }
    ofs1.close();
    
    string filename2 = "./pda_data/ind_data/RMDPS_" + to_string(nStates) + "_"+ to_string(instance_num) + "_pda2.csv";
    ofstream ofs2(filename2, ofstream::out);
    // write the header
    ofs2 << "obj" << endl;
    for (size_t i = 0; i < obj_ls_pda2.size(); i++){ ofs2 << obj_ls_pda2[i] << endl; }
    ofs2.close();
    
    string filename3 = "./pda_data/ind_data/RMDPS_" + to_string(nStates) + "_"+ to_string(instance_num) + "_pda3.csv";
    ofstream ofs3(filename3, ofstream::out);
    // write the header
    ofs3 << "obj" << endl;
    for (size_t i = 0; i < obj_ls_pda3.size(); i++){ ofs3 << obj_ls_pda3[i] << endl; }
    ofs3.close();
//    cout << "States " << dur_gurobi_rs << " " << dur_pda_rs << " " << dur_pda2_rs << " " << dur_pda3_rs << endl;
    
    return {dur_gurobi_rs, dur_pda_rs, dur_pda2_rs, dur_pda3_rs, err_rate_pda1, err_rate_pda2, err_rate_pda3,iter_pda1,iter_pda2,iter_pda3};
}


/*
 * @brief Benchmarks s-rectangular Bellman updates and saves the timing to a CSV file
 */
void runsave_rsmdps_speed(const function<RSMDPs(size_t nStates, size_t nActions)>& prob_gen, const sizvec nStates_ls, const size_t repetitions) {
    
    // run the benchmark
    for (size_t s = 0; s < nStates_ls.size(); s++) {
        size_t nStates  = nStates_ls[s];
        size_t nActions = nStates_ls[s];

        string filename = "./pda_data/RMDPS_" + to_string(nStates) + "_overall.csv";
        
        ofstream ofs(filename, ofstream::out);
        
        
        // write the header
        ofs << "states,instance,rs_time,pda1_t,pda2_t,pda3_t,pda1_err,pda2_err,pda3_err,pda1_iter,pda2_iter,pda3_iter" << endl;
        
        for (size_t i = 0; i < repetitions; i++) {
            cout << "States " << nStates << ", Actions " << nActions << endl;

            RSMDPs instance = prob_gen(nStates, nActions);
                
            auto [T_rs, T_p1, T_p2, T_p3, err_p1, err_p2, err_p3, iter_p1, iter_p2, iter_p3] = get_speed_gurobi(instance, nStates, i);
            
            ofs << nStates << "," <<
                   i << "," <<
                T_rs << "," <<
                T_p1 << "," <<
                T_p2 << "," <<
                T_p3 << "," <<
                err_p1 << "," <<
                err_p2 << "," <<
                err_p3 << "," <<
                iter_p1 << "," <<
                iter_p2 << "," <<
                iter_p3 << "," << endl;
        }
        ofs.close();
    }
}




// Get the run time for Gurobi for ROBUST MDP
prec_t get_speed_gurobi_RMDP(RSMDPs& prob, const size_t nStates){
    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.0, 0.3);
    
    // solve rmdps
    auto start_gurobi_rmdp  = std::chrono::high_resolution_clock::now();
    
    
    vector<vector<numvec>> P = prob.P;
    
    size_t nActions = prob.nActions;
    
    numvec r; r.reserve(prob.nActions * prob.nStates);
    for ( size_t s = 0; s < prob.nStates; s++ ) {
        for ( size_t a = 0; a < prob.nActions; a++ ){
            r.push_back(prob.r[s][a]);
        }
    }
    
    prec_t gamma = prob.gamma;
    
    prec_t radius = distribution(generator);
    
    
    auto [ V_rmdp, policy_rmdp ] = VI_rmdp_sarect(P, prob.nStates, prob.nActions, r, gamma, radius);
    
    auto finish_gurobi_rmdp = std::chrono::high_resolution_clock::now();
    prec_t dur_gurobi_rmdp = std::chrono::duration_cast<std::chrono::milliseconds> (finish_gurobi_rmdp - start_gurobi_rmdp).count();

    cout << "time used :" << dur_gurobi_rmdp << endl;
    
    
    return dur_gurobi_rmdp;
    
}





/*
 * @brief Computing RMDP time
 */
void run_rmdps_speed(const function<RSMDPs(size_t nStates, size_t nActions)>& prob_gen, const sizvec nStates_ls, const size_t repetitions) {
    
    // run the benchmark
    for (size_t s = 0; s < nStates_ls.size(); s++) {
        size_t nStates  = nStates_ls[s];
        size_t nActions = nStates_ls[s];

        prec_t time_total = 0.0;
        
        for (size_t i = 0; i < repetitions; i++) {
            cout << "States " << nStates << ", Actions " << nActions << endl;

            RSMDPs instance = prob_gen(nStates, nActions);

            
            auto time_temp =  get_speed_gurobi_RMDP(instance, nStates);
            
            
            time_total += time_temp;

        }
        
        cout << "States " << nStates << ", average time " << time_total/repetitions << endl;
    }
}









