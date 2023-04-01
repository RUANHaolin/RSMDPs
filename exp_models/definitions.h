#ifndef definitions_h
#define definitions_h
#include <stdio.h>
#include <random>
#include <iostream>
#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <limits>
#include <cmath>
#include <memory>
#include <functional>
#include <string>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <tuple>
#include "gurobi_c++.h"
#include <fstream>
#include <sstream>
#include <typeinfo>
#include "fusion.h"
#include "supportLib.hpp"


using namespace std;

using prec_t = double;
using numvec = vector<prec_t>;
using indvec = vector<long>;
using sizvec = vector<size_t>;

constexpr prec_t EPSILON = 1e-5;
constexpr prec_t EPSILON_SMALL = 1e-9;
constexpr prec_t MY_PI = 3.14159265358979323846;


//------ inf norm start-----
bool compare(prec_t a, prec_t b);


prec_t linf_norm(vector<prec_t>& v);
//------ inf norm end -----


// L2-norm of vectors
prec_t l2_norm(const vector<prec_t>& u);

// cdf of normal distribution
prec_t norm_cdf(prec_t x);


// -----------inverse cdf of normal distribution start----------
prec_t RationalApproximation(prec_t t);


// inverse cdf of normal distribution
prec_t norm_ppf(prec_t p);
// -----------inverse cdf of normal distribution end----------


// normal random number
prec_t normal_rand(prec_t mean, prec_t stddev);


// index of the max element in a numvec
size_t maxIdx(numvec v1);


// index of the max (absolute value) element in a numvec
size_t maxIdx_abs(numvec v);


// index of the max (absolute value) element in a vector<numvec>
pair<size_t, size_t> maxIdx_abs_mtx(const vector<numvec> &v);


// matrix transpose
vector<numvec> transpose(const vector<numvec> &Phi);

// print a vector
void printVec(const numvec &vec1);


// average of a double vector
double average(const numvec & v);


// percentile of a vector "vec1" (perc is lower percentile)
double percentile(const numvec &vec1, double perc);


// indices of elements from smallest element to largest element
template <typename T>
vector<size_t> argsort(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


// read a kernel in vector from csv file and return it in the form of a S*A*S tensor
vector<vector<numvec>> read_kernel(string fileName, int S, int A);


// transform a vector to a tensor of S*A*S
vector<vector<numvec>> tensor_kernel(const numvec &vec, int S, int A);


// read transition kernel as a vector
numvec read_kernel_vec(string fileName, int S, int A);


// generate a dirichlet random vector
// k is the length of alpha, alpha is the param of the dirichlet distribution
numvec dirichlet(const numvec &alpha, int k);


// non-uniform distribution (output are integers)
// samples: possible outcomes
// probabilities: corresponding probs
// outputSize: number of samples to draw
vector<int> discrete_int(const numvec &probabilities_in, const vector<int> &samples_in, int outputSize);


// return a random vector, where each of its entry is subject to a uniform distribution Uniform(lb, ub)
numvec uniform_vec(double lb, double ub, int len);






















class GW{
public:
    int S;                       // number of states
    int A;                       // number of actions
    int nrow;                    // number of rows
    int ncol;                    // number of cols
    numvec R;                    // rewards
    numvec R_sa;
    // auxiliary vars
    int SA;                      // S * A
    
    // current state ob, action a, return next-state and reward received
    pair<int, double> Tran(numvec rand_unif_2num, vector<int> rand_int_2num, int ob, int a){
        default_random_engine generator;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        // comment this if you need the same instance all the time.
        uniform_real_distribution<double> distribution(0.0, 1.0);
        vector<int> colIdx_all; colIdx_all.reserve(ncol);           // all possible column indice
        for ( int i = 0; i < ncol; i++ ){
            colIdx_all.push_back(i);
        }
        numvec z = {0.9, 0.2};
//        numvec alp_dirich = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 30.0, 3.0, 4.0, 5.0};            // param for the dirichlet distribution
//        vector<numvec> P_ran; P_ran.reserve(nrow);
//        for ( int i = 0; i < nrow; i++ ){
//            P_ran.push_back(dirichlet(alp_dirich, ncol));
//        }
        int row_curr = ob / ncol;         // current column index
        int col_curr= ob % ncol;         // current row index
        int k;                          // new column index
        int l;                          // new row index
        if ( rand_unif_2num[0] > z[row_curr] ){
            if ( a == 0 ){
                k = col_curr + 1;
            }
            else if (a == 1){
                k = col_curr - 1;
            }
            else {
                k = rand_int_2num[0];
            }
            k = max(min(k, ncol - 1), 0);
        }
        else {
            k = rand_int_2num[1];
        }

        if ( a == 2 ){
            l = row_curr + 1;
        }
        else if( a == 3 ){
            l = row_curr - 1;
        }
        else{
//            double randNum = distribution(generator);
            if ( rand_unif_2num[1] <= 0.35 ){
                l = row_curr + 1;
            }
            else if ( rand_unif_2num[1] <= 0.7 ){
                l = row_curr - 1;
            }
            else {
                l = row_curr;
            }
        }
        l = max(min(l, nrow - 1), 0);
//
        int ob_next = k + l * ncol;         // next-state
        double r_new = R[ob_next];
        
        return { ob_next, r_new };
        
    }
    
    
    // compute the reward of a single traj
    double run_single(int ob, vector<int> policy_det, int traj_len, double gamma, vector<numvec> policy_rd = {}){
        default_random_engine generator;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        // comment this if you need the same instance all the time.
        uniform_real_distribution<double> distribution(0.0, 1.0);
        double episode_reward = 0.0;
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        vector<int> action_all; action_all.reserve(A);           // all possible states
        for ( int a = 0; a < A; a++ ){
            action_all.push_back(a);
        }
        
        
        numvec alp_dirich = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 30.0, 3.0, 4.0, 5.0};            // param for the dirichlet distribution
        
        // generate all random numbers beforehands
        vector<numvec> rand_dirich; rand_dirich.reserve(traj_len);
        for ( int i = 0; i < traj_len * nrow; i++ ) {
            rand_dirich.push_back(dirichlet(alp_dirich, ncol));
        }
        vector<numvec> rand_unif; rand_unif.reserve(traj_len);
        for ( int i = 0 ; i < traj_len; i++ ){
            numvec rand_unif_i; rand_unif_i.reserve(2);
            for ( int j = 0; j < 2; j++ ){
                rand_unif_i.push_back(distribution(generator));
            }
            rand_unif.push_back(rand_unif_i);
        }
        vector<vector<int>> rand_int; rand_int.reserve(traj_len);
        for ( int i = 0; i < traj_len; i++ ){
            for ( int j = 0; j < 2; j++ ){
                rand_int.push_back(discrete_int(rand_dirich[(i*nrow)+j], state_all, nrow));
            }
        }
        
        
//        int ob =  discrete_int(numvec(S, 1.0/double(S)), state_all, 1)[0];  // randomly choose an initial state
        // if randomized policy is input, use it
        if (policy_rd.size()){
            for ( int t = 0; t < traj_len; t++ ){
                int a = discrete_int(policy_rd[ob], action_all, 1)[0];
                auto [ ob_next, r_new ] = Tran(rand_unif[t], rand_int[t], ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
        }
        else {
            for ( int t = 0; t < traj_len; t++ ){
                int a = policy_det[ob];
                auto [ ob_next, r_new ] = Tran(rand_unif[t], rand_int[t], ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
        }
        return episode_reward;
    }
    
    
    // Collect experimental return based on the given policy
    // "numOfTraj": number of trajs to evaluate the policy
    // "MaxStePE": number of steps in a single traj
    numvec EVA(int numOfTraj_eva, vector<int> policy_det, int MaxStePE, double gamma, vector<numvec> policy_rd = {}){
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        numvec R_record; R_record.reserve(numOfTraj_eva);
        vector<int> obs = discrete_int(numvec(S, 1.0/double(S)), state_all, numOfTraj_eva);
        for ( int i = 0; i < numOfTraj_eva; i++ ){
            double episode_reward = run_single(obs[i], policy_det, MaxStePE, gamma, policy_rd);
            R_record.push_back(episode_reward);
        }
        return R_record;
    }
    
    // extract randomized policy from the u matrix
    vector<numvec> u_to_policy(vector<numvec> u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[s][a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[s][a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    // extract randomized policy from the u vector
    vector<numvec> uVec_to_policy(numvec u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[(s*A)+a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[(s*A)+a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    
    // policy evaluation
    pair<numvec, double> policyEval(vector<int> policy_det, vector<vector<numvec>> P, double gamma, vector<numvec> policy_rd = {}, int max_iter=5000, double tol=1e-3){
        numvec V(S, 1000.0);
        if (policy_rd.size()){
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_rd[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V += policy_rd[s][a] * gamma_P_s_V[a];
                    }

                    newV_s += pi_s_gamma_P_s_V;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
//                cout << "Iter " << iter << "; newV is: " << endl;
//                printVec(newV);
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        else {
            vector<numvec> policy_det_forEval; policy_det_forEval.reserve(S);
            for ( int s = 0; s < S; s++ ){
                numvec policy_det_forEval_s(A, 0.0);
                policy_det_forEval_s[policy_det[s]] = 1.0;
                policy_det_forEval.push_back(policy_det_forEval_s);
            }
            
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V_a = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_det_forEval[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V_a += policy_det_forEval[s][a] * gamma_P_s_V[a];
                    }
                    newV_s += pi_s_gamma_P_s_V_a;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        
        return { V, average(V) };
        
    }
    
    tuple<numvec, numvec, numvec> EVA_Obs(vector<vector<numvec>> P_clean, int testSize, vector<int> policy_det, double predictedReturn, double gamma, vector<numvec> policy_rd = {}){
        numvec Y_record; Y_record.reserve(testSize);
        numvec X_record; X_record.reserve(testSize);
        numvec sampleReturns; sampleReturns.reserve(testSize);
        for ( int i = 0; i < testSize; i++ ){
//            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_RiSw/test/MatID1.csv");
            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_GW/test/MatID") + to_string(i+1) + string(".csv");
            vector<vector<numvec>> P = read_kernel(fileName, S, A);
//            cout << "P[0] is:" << endl;
//            printMat(P[0]);
//            cout << "policy is: " << endl;
//            for ( int s = 0; s < S; s++ ){
//                cout << policy_det[s] << endl;
//            }
            auto [ V, expectedReturn ] = policyEval(policy_det, P, gamma, policy_rd);
//            cout << "V is: " << endl;
//            printVec(V);
            sampleReturns.push_back(expectedReturn);
            Y_record.push_back(expectedReturn-predictedReturn);
            double P_diff = 0.0;
            for ( int s = 0; s < S; s++ ){
                for ( int a = 0; a < A; a++ ){
                    for ( int s2 = 0; s2 < S; s2++ ){
                        P_diff += abs(P[s][a][s2] - P_clean[s][a][s2]);
                    }
                }
            }
            X_record.push_back(P_diff);
        }
        
        return { Y_record, X_record, sampleReturns };
    }
    
    
};


class MR{
public:
    int S;                       // number of states
    int A;                       // number of actions
    numvec R;                    // rewards (of one row) (rewards ONLY depends on row index)
    numvec R_sa;
    // auxiliary vars
    int SA;                      // S * A
    
    pair<int, double> Tran(numvec rand_unif_5num, int ob, int a){
//        default_random_engine generator;
//        generator.seed(chrono::system_clock::now().time_since_epoch().count());
//        // comment this if you need the same instance all the time.
//        uniform_real_distribution<double> distribution(0.0, 1.0);
        int ob_next;            // next-state
        if ( ob == 9 ){
            if ( a == 0 ){
                ob_next = ob;
            }
            else {
//                double randNum = distribution(generator);
                if ( rand_unif_5num[0] >= 0.4 ){
                    ob_next = ob - 1;
                }
                else {
                    ob_next = ob;
                }
            }
        }
        else if ( ob == 8 ){
            if ( a == 0 ){
//                double randNum = distribution(generator);
                if ( rand_unif_5num[1] >= 0.2 ){
                    ob_next = 0;
                }
                else {
                    ob_next = ob;
                }
            }
            else {
                ob_next = ob;
            }
        }
        else if ( ob == 7 ){
            if ( a == 0 ){
                ob_next = ob;
            }
            else {
//                double randNum = distribution(generator);
                if ( rand_unif_5num[2] >= 0.9 ){
                    ob_next = 9;
                }
                else if ( rand_unif_5num[2] >= 0.6 ){
                    ob_next = ob;
                }
                else {
                    ob_next = 8;
                }
            }
        }
        else {
            if ( a == 0 ){
//                double randNum = distribution(generator);
                if ( rand_unif_5num[3] >= 0.2 ){
                    ob_next = ob + 1;
                }
                else {
                    ob_next = ob;
                }
            }
            else {
//                double randNum = distribution(generator);
                if ( rand_unif_5num[4] >= 0.9 ){
                    ob_next = 9;
                }
                else if (rand_unif_5num[4] >= 0.6){
                    ob_next = ob + 1;
                }
                else {
                    ob_next = 8;
                }
            }
        }
        double r_new = R[ob_next];
        return { ob_next, r_new };
    }
    
    // compute the reward of a single traj
    double run_single(int ob, vector<int> policy_det, int traj_len, double gamma, vector<numvec> policy_rd = {}){
        default_random_engine generator;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        // comment this if you need the same instance all the time.
        uniform_real_distribution<double> distribution(0.0, 1.0);
        double episode_reward = 0.0;
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        vector<int> action_all; action_all.reserve(A);           // all possible states
        for ( int a = 0; a < A; a++ ){
            action_all.push_back(a);
        }
        // generate all random numbers beforehands
        vector<numvec> rand_unif; rand_unif.reserve(traj_len);
        for ( int i = 0 ; i < traj_len; i++ ){
            numvec rand_unif_i; rand_unif_i.reserve(5);
            for ( int j = 0; j < 5; j++ ){
                rand_unif_i.push_back(distribution(generator));
            }
            rand_unif.push_back(rand_unif_i);
        }
        
//        int ob =  discrete_int(numvec(S, 1.0/double(S)), state_all, 1)[0];  // randomly choose an initial state
        // if randomized policy is input, use it
        if (policy_rd.size()){
            for ( int t = 0; t < traj_len; t++ ){
                int a = discrete_int(policy_rd[ob], action_all, 1)[0];
                auto [ ob_next, r_new ] = Tran(rand_unif[t], ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
        }
        else {
            for ( int t = 0; t < traj_len; t++ ){
                int a = policy_det[ob];
                auto [ ob_next, r_new ] = Tran(rand_unif[t], ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
        }
        return episode_reward;
    }
    
    
    // Collect experimental return based on the given policy
    // "numOfTraj": number of trajs to evaluate the policy
    // "MaxStePE": number of steps in a single traj
    numvec EVA(int numOfTraj_eva, vector<int> policy_det, int MaxStePE, double gamma, vector<numvec> policy_rd = {}){
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        numvec R_record; R_record.reserve(numOfTraj_eva);
        vector<int> obs = discrete_int(numvec(S, 1.0/double(S)), state_all, numOfTraj_eva);
        for ( int i = 0; i < numOfTraj_eva; i++ ){
            double episode_reward = run_single(obs[i], policy_det, MaxStePE, gamma, policy_rd);
            R_record.push_back(episode_reward);
        }
        return R_record;
    }
    
    // extract randomized policy from the u vector
    vector<numvec> u_to_policy(vector<numvec> u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[s][a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[s][a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    vector<numvec> uVec_to_policy(numvec u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[(s*A)+a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[(s*A)+a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    
    // policy evaluation
    pair<numvec, double> policyEval(vector<int> policy_det, vector<vector<numvec>> P, double gamma, vector<numvec> policy_rd = {}, int max_iter=5000, double tol=1e-3){
        numvec V(S, 1000.0);
        if (policy_rd.size()){
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_rd[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V += policy_rd[s][a] * gamma_P_s_V[a];
                    }

                    newV_s += pi_s_gamma_P_s_V;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
//                cout << "Iter " << iter << "; newV is: " << endl;
//                printVec(newV);
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        else {
            vector<numvec> policy_det_forEval; policy_det_forEval.reserve(S);
            for ( int s = 0; s < S; s++ ){
                numvec policy_det_forEval_s(A, 0.0);
                policy_det_forEval_s[policy_det[s]] = 1.0;
                policy_det_forEval.push_back(policy_det_forEval_s);
            }
            
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V_a = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_det_forEval[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V_a += policy_det_forEval[s][a] * gamma_P_s_V[a];
                    }
                    newV_s += pi_s_gamma_P_s_V_a;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        
        return { V, average(V) };
        
    }
    
    tuple<numvec, numvec, numvec> EVA_Obs(vector<vector<numvec>> P_clean, int testSize, vector<int> policy_det, double predictedReturn, double gamma, vector<numvec> policy_rd = {}){
        numvec Y_record; Y_record.reserve(testSize);
        numvec X_record; X_record.reserve(testSize);
        numvec sampleReturns; sampleReturns.reserve(testSize);
        for ( int i = 0; i < testSize; i++ ){
//            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_RiSw/test/MatID1.csv");
            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_MR/test/MatID") + to_string(i+1) + string(".csv");
            vector<vector<numvec>> P = read_kernel(fileName, S, A);
//            cout << "P[0] is:" << endl;
//            printMat(P[0]);
//            cout << "policy is: " << endl;
//            for ( int s = 0; s < S; s++ ){
//                cout << policy_det[s] << endl;
//            }
            auto [ V, expectedReturn ] = policyEval(policy_det, P, gamma, policy_rd);
//            cout << "V is: " << endl;
//            printVec(V);
            sampleReturns.push_back(expectedReturn);
            Y_record.push_back(expectedReturn-predictedReturn);
            double P_diff = 0.0;
            for ( int s = 0; s < S; s++ ){
                for ( int a = 0; a < A; a++ ){
                    for ( int s2 = 0; s2 < S; s2++ ){
                        P_diff += abs(P[s][a][s2] - P_clean[s][a][s2]);
                    }
                }
            }
            X_record.push_back(P_diff);
        }
        
        return { Y_record, X_record, sampleReturns };
    }
};


class RiSW{
public:
    int S;                       // number of states
    int A;                       // number of actions
    numvec R;                    // rewards (of one row) (rewards ONLY depends on row index)
    numvec R_sa;
    // auxiliary vars
    int SA;                      // S * A
    

    
    // test whether the uniform distributions are different...
    pair<int, double> Tran(numvec randNums, int ob, int a){
        int ob_next;            // next-state
        if ( ob == 0 ){
            if ( a == 0 ){
                ob_next = ob;
            }
            else {
                if ( randNums[0] >= 0.3 ){
                    ob_next = ob;
                }
                else{
                    ob_next = ob + 1;
                }
            }
        }
        else if ( ob == S - 1 ){
            if ( a == 0 ){
                ob_next = ob - 1;
            }
            else {
                if ( randNums[1] >= 0.3 ){
                    ob_next = ob - 1;
                }
                else {
                    ob_next = ob;
                }
            }
        }
        else {
            if ( a == 0 ){
                ob_next = ob - 1;
            }
            else {
                if ( randNums[2] >= 0.9 ){
                    ob_next = ob - 1;
                }
                else if ( randNums[2] >= 0.3 ){
                    ob_next = ob;
                }
                else {
                    ob_next = ob + 1;
                }
            }
        }
        double r_new = R[ob_next];
        return { ob_next, r_new };
    }
    
    
    // compute the reward of a single traj
    double run_single(int ob, vector<int> policy_det, int traj_len, double gamma, vector<numvec> policy_rd = {}){
        default_random_engine generator;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        // comment this if you need the same instance all the time.
        uniform_real_distribution<double> distribution(0.0, 1.0);
        
        double episode_reward = 0.0;
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        vector<int> action_all; action_all.reserve(A);           // all possible actions
        for ( int a = 0; a < A; a++ ){
            action_all.push_back(a);
        }
//        int ob =  discrete_int(numvec(S, 1.0/double(S)), state_all, 1)[0];  // randomly choose an initial state
//        cout << "initial state chosen is: " << ob << endl;    // check initial state
        
        
//        int ob1 =  discrete_int(numvec(S, 1.0/double(S)), state_all, 1)[0];  // randomly choose an initial state
        
        // ## try generate random numbers from here
        vector<numvec> randNums; randNums.reserve(traj_len);
        for ( int i = 0; i < traj_len; i++ ){
            numvec randNums_i; randNums_i.reserve(3);
            for ( int j = 0; j < 3; j++ ){
                randNums_i.push_back(distribution(generator));
            }
            randNums.push_back(randNums_i);
        }
//        printMat(policy_rd);
        // if randomized policy is input, use it
        if (policy_rd.size()){
//            cout << "randomized" << endl;// check if randomized policy is used
            for ( int t = 0; t < traj_len; t++ ){
                int a = discrete_int(policy_rd[ob], action_all, 1)[0];
//                cout << "ob = " << ob << "; a = " << a << endl;
                auto [ ob_next, r_new ] = Tran(randNums[t], ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
        }
        else {
            for ( int t = 0; t < traj_len; t++ ){
                int a = policy_det[ob];
                auto [ ob_next, r_new ] = Tran(randNums[t], ob, a);
//                auto [ ob_next, r_new ] = Tran(ob, a);
                episode_reward += pow(gamma, t) * r_new;
                ob = ob_next;
            }
            // ##
            
        }
        
        return episode_reward;
    }
    
    
    numvec EVA(int numOfTraj_eva, vector<int> policy_det, int MaxStePE, double gamma, vector<numvec> policy_rd = {}){
        vector<int> state_all; state_all.reserve(S);           // all possible states
        for ( int s = 0; s < S; s++ ){
            state_all.push_back(s);
        }
        numvec R_record; R_record.reserve(numOfTraj_eva);
        vector<int> obs = discrete_int(numvec(S, 1.0/double(S)), state_all, numOfTraj_eva);
        for ( int i = 0; i < numOfTraj_eva; i++ ){
//            double episode_reward = run_single_foo(obs[i], randNums_temp, policy_det, MaxStePE, gamma, policy_rd);
            double episode_reward = run_single(obs[i], policy_det, MaxStePE, gamma, policy_rd);
            R_record.push_back(episode_reward);
        }
        return R_record;
    }
    
    // extract randomized policy from the u vector
    vector<numvec> u_to_policy(vector<numvec> u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[s][a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[s][a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    
    vector<numvec> uVec_to_policy(numvec u){
        vector<numvec> policy; policy.reserve(S);
        for ( int s = 0; s < S; s++ ){
            numvec policy_s; policy_s.reserve(A);
            double u_s_sum = 0.0;
            for ( int a = 0; a < A; a++){
                u_s_sum += u[(s*A)+a];
            }
            if ( u_s_sum > 0.0 ){
                for ( int a = 0; a < A; a++){
                    policy_s.push_back(u[(s*A)+a]/u_s_sum);
                }
            }
            else {
                policy_s = numvec(A, 1.0/double(A));
            }
            policy.push_back(policy_s);
        }
        return policy;
    }
    
    
    // policy evaluation
    pair<numvec, double> policyEval(vector<int> policy_det, vector<vector<numvec>> P, double gamma, vector<numvec> policy_rd = {}, int max_iter=5000, double tol=1e-3){
        numvec V(S, 1000.0);
        if (policy_rd.size()){
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_rd[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V += policy_rd[s][a] * gamma_P_s_V[a];
                    }

                    newV_s += pi_s_gamma_P_s_V;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
//                cout << "Iter " << iter << "; newV is: " << endl;
//                printVec(newV);
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        else {
            vector<numvec> policy_det_forEval; policy_det_forEval.reserve(S);
            for ( int s = 0; s < S; s++ ){
                numvec policy_det_forEval_s(A, 0.0);
                policy_det_forEval_s[policy_det[s]] = 1.0;
                policy_det_forEval.push_back(policy_det_forEval_s);
            }
            
            for ( int iter = 0; iter < max_iter; iter++ ){
                double Vdiff = 0.0;
                numvec newV; newV.reserve(S);
                for ( int s = 0; s < S; s++ ){
                    double newV_s = 0.0;
                    numvec gamma_P_s_V; gamma_P_s_V.reserve(A);                  // gamma * P_s V
                    for ( int a = 0; a < A; a++ ){
                        double gamma_P_s_V_a = 0.0;
                        for ( int ss = 0; ss < S; ss++ ){
                            gamma_P_s_V_a += gamma * P[s][a][ss] * V[ss];
                        }
                        gamma_P_s_V.push_back(gamma_P_s_V_a);
                    }
                    double pi_s_gamma_P_s_V_a = 0.0;                            // pi_s^\top (gamma * P_s V)
                    for ( int a = 0; a < A; a++ ){
                        newV_s += policy_det_forEval[s][a] * R_sa[(s*A)+a];
                        pi_s_gamma_P_s_V_a += policy_det_forEval[s][a] * gamma_P_s_V[a];
                    }
                    newV_s += pi_s_gamma_P_s_V_a;
                    
                    // push back and compare to the current max
                    newV.push_back(newV_s);
                    if ( abs(newV_s-V[s]) > Vdiff ){
                        Vdiff = abs(newV_s-V[s]);
                    }
                }
                V = newV;
                if ( Vdiff < tol ){
                    break;
                }
            }
        }
        
        return { V, average(V) };
        
    }
    
    tuple<numvec, numvec, numvec> EVA_Obs(vector<vector<numvec>> P_clean, int testSize, vector<int> policy_det, double predictedReturn, double gamma, vector<numvec> policy_rd = {}){
        numvec Y_record; Y_record.reserve(testSize);
        numvec X_record; X_record.reserve(testSize);
        numvec sampleReturns; sampleReturns.reserve(testSize);
        for ( int i = 0; i < testSize; i++ ){
//            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_RiSw/test/MatID1.csv");
            string fileName = string("/Users/datou/CplusplusFile/RSMDP/RSIRL/exp2_samples_RiSw/test/MatID") + to_string(i+1) + string(".csv");
            vector<vector<numvec>> P = read_kernel(fileName, S, A);
//            cout << "P[0] is:" << endl;
//            printMat(P[0]);
//            cout << "policy is: " << endl;
//            for ( int s = 0; s < S; s++ ){
//                cout << policy_det[s] << endl;
//            }
            auto [ V, expectedReturn ] = policyEval(policy_det, P, gamma, policy_rd);
//            cout << "V is: " << endl;
//            printVec(V);
            sampleReturns.push_back(expectedReturn);
            Y_record.push_back(expectedReturn-predictedReturn);
            double P_diff = 0.0;
            for ( int s = 0; s < S; s++ ){
                for ( int a = 0; a < A; a++ ){
                    for ( int s2 = 0; s2 < S; s2++ ){
                        P_diff += abs(P[s][a][s2] - P_clean[s][a][s2]);
                    }
                }
            }
            X_record.push_back(P_diff);
        }
        
        return { Y_record, X_record, sampleReturns };
    }
    
    
};




#endif /* definitions_h */
