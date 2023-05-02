//
//  PDA.h
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#ifndef PDA_h
#define PDA_h
#include "definitions.h"

/*
 This is ENUM that define the different types of PDA
 */
enum PDA_type{
    PDA_org = 0,
    PDA_block,
    PDA_block_plus
};


/*
 This is a CLASS that contains all the information for PDA
 */
class PDA {
public:
    PDA_type algType;
    
    vector<numvec> u;
    vector<numvec> u0;
    vector<numvec> u_avg;

    numvec lambda;
    numvec lambda0;
    numvec lambda_avg;

    vector<vector<vector<numvec>>> theta;
    vector<vector<vector<numvec>>> theta0;
    vector<vector<vector<numvec>>> theta_avg;

    prec_t sigma;
    prec_t nu;

    // auxiliary MDPs variables
    size_t nActions;      // get the number of actions
    size_t nStates;       // get the number of states

    // primal update parameters
    vector<numvec> z_primal;
    
    // dual update parameters
    vector<vector<vector<numvec>>> z_dual;
    numvec c_dual;
    
    /*****************************************************************************
                        functions
    ******************************************************************************/

    numvec get_zeros_S() {
        numvec zeros(nStates, 0.0);
        return zeros;
    }

    vector<numvec> get_zeros_SA() {
        numvec zeros_s(nActions, 0.0);
        vector<numvec> zeros(nStates, zeros_s);
        return zeros;
    }

    vector<vector<numvec>> get_zeros_SAS() {
        numvec zero_sa(nStates, 0.0);
        vector<numvec> zeros_s(nActions, zero_sa);
        vector<vector<numvec>> zeros(nStates, zeros_s);
        return zeros;
    }

    vector<vector<vector<numvec>>> get_zeros_SSAS() {
        numvec zero_ssa(nStates, 0.0);
        vector<numvec> zeros_ss(nActions, zero_ssa);
        vector<vector<numvec>> zeros_s(nStates, zeros_ss);
        vector<vector<vector<numvec>>> zeros(nStates, zeros_s);
        return zeros;
    }

    void initialization(const RSMDPs& prob) {
        nStates  = prob.nStates;
        nActions = prob.nActions;

        prec_t sum_r = 0.0;
        for (size_t s = 0; s < nStates; s++) {
            for (size_t a = 0; a < nActions; a++) {
                sum_r += prob.r[s][a];
            }
        }
        prec_t u_initial = prob.tau / sum_r;
        numvec u_s(nActions, u_initial);
        vector<numvec> u0_temp(nStates, u_s);
        u0    = u0_temp;
        u     = u0;
        u_avg = u0;

        numvec ones(nStates, 1.0);
        lambda0    = ones;
        lambda     = ones;
        lambda_avg = ones;

        vector<vector<vector<numvec>>> theta_temp(nStates, prob.P);
        theta0    = theta_temp;
        theta     = theta_temp;
        theta_avg = theta_temp;

        // compute sigma and nu******
        /////////////////////////////////////////////////////////////
        //////////////////////Siyu_update////////////////////////////
        /////////////////////////////////////////////////////////////

        // Norm
        // Frobenius_norm
        // prec_t X_norm = sqrt((nStates * nActions) + (prob.gamma * prob.gamma * nStates * nStates * nActions));
        // 1_norm
        // prec_t X_norm = 1 + nStates * prob.gamma;
        // 2_norm
        // prec_t X_norm = sqrt(1 + nStates * prob.gamma * prob.gamma + nActions - 1);
        // inf_norm
        // prec_t X_norm = nActions;
        // sigma = 1.0 / X_norm;
        //cout << 1.0/Frobenius_norm << endl;
        // nu = sigma;

        // original step size
        // sigma = 0.01;
        // nu = 0.01;

        // new step size__G0 PDA?
        // sigma = pow(nStates, 1/4) / X_norm;
        // nu = 1.0 / (pow(nStates, 1/4) * X_norm);

        // new step size__G1 PDA1
        //sigma = sqrt(nStates) / X_norm;
        //nu = 1.0 / (sqrt(nStates) * X_norm);

        // new step size__G2 PDA2
        // sigma = nStates / X_norm;
        // nu = 1.0 / (nStates * X_norm);

        // new step size__G3 PDA3
        // sigma = pow(nStates,2) / X_norm;
        // nu = 1.0 / (pow(nStates, 2) * X_norm);
        
        /////////////////////////////////////////////////////////////
        //////////////////////Siyu_update////////////////////////////
        /////////////////////////////////////////////////////////////
        
        z_primal = get_zeros_SA();
        z_dual   = get_zeros_SSAS();
        c_dual   = get_zeros_S();
    }
    
    void update_coeff_primal(const RSMDPs& prob) {
        vector<numvec> z_temp = get_zeros_SA();
        for (size_t s1 = 0; s1 < nStates; s1++) {
            for (size_t a = 0; a < nActions; a++) {
                z_temp[s1][a] = lambda0[s1];
                for (size_t s2 = 0; s2 < nStates; s2++) {
                    z_temp[s1][a] -= prob.gamma * (theta0[s2][s1][a][s2]);
                }
            }
        }
        z_primal = z_temp;
    }
    
    // update the c value for the s_hat case
    void update_coeff_dual_c(const RSMDPs& prob, const size_t s_hat) {
        prec_t c_temp = prob.d[s_hat];
        for (size_t a = 0; a < nActions; a++) {
            c_temp -= (2.0 * u[s_hat][a]) - u0[s_hat][a];
        }
        //assert(c_temp < 0.0);
        c_dual[s_hat] = c_temp;
    }
    
    // update the z values for the (s_hat, s_prime, a_prime) case
    void update_coeff_dual_z_ssa(const RSMDPs& prob, const size_t s_hat, const size_t s_prime, const size_t a_prime) {
        numvec z_temp(nStates, 0.0);
        z_temp[s_hat] = prob.gamma *( (2.0 * u[s_prime][a_prime]) - u0[s_prime][a_prime] );
        
        z_dual[s_hat][s_prime][a_prime] = z_temp;
    }
    
    // compute the upper bound of lambda at s_hat
    prec_t compute_lambda_UB(const RSMDPs& prob, const size_t s_hat){
        
        // compute the old obj, z^top p, ||z||_1
        prec_t obj_old = c_dual[s_hat] * lambda0[s_hat];
        prec_t zp      = 0.0;
        prec_t z_norm1 = 0.0;
        
        for (size_t s1 = 0; s1 < nStates; s1++){
            for (size_t a1 = 0; a1 < nActions; a1++){
                for (size_t s2 = 0; s2 < nStates; s2++){
                    obj_old += theta0[s_hat][s1][a1][s2] * z_dual[s_hat][s1][a1][s2];
                    zp      += prob.P[s1][a1][s2] * z_dual[s_hat][s1][a1][s2];
                    z_norm1 += abs(z_dual[s_hat][s1][a1][s2]);
                }
            }
        }
        
        // we know the sufficient condition  A DeltaLambda^2 +  B DeltaLambda >= C
        prec_t A_temp = 1.0 / (2.0 * sigma);
        prec_t B_temp = c_dual[s_hat] + zp;
        prec_t C_temp = obj_old - (c_dual[s_hat] * lambda0[s_hat]) - (lambda0[s_hat] * zp) + (prob.w[s_hat] * z_norm1);
        prec_t B_2A   = B_temp / (2.0 * A_temp);
         
        prec_t discrim = (B_temp * B_temp) + (4.0 * A_temp * C_temp);
        
        prec_t lambda_ub = 0.0;
        if (discrim < 1e-13){
            lambda_ub = lambda0[s_hat];
        }
        else{
            prec_t delta_lambda_ub = - B_2A + sqrt((C_temp / A_temp) + (B_2A * B_2A));
//            cout << "lambda0[s_hat] is " << lambda0[s_hat] << endl;
//            cout << "delta_lambda_ub is " << delta_lambda_ub << endl;
//            cout << "lambda0[s_hat] is " << lambda0[s_hat] << endl;
//            cout << "c_dual[s_hat] is " << c_dual[s_hat] << endl;
//            cout << "zp is " << zp<< endl;
//            cout << "A_temp is " << A_temp << endl;
//            cout << "B_temp is " << B_temp << endl;
//            cout << "C_temp is " << C_temp << endl;
//            cout << "B_2A is " << B_2A << endl;
//            cout << "(C_temp / A_temp) is " << (C_temp / A_temp) << endl;
//            cout << "test 0.00001 " << A_temp + B_temp - C_temp << endl;
            if (delta_lambda_ub < 0.0){
                lambda_ub  = lambda0[s_hat];
            }
            else{
                lambda_ub  = lambda0[s_hat] + delta_lambda_ub;
            }
        }
        
        return lambda_ub;
    }

    
    
    // compute theta_UB and theta_LB for the (s_hat, s_prime, a_prime) case with fixed lambda_p
    pair<numvec,numvec> compute_theta_bound_ssa(const RSMDPs& prob, const size_t& s_hat, const size_t& s_prime, const size_t& a_prime, const prec_t& lambda_p) {
        
        numvec theta_LB(nStates, 0.0);
        numvec theta_UB(nStates, 0.0);
        
        for (size_t s1 = 0; s1 < nStates; s1++) {
            theta_LB[s1] = -prob.w[s_hat] + ( lambda_p * prob.P[s_prime][a_prime][s1]);
            theta_UB[s1] =  prob.w[s_hat] + ( lambda_p * prob.P[s_prime][a_prime][s1]);
            
            theta_LB[s1] = max(theta_LB[s1], 0.0);
        }
        return {theta_LB, theta_UB};
    }
    
    
    void copy_new2old() {
        u0      = u;
        lambda0 = lambda;
        theta0  = theta;
        return;
    }
   
    
    void update_avg_soln_iter_k(const size_t& k) {
        // update primal
        for (size_t s1 = 0; s1 < nStates; s1++) {
            for (size_t a = 0; a < nActions; a++) {
                u_avg[s1][a] = u_avg[s1][a] + ( (1.0/k) * (u[s1][a] - u_avg[s1][a]) );
            }
        }

        // update dual
        for (size_t s1 = 0; s1 < nStates; s1++) {
            lambda_avg[s1] = lambda_avg[s1] + ((1.0 / k) * (lambda[s1] - lambda_avg[s1]));
            for (size_t s2 = 0; s2 < nStates; s2++) {
                for (size_t a = 0; a < nActions; a++) {
                    for (size_t s3 = 0; s3 < nStates; s3++) {
                        theta_avg[s1][s2][a][s3] = theta_avg[s1][s2][a][s3] + ((1.0 / k) * (theta[s1][s2][a][s3] - theta_avg[s1][s2][a][s3]));
                    }
                }
            }
        }
        return;
    }

    
    prec_t compute_error(const vector<numvec>& u_star) {
        const prec_t Infty = numeric_limits<prec_t>::infinity();
        
        prec_t err = 0.0;
        for (size_t s1 = 0; s1 < nStates; s1++) {
            for (size_t a = 0; a < nActions; a++) {
                prec_t err_i = u_avg[s1][a] - u_star[s1][a];
                //
                err += err_i * err_i;
                //cout << err_i << endl;
                //if (err_i > err) { err = err_i; }
            }
        }
        err = sqrt(err);
        return err;
    }
    
    
    prec_t compute_obj(const RSMDPs& prob){
        prec_t obj = 0.0;
        for (size_t s1 = 0; s1 < nStates; s1++) {
            prec_t sum_us = 0.0;
            for(size_t a = 0; a < nActions; a++){
                sum_us += u_avg[s1][a];
            }
            prec_t part1 = sum_us - prob.d[s1];
            part1 = part1 * lambda_avg[s1];
            
            prec_t part_2 = 0.0;
            for (size_t s2 = 0; s2 < nStates; s2++) {
                for(size_t a = 0; a < nActions; a++){
                    part_2 += theta_avg[s1][s2][a][s1] * prob.gamma * u_avg[s2][a];
                }
            }
            obj += part1 - part_2;
        }
        return obj;
    }
};

tuple<size_t,size_t,size_t,prec_t> sampling_SSA_RV(const RSMDPs& prob);


vector<size_t> sampling_multiple_S_RV(const RSMDPs& prob, size_t n);



void PDA_primal_update(const RSMDPs& prob, PDA& pda);


prec_t solver_BoxCon_QP(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, const size_t& s_hat, const size_t& s_prime, const size_t& a_prime, bool update_theta);


prec_t solver_dual_fixed_lambda_s(const RSMDPs& prob, PDA& pda, const prec_t lambda_p, size_t& s_hat, bool update_theta);


void PDA_dual_update_fixed_s(const RSMDPs& prob, PDA& pda, size_t& s_hat);


void PDA_dual_update(const RSMDPs& prob, PDA& pda);


pair<numvec,size_t> solver_PDA(const RSMDPs& prob, const PDA_type algType, const vector<numvec>& u_star, const prec_t& f_star);


// testing functions for subproblems below

void PDA_primal_update_Gurobi_check(const RSMDPs& prob, PDA& pda);


prec_t solver_BoxCon_QP_Gurobi(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, const size_t& s_hat, const size_t& s_prime, const size_t& a_prime, bool update_theta);


prec_t solver_dual_fixed_lambda_s_Gurobi_check(const RSMDPs& prob, PDA& pda, const prec_t& lambda_p, size_t& s_hat, bool update_theta);


void PDA_dual_update_fixed_s_Gurobi_check(const RSMDPs& prob, PDA& pda, size_t& s_hat);

#endif /* PDA_h */
