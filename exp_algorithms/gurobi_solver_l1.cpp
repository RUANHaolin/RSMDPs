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



numvec sarect_solve_gurobi_l1(size_t S, numvec V, numvec p_hat, prec_t radius){
    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();

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
    
    auto p = unique_ptr<GRBVar[]>(model.addVars(numvec(S, 0.0).data(), nullptr,
        nullptr,
        vector<char>(S, GRB_CONTINUOUS).data(),
        nullptr, int(S)));
    
    auto y = unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
        nullptr,
        vector<char>(S, GRB_CONTINUOUS).data(),
        nullptr, int(S)));
    
    
    /*********************************************************************
             Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    GRBLinExpr e_y;                     // e^\top y
    GRBLinExpr e_p;                     // e^\top p

    
    for ( size_t s = 0; s < S; s++ ){
        e_y += y[s];
        e_p += p[s];
        model.addConstr( p[s] - p_hat[s] <= y[s] );
        model.addConstr( p[s] - p_hat[s] >= -y[s] );
        objective += p[s] * V[s];
    }
    model.addConstr( e_y <= radius );
    model.addConstr( e_p == 1.0 );
    
    // set objective
    model.setObjective(objective, GRB_MINIMIZE);
    
//    auto start_gurobi_runtimeTimer  = std::chrono::high_resolution_clock::now();
    // run optimization
    model.optimize();
//    auto finish_gurobi_runtimeTimer = std::chrono::high_resolution_clock::now();
//    prec_t dur_gurobi_runtimeTimer = std::chrono::duration_cast<std::chrono::milliseconds> (finish_gurobi_runtimeTimer - start_gurobi_runtimeTimer).count();
//    cout << "time (timer) used :" << dur_gurobi_runtimeTimer << endl;
    // retrieve policy values
    numvec p_star; p_star.reserve(S);


    for (size_t s = 0; s < S; s++) {
        p_star.push_back(p[s].get(GRB_DoubleAttr_X));
    }
    
//    double Runtime = model.get(GRB_DoubleAttr_Runtime);
    
    
//    cout << "Runtime: " << Runtime << endl;

//     retrieve the worst-case response values
    return p_star ;
    
}


pair<numvec, vector<int>> VI_rmdp_sarect(const vector<vector<numvec>> &P, int S, int A, const numvec &r, double gamma, double radius, int max_iter, double tol){
    numvec V(S, 1000.0);                    // initialize values
    vector<int> policy; policy.reserve(S);
    numvec z; z.reserve(S);
    for ( int s = 0; s < S; s++ ){
        z.push_back(gamma * V[s]);
    }
    
    // compute the robust value function (grb)
    for ( int iter = 0; iter < max_iter; iter++ ){
        numvec newV; newV.reserve(S);
        for ( int s = 0; s < S; s++ ){
            double BV_max = -99.0;
            for ( int a = 0; a < A; a++ ){
                numvec p = sarect_solve_gurobi_l1(S, V, P[s][a], radius);
                double BV_a = r[s*A+a];
                for ( int s2 = 0; s2 < S; s2++ ){
                    BV_a += p[s2] * z[s2];
                }
                if ( BV_a > BV_max ){
                    BV_max = BV_a;
                }
            }
            newV.push_back(BV_max);
        }


        double Vdiff = 0.0;
        for ( int s = 0; s < S; s++ ){
            if ( abs(newV[s]-V[s]) > Vdiff ){
                Vdiff = abs(newV[s]-V[s]);
            }
        }
        V = newV;
        for ( int s = 0; s < S; s++ ){
            z[s] = gamma * V[s];
        }
        if ( Vdiff < tol ){
            break;
        }
    }
    
    
//    // compute the robust value function (raam)
//    for ( int iter = 0; iter < max_iter; iter++ ){
//        numvec newV; newV.reserve(S);
//        vector<size_t> index = argsort(z);
//        for ( int s = 0; s < S; s++ ){
//            double BV_max = -99.0;
//            for ( int a = 0; a < A; a++ ){
//                numvec p = lp_raam(S, index, P[s][a], radius);      // solve the linear program by Marek's method in the RAAM paper
//                double BV_a = r[s*A+a];
//                for ( int s2 = 0; s2 < S; s2++ ){
//                    BV_a += p[s2] * z[s2];
//                }
//                if ( BV_a > BV_max ){
//                    BV_max = BV_a;
//                }
//            }
//            newV.push_back(BV_max);
//        }
//        double Vdiff = 0.0;
//        for ( int s = 0; s < S; s++ ){
//            if ( abs(newV[s]-V[s]) > Vdiff ){
//                Vdiff = abs(newV[s]-V[s]);
//            }
//        }
//        V = newV;
//        for ( int s = 0; s < S; s++ ){
//            z[s] = gamma * V[s];
//        }
//        if ( Vdiff < tol ){
//            break;
//        }
//    }
    
    
    
    // compute the policy based on the robust value function
    for ( int s = 0; s < S; s++ ){
        double BV_max = -99.0;
        int idx_max = 0;
        for ( int a = 0; a < A; a++ ){
            double BV_a = r[s*A+a];
            for ( int s2 = 0; s2 < S; s2++ ){
                BV_a += P[s][a][s2] * gamma * V[s2];
            }
            if ( BV_a > BV_max ){
                BV_max = BV_a;
                idx_max = a;
            }
        }
        policy.push_back(idx_max);
    }
    return { V, policy };
}






