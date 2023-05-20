//
//  gurobi_solver_l1.h
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#ifndef gurobi_solver_l1_h
#define gurobi_solver_l1_h
#include "definitions.h"
#include "gurobi_c++.h"



numvec sarect_solve_gurobi_l1(size_t S, numvec V, numvec p_hat, prec_t radius);


// value iteration for RMDPs (gurobi)
pair<numvec, vector<int>> VI_rmdp_sarect(const vector<vector<numvec>> &P, int S, int A, const numvec &r, double gamma, double radius = 0.8, int max_iter = 10000, double tol = 1e-3);


// value iteration for RMDPs (gurobi) (sum up runtimes)
tuple<numvec, vector<int>, prec_t> VI_rmdp_sarect_runtimes(const vector<vector<numvec>> &P, int S, int A, const numvec &r, double gamma, double radius = 0.8, int max_iter = 10000, double tol = 1e-3);

// (sum up runtimes)
tuple<numvec, prec_t> sarect_solve_gurobi_l1_rumtimes(size_t S, const numvec &V, const numvec &p_hat, prec_t radius);



#endif /* gurobi_solver_l1_h */
