//
//  gurobi_solver_rsmdps.h
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#ifndef gurobi_solver_rsmdps_h
#define gurobi_solver_rsmdps_h
#include "definitions.h"
#include "gurobi_c++.h"

pair<vector<numvec>, double> srect_solve_gurobi_rsmdps(const RSMDPs& prob);
pair<vector<numvec>, double> srect_solve_gurobi_mdps(const RSMDPs& prob);

#endif /* gurobi_solver_rsmdps_h */
