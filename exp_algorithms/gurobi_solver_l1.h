//
//  gurobi_solver_l1.h
//  PDA_submit
//
//  Created by datou on 31/3/2023.
//

#ifndef gurobi_solver_l1_h
#define gurobi_solver_l1_h
#include "definitions.h"
#include "gurobi_c++.h"

pair<numvec, double> srect_solve_gurobi_l1(const BellmanEq_s& instanceS);





#endif /* gurobi_solver_l1_h */
