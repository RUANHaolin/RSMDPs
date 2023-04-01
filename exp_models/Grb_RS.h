#ifndef Grb_RS_h
#define Grb_RS_h
#include "definitions.h"



// NMDPs
pair<double, vector<numvec>> nmdp(const vector<vector<numvec>> P, const numvec &r, const numvec &p0, prec_t gamma, int S, int A);


// RSMDPs
pair<double, vector<numvec>> rsmdp(const vector<vector<numvec>> P, const numvec &r, const numvec &d, prec_t gamma, int nStates, int nActions, numvec w, prec_t tau);


// DRMDPs by gurobi
pair<double, numvec> drmdp_srect(int S, int A, const vector<vector<numvec>> &P_est, const numvec &r, const numvec &V, double gamma, double theta, double k=1.0);


// value iteration
pair<numvec, vector<int>> VI(const vector<vector<numvec>> &P, int S, int A, numvec r, double gamma, int max_iter = 10000, double tol = 1e-3);


// the algorithm in the raam paper to solve the lp in VI_rmdp
// min_x   z^\top x
// s.t.    \Vert x-q \Vert_1 <= radius
//         e^\top x = 1
//         x >= 0
// "index" is the order of z
numvec lp_raam(int S, const vector<size_t> &index, const numvec &q, double radius);


// value iteration for RMDPs
pair<numvec, vector<int>> VI_rmdp(const vector<vector<numvec>> &P, int S, int A, const numvec &r, double gamma, double radius = 0.8, int max_iter = 10000, double tol = 1e-3);


// value iteration for DRMDPs
pair<numvec, vector<numvec>> VI_drmdp(const vector<vector<vector<numvec>>> &p_est, int S, int A, const numvec &r, double gamma, double theta, double k=1.0, int max_iter = 10000, double tol = 1e-3);


// generate preturbed transition matrices for DRMDPs
vector<vector<vector<numvec>>> matGen(const vector<vector<numvec>> &P_est, int S, int A, int genSize = 30);















#endif /* Grb_RS_h */
