#ifndef definitions_h
#define definitions_h
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <limits>
#include <cmath>
#include <memory>
#include <functional>

using namespace std;


using prec_t = double;
using numvec = vector<prec_t>;
using indvec = vector<long>;
using sizvec = vector<size_t>;

constexpr prec_t EPSILON = 1e-5;


/**
 * Returns sorted indexes for the given array (in ascending order).
 */
template <typename T>
inline sizvec sort_indexes_ascending(vector<T> const& v){
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}


/**
 * Returns sorted indexes for the given array (in descending order).
 */
template <typename T>
inline sizvec sort_indexes_descending(vector<T> const& v){
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    return idx;
}

/** This is useful functionality for debugging.  */
template<class T>
inline std::ostream & operator<<(std::ostream &os, const std::vector<T>& vec)
{
    for(const auto& p : vec){
        cout << p << " ";
    }
    return os;
}


/*
 This is a CLASS that contains all the information for problem instance
 
 */
class RSMDPs{
public:

    vector<vector<numvec>> P;       // transition kernel
    vector<numvec> r;               // reward
    prec_t gamma;                   // discount factor
    numvec d;                       // initial distribution
    
    prec_t tau;                     // target
    numvec w;                       // weights for rsmdps

    // auxiliary MDPs variables
    size_t nActions;      // get the number of actions
    size_t nStates;      // get the number of states

    // create auxiliary variables from input variables
    void createAuxiliaryVar(){
        // define nAction and nStates, and initialize pb_max and b_max_min
        nStates = P.size();
        nActions = P[0].size();
        numvec w0(nStates, 1.0 / nStates);
        w = w0;
    }
};


/*
 This is a CLASS that contains all the information for a robust Bellman equation in *state s*
 (i.e. one robust value iteration in one state):
 
 max_{pi_s} min_{p_s} \sum_a pi_sa * (p_sa^T*b)
 
 @param r_s        : rewards from state s (r_{sas})'s
 @param v0         : current value function
 @param lambda     : discount factor
 @param b          : The combined rewards and discounted value function
 @param p_bar      : Nominal distributions for all actions
 @param kappa      : Budgets in the ambiguity set
 @param sigma      : weights for the norm-based ambiguity sets
 @param vi_epsilon : accuracy of this value iteration
 */
class BellmanEq_s{
public:
    // bellman equation inputs
    numvec v0;
    vector<numvec> r_s;
    vector<numvec> p_bar;
    vector<numvec> sigma;
    prec_t kappa;
    prec_t lambda;
    prec_t vi_epsilon;
    
    
    // auxiliary variables
    size_t nAction;      // get the number of actions
    size_t nStates;      // get the number of states
    vector<numvec> b;    // get the combined rewards and discounted value function
    prec_t b_max_min;    // get the max_a min_s b[a][s]
    numvec pb_product;   // the inner product between p_bar and b for each a
    prec_t pb_max;       // the max of all pb_product
    numvec ba_min;       // the min of b_a, for each action
    prec_t p_min_min;    // the min_a min_s' p_{sas'}
    prec_t r_s_max;      // the max of all reward
    prec_t delta_phi;
    
    // create auxiliary variables from input variables
    void createAuxiliaryVar(){
        // define nAction and nStates, and initialize pb_max and b_max_min
        nAction   = r_s.size();
        nStates   = r_s[0].size();
        pb_max    = -numeric_limits<double>::infinity();
        b_max_min = 0.0;
        p_min_min = numeric_limits<double>::infinity();
        r_s_max   = -numeric_limits<double>::infinity();
        
        // compute lambda*v0
        numvec lambdaV0 = v0; transform(lambdaV0.begin(), lambdaV0.end(), lambdaV0.begin(), [lambda0 = lambda](prec_t x) { return x * lambda0; });
        for (size_t a = 0; a < nAction; a++) {
            // b[a] = r[a] + lambda*v0, for every action a
            numvec b_a = r_s[a]; transform(b_a.begin(), b_a.end(), lambdaV0.begin(), b_a.begin(), std::plus<prec_t>());
            b.push_back(b_a);
            
            // take the inner product between p_bar and b
            prec_t pb_t = inner_product(b_a.cbegin(),b_a.cend(),p_bar[a].cbegin(),0.0);
            pb_product.push_back(pb_t);
            // take the max of pb
            if (pb_t > pb_max) {pb_max = pb_t;}
            
            // take the max of b_a_min
            prec_t b_a_min = *min_element(b_a.cbegin(), b_a.cend());
            ba_min.push_back(b_a_min);
            if (b_a_min > b_max_min) {b_max_min = b_a_min;}
            
            prec_t p_a_min = *min_element(p_bar[a].cbegin(), p_bar[a].cend());
            if (p_a_min < p_min_min) {p_min_min = p_a_min;}
            
            prec_t r_sa_max = *max_element(r_s[a].cbegin(), r_s[a].cend());
            if (r_sa_max > r_s_max) {r_s_max = r_sa_max;}
        };
        
        // compute the delta_accuracy
        delta_phi = (vi_epsilon * kappa)/( (2.0 * nAction * r_s_max) + (nAction * vi_epsilon) );
    }
    
    
    // do all the checking after containing all variables
    void InputCheck() const{
        if (v0.size()   != nStates)       throw invalid_argument("v0 must have size nStates.");
        if (r_s.size()  != nAction)       throw invalid_argument("r_s must have size nAction.");
        if (p_bar.size()!= nAction)       throw invalid_argument("p_bar must have size nAction.");
        if (sigma.size()!= nAction)       throw invalid_argument("sigma must have size nAction.");
        if (b.size()    != nAction)       throw invalid_argument("b must have size nAction.");
        if (pb_product.size() != nAction) throw invalid_argument("pb_product must have size nAction.");
        if (kappa      <= 0.0)            throw invalid_argument("kappa must be strictly positive");
        if (lambda     <= 0.0)            throw invalid_argument("lambda must be strictly positive");
        if (vi_epsilon <= 0.0)            throw invalid_argument("vi_epsilon must be strictly positive");
        
        
        for (size_t a = 0; a < nAction; a++) {
            assert( abs(1.0 - accumulate(p_bar[a].cbegin(), p_bar[a].cend(), 0.0)) < EPSILON);
            assert( *min_element(p_bar[a].cbegin(), p_bar[a].cend()) >= 0.0 );
            assert( *min_element(b[a].cbegin(), b[a].cend()) >= 0.0 );
            
            if (r_s[a].size()  != nStates) throw invalid_argument("r_s[a] must have size nStates.");
            if (p_bar[a].size()!= nStates) throw invalid_argument("p_bar[a] must have size nStates.");
            if (sigma[a].size()!= nStates) throw invalid_argument("sigma[a] must have size nStates.");
            if (b[a].size()    != nStates) throw invalid_argument("b[a] must have size nStates.");
        }
    }
    
};


#endif /* definitions_h */
