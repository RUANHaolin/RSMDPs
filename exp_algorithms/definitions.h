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



#endif /* definitions_h */
