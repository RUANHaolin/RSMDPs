//
//  L1support.h
//  PDA_submit
//
//  Created by Haolin on 31/3/2023.
//

#ifndef L1support_h
#define L1support_h
#include "definitions.h"

class L1support{
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



#endif /* L1support_h */
