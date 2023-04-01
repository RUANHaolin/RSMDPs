using namespace std;
#include "definitions.h"
#include "experiment_rand.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <numeric>


int main() {

    sizvec nStates_ls = {10,13,15,17};  // S = A

    size_t nRepeats = 100;               // number of instances we repeat for each setting
    
    // test speeds of algorithms (Gurobi, PDA, PDA_block, PDA_block+) for RSMDPs
    runsave_rsmdps_speed(GenRandInstance_rsmdps, nStates_ls, nRepeats);
    
    // test speed of Gurobi for RMDPs
    run_rmdps_speed(GenRandInstance_rsmdps, nStates_ls, nRepeats);
    
    return 0;

}

