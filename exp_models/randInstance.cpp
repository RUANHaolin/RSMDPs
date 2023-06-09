#include "randInstance.h"
#include "definitions.h"




// generate an instance of the grid world env
GW genInst_GW(){
    GW inst;
    
    inst.S = 24;
    inst.A = 4;
    inst.nrow = 2;
    inst.ncol = 12;
    inst.R = {0.0, 3.0, 21.0, 27.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 24.0, 0.0, 3.0, 21.0, 27.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 24.0};
    inst.R_sa = {0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 21.0, 21.0, 21.0, 21.0, 27.0, 27.0, 27.0, 27.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0, 24.0, 24.0, 24.0, 24.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 21.0, 21.0, 21.0, 21.0, 27.0, 27.0, 27.0, 27.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0, 24.0, 24.0, 24.0, 24.0};
    inst.SA = inst.S * inst.A;
    return inst;
}


MR genInst_MR(){
    MR inst;
    
    inst.S = 10;
    inst.A = 2;
    inst.R = {20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 0.0, 18.0, 10.0};
    inst.R_sa = {20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 0.0, 0.0, 18.0, 18.0, 10.0, 10.0};
    inst.SA = inst.S * inst.A;
    return inst;
}


RiSW genInst_RiSW(){
    RiSW inst;
    
    inst.S = 10;
    inst.A = 2;
    inst.R = {5.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0};
    inst.R_sa = {5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 15.0};
    inst.SA = inst.S * inst.A;
    return inst;
}


