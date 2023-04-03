using namespace std;
#include "definitions.h"
#include "Grb_RS.h"
#include "randInstance.h"


// "Improvements on Percentiles" experiment under "river swim" environment in Section 5.1.
void exp1_RiSW(){
    cout << "This is the \"Improvements on Percentiles\" experiment under \"river swim\" environment in Section 5.1." << endl;
    cout << "The results are stored in the variables below:" << endl;
    cout << "\"nmdp_XXperc\" stores the XX percentiles of NMDPs under all sample sizes." << endl;
    cout << "\"rmdp_XXperc_YY\" stores the XX percentiles of RMDPs with radius YY under all sample sizes." << endl;
    cout << "\"drmdp_XXperc_YY\" stores the XX percentiles of DRMDPs with radius YY under all sample sizes." << endl;
    cout << "\"rsmdp_XXperc_YY\" stores the XX percentiles of RSMDPs with tau = YY * Z_N under all sample sizes." << endl;
    
    // parameters
    RiSW inst = genInst_RiSW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    int max_iter = 10000;
    int numOfTraj_eva = 5000;
    double tol = 1e-3;
    int MaxStePE = 100;                 // to evaluate policy
    int MaxStePE_Sam = 10;              // to read file (10 for RiSW and MR, 100 for GW)
    
    
    
    int num_of_sampSizes = 15;
    numvec nmdp_10perc; nmdp_10perc.reserve(num_of_sampSizes);                // ndmp, 10% percentiles
    numvec rmdp_10perc_08; rmdp_10perc_08.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 0.8
    numvec rmdp_10perc_10; rmdp_10perc_10.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 1.0
    numvec drmdp_10perc_10; drmdp_10perc_10.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 1.0
    numvec drmdp_10perc_16; drmdp_10perc_16.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 1.6
    numvec rsmdp_10perc_08; rsmdp_10perc_08.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.8
    numvec rsmdp_10perc_085; rsmdp_10perc_085.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.85
    
    numvec nmdp_20perc; nmdp_20perc.reserve(num_of_sampSizes);                 // ndmp, 20% percentiles
    numvec rmdp_20perc_08; rmdp_20perc_08.reserve(num_of_sampSizes);             // rmdp, 20% percentiles, radius 0.8
    numvec rmdp_20perc_10; rmdp_20perc_10.reserve(num_of_sampSizes);              // rmdp, 20% percentiles, radius 1.0
    numvec drmdp_20perc_10; drmdp_20perc_10.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 1.0
    numvec drmdp_20perc_16; drmdp_20perc_16.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 1.6
    numvec rsmdp_20perc_08; rsmdp_20perc_08.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.8
    numvec rsmdp_20perc_085; rsmdp_20perc_085.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.85
    
    numvec nmdp_30perc; nmdp_30perc.reserve(num_of_sampSizes);                // ndmp, 30% percentiles
    numvec rmdp_30perc_08; rmdp_30perc_08.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 0.8
    numvec rmdp_30perc_10; rmdp_30perc_10.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 1.0
    numvec drmdp_30perc_10; drmdp_30perc_10.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 1.0
    numvec drmdp_30perc_16; drmdp_30perc_16.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 1.6
    numvec rsmdp_30perc_08; rsmdp_30perc_08.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.8
    numvec rsmdp_30perc_085; rsmdp_30perc_085.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.85
    
    numvec nmdp_avg; nmdp_avg.reserve(num_of_sampSizes);                // ndmp, average
    numvec rmdp_avg_08; rmdp_avg_08.reserve(num_of_sampSizes);             // rmdp, average, radius 0.8
    numvec rmdp_avg_10; rmdp_avg_10.reserve(num_of_sampSizes);             // rmdp, average, radius 1.0
    numvec drmdp_avg_10; drmdp_avg_10.reserve(num_of_sampSizes);             // drmdp, average, radius 1.0
    numvec drmdp_avg_16; drmdp_avg_16.reserve(num_of_sampSizes);             // drmdp, average, radius 1.6
    numvec rsmdp_avg_08; rsmdp_avg_08.reserve(num_of_sampSizes);            // rsmdp, average, target ratio 0.8
    numvec rsmdp_avg_085; rsmdp_avg_085.reserve(num_of_sampSizes);             // rsmdp, average, target ratio 0.85
    
    for (int sam_size = 1; sam_size < num_of_sampSizes+1; sam_size++){
        auto start = std::chrono::system_clock::now();    // timing
        cout << "RiSW. Now is sam_size = " << sam_size << endl;
        
        // read estimated transition kernel
        string fileName = string("./exp1_samples_RiSw/SamSiz") +
        to_string(sam_size) + string("MaxSte") + to_string(MaxStePE_Sam) + string(".csv");
        vector<vector<numvec>> P_est = read_kernel(fileName, S, A);


        // nmdp
        auto [ V_nmdp_est, policy_nmdp_est ] = VI(P_est, S, A, R_sa, gamma, max_iter, tol);
        numvec R_record_nmdp = inst.EVA(numOfTraj_eva, policy_nmdp_est, MaxStePE, gamma);
        nmdp_10perc.push_back(percentile(R_record_nmdp, 0.1));
        nmdp_20perc.push_back(percentile(R_record_nmdp, 0.2));
        nmdp_30perc.push_back(percentile(R_record_nmdp, 0.3));
        nmdp_avg.push_back(average(R_record_nmdp));
        
        // rmdp radius 0.8
        auto [ V_rmdp_08, policy_rmdp_08 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 0.8);
        numvec R_record_rmdp_08 = inst.EVA(numOfTraj_eva, policy_rmdp_08, MaxStePE, gamma);
        rmdp_10perc_08.push_back(percentile(R_record_rmdp_08, 0.1));
        rmdp_20perc_08.push_back(percentile(R_record_rmdp_08, 0.2));
        rmdp_30perc_08.push_back(percentile(R_record_rmdp_08, 0.3));
        rmdp_avg_08.push_back(average(R_record_rmdp_08));
        
        // rmdp radius 1.0
        auto [ V_rmdp_10, policy_rmdp_10 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 1.0);
        numvec R_record_rmdp_10 = inst.EVA(numOfTraj_eva, policy_rmdp_10, MaxStePE, gamma);
        rmdp_10perc_10.push_back(percentile(R_record_rmdp_10, 0.1));
        rmdp_20perc_10.push_back(percentile(R_record_rmdp_10, 0.2));
        rmdp_30perc_10.push_back(percentile(R_record_rmdp_10, 0.3));
        rmdp_avg_10.push_back(average(R_record_rmdp_10));
        
        // drmdp radius 1.0
        vector<vector<vector<numvec>>> P_drmdp_10 = matGen(P_est, S, A);
        auto [ V_drmdp_10, policy_drmdp_10 ] = VI_drmdp(P_drmdp_10, S, A, R_sa, gamma, 1.0);
        numvec R_record_drmdp_10 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_10);
        drmdp_10perc_10.push_back(percentile(R_record_drmdp_10, 0.1));
        drmdp_20perc_10.push_back(percentile(R_record_drmdp_10, 0.2));
        drmdp_30perc_10.push_back(percentile(R_record_drmdp_10, 0.3));
        drmdp_avg_10.push_back(average(R_record_drmdp_10));
        
        // drmdp radius 1.6
        vector<vector<vector<numvec>>> P_drmdp_16 = matGen(P_est, S, A);
        auto [ V_drmdp_16, policy_drmdp_16 ] = VI_drmdp(P_drmdp_16, S, A, R_sa, gamma, 1.6);
        numvec R_record_drmdp_16 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_16);
        drmdp_10perc_16.push_back(percentile(R_record_drmdp_16, 0.1));
        drmdp_20perc_16.push_back(percentile(R_record_drmdp_16, 0.2));
        drmdp_30perc_16.push_back(percentile(R_record_drmdp_16, 0.3));
        drmdp_avg_16.push_back(average(R_record_drmdp_16));
        
        
        // rsmdp
        auto [ obj_nmdp, u_nmdp ] = nmdp(P_est, R_sa, p0, gamma, S, A);
        // rsmdp tau ratio 0.8
        double tau_08 = 0.8 * obj_nmdp;
        auto [ obj_rsmdp_08, u_rsmdp_08 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_08);
        vector<numvec> policy_rsmdp_08 = inst.u_to_policy(u_rsmdp_08);
        numvec R_record_rsmdp_08 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_08);
        rsmdp_10perc_08.push_back(percentile(R_record_rsmdp_08, 0.1));
        rsmdp_20perc_08.push_back(percentile(R_record_rsmdp_08, 0.2));
        rsmdp_30perc_08.push_back(percentile(R_record_rsmdp_08, 0.3));
        rsmdp_avg_08.push_back(average(R_record_rsmdp_08));
        // rsmdp tau ratio 0.85
        double tau_085 = 0.85 * obj_nmdp;
        auto [ obj_rsmdp_085, u_rsmdp_085 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_085);
        vector<numvec> policy_rsmdp_085 = inst.u_to_policy(u_rsmdp_085);
        numvec R_record_rsmdp_085 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_085);
        rsmdp_10perc_085.push_back(percentile(R_record_rsmdp_085, 0.1));
        rsmdp_20perc_085.push_back(percentile(R_record_rsmdp_085, 0.2));
        rsmdp_30perc_085.push_back(percentile(R_record_rsmdp_085, 0.3));
        rsmdp_avg_085.push_back(average(R_record_rsmdp_085));
        auto end = std::chrono::system_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        cout << "It took " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " secs." << endl;
    }
    
    
    
}

// "Improvements on Percentiles" experiment under "machine replacement" environment in Section 5.1.
void exp1_MR(){
    cout << "This is the \"Improvements on Percentiles\" experiment under \"machine replacement\" environment in Section 5.1." << endl;
    cout << "The results are stored in the variables below:" << endl;
    cout << "\"nmdp_XXperc\" stores the XX percentiles of NMDPs under all sample sizes." << endl;
    cout << "\"rmdp_XXperc_YY\" stores the XX percentiles of RMDPs with radius YY under all sample sizes." << endl;
    cout << "\"drmdp_XXperc_YY\" stores the XX percentiles of DRMDPs with radius YY under all sample sizes." << endl;
    cout << "\"rsmdp_XXperc_YY\" stores the XX percentiles of RSMDPs with tau = YY * Z_N under all sample sizes." << endl;
    
    // parameters
    MR inst = genInst_MR();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    int max_iter = 10000;
    int numOfTraj_eva = 5000;
    double tol = 1e-3;
    int MaxStePE = 100;                 // to evaluate policy
    int MaxStePE_Sam = 10;              // to read file (10 for RiSW and MR, 100 for GW)
    

    int num_of_sampSizes = 15;
    numvec nmdp_10perc; nmdp_10perc.reserve(num_of_sampSizes);                // ndmp, 10% percentiles
    numvec rmdp_10perc_02; rmdp_10perc_02.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 0.2
    numvec rmdp_10perc_12; rmdp_10perc_12.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 1.2
    numvec drmdp_10perc_02; drmdp_10perc_02.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 0.2
    numvec drmdp_10perc_05; drmdp_10perc_05.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 0.4
    numvec rsmdp_10perc_09; rsmdp_10perc_09.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.9
    numvec rsmdp_10perc_085; rsmdp_10perc_085.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.85
    
    numvec nmdp_20perc; nmdp_20perc.reserve(num_of_sampSizes);                 // ndmp, 20% percentiles
    numvec rmdp_20perc_02; rmdp_20perc_02.reserve(num_of_sampSizes);             // rmdp, 20% percentiles, radius 0.2
    numvec rmdp_20perc_12; rmdp_20perc_12.reserve(num_of_sampSizes);              // rmdp, 20% percentiles, radius 1.2
    numvec drmdp_20perc_02; drmdp_20perc_02.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 0.2
    numvec drmdp_20perc_05; drmdp_20perc_05.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 0.4
    numvec rsmdp_20perc_09; rsmdp_20perc_09.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.9
    numvec rsmdp_20perc_085; rsmdp_20perc_085.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.85
    
    numvec nmdp_30perc; nmdp_30perc.reserve(num_of_sampSizes);                // ndmp, 30% percentiles
    numvec rmdp_30perc_02; rmdp_30perc_02.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 0.2
    numvec rmdp_30perc_12; rmdp_30perc_12.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 1.2
    numvec drmdp_30perc_02; drmdp_30perc_02.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 0.2
    numvec drmdp_30perc_05; drmdp_30perc_05.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 0.4
    numvec rsmdp_30perc_09; rsmdp_30perc_09.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.9
    numvec rsmdp_30perc_085; rsmdp_30perc_085.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.85
    
    numvec nmdp_avg; nmdp_avg.reserve(num_of_sampSizes);                // ndmp, average
    numvec rmdp_avg_02; rmdp_avg_02.reserve(num_of_sampSizes);             // rmdp, average, radius 0.2
    numvec rmdp_avg_12; rmdp_avg_12.reserve(num_of_sampSizes);             // rmdp, average, radius 1.2
    numvec drmdp_avg_02; drmdp_avg_02.reserve(num_of_sampSizes);             // drmdp, average, radius 0.2
    numvec drmdp_avg_05; drmdp_avg_05.reserve(num_of_sampSizes);             // drmdp, average, radius 0.4
    numvec rsmdp_avg_09; rsmdp_avg_09.reserve(num_of_sampSizes);            // rsmdp, average, target ratio 0.9
    numvec rsmdp_avg_085; rsmdp_avg_085.reserve(num_of_sampSizes);             // rsmdp, average, target ratio 0.85
    
//    for (int sam_size = 1; sam_size < num_of_sampSizes+1; sam_size++){
    for (int sam_size = 1; sam_size < num_of_sampSizes+1; sam_size++){
        
        auto start = std::chrono::system_clock::now();    // timing
        cout << "MR. Now is sam_size = " << sam_size << endl;
        
        // read the estimated transition kernel
        string fileName = string("./exp1_samples_MR/SamSiz") +
        to_string(sam_size) + string("MaxSte") + to_string(MaxStePE_Sam) + string(".csv");
        vector<vector<numvec>> P_est = read_kernel(fileName, S, A);

        
        // nmdp
        auto [ V_nmdp_est, policy_nmdp_est ] = VI(P_est, S, A, R_sa, gamma, max_iter, tol);
        numvec R_record_nmdp = inst.EVA(numOfTraj_eva, policy_nmdp_est, MaxStePE, gamma);
        nmdp_10perc.push_back(percentile(R_record_nmdp, 0.1));
        nmdp_20perc.push_back(percentile(R_record_nmdp, 0.2));
        nmdp_30perc.push_back(percentile(R_record_nmdp, 0.3));
        nmdp_avg.push_back(average(R_record_nmdp));
        
        // rmdp radius 0.2
        auto [ V_rmdp_02, policy_rmdp_02 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 0.2);
        numvec R_record_rmdp_02 = inst.EVA(numOfTraj_eva, policy_rmdp_02, MaxStePE, gamma);
        rmdp_10perc_02.push_back(percentile(R_record_rmdp_02, 0.1));
        rmdp_20perc_02.push_back(percentile(R_record_rmdp_02, 0.2));
        rmdp_30perc_02.push_back(percentile(R_record_rmdp_02, 0.3));
        rmdp_avg_02.push_back(average(R_record_rmdp_02));
        
        // rmdp radius 1.2
        auto [ V_rmdp_12, policy_rmdp_12 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 1.2);
        numvec R_record_rmdp_12 = inst.EVA(numOfTraj_eva, policy_rmdp_12, MaxStePE, gamma);
        rmdp_10perc_12.push_back(percentile(R_record_rmdp_12, 0.1));
        rmdp_20perc_12.push_back(percentile(R_record_rmdp_12, 0.2));
        rmdp_30perc_12.push_back(percentile(R_record_rmdp_12, 0.3));
        rmdp_avg_12.push_back(average(R_record_rmdp_12));
        
        // drmdp radius 0.2
        vector<vector<vector<numvec>>> P_drmdp_02 = matGen(P_est, S, A);
        auto [ V_drmdp_02, policy_drmdp_02 ] = VI_drmdp(P_drmdp_02, S, A, R_sa, gamma, 0.2);
        numvec R_record_drmdp_02 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_02);
        drmdp_10perc_02.push_back(percentile(R_record_drmdp_02, 0.1));
        drmdp_20perc_02.push_back(percentile(R_record_drmdp_02, 0.2));
        drmdp_30perc_02.push_back(percentile(R_record_drmdp_02, 0.3));
        drmdp_avg_02.push_back(average(R_record_drmdp_02));
        
        // drmdp radius 0.5
        vector<vector<vector<numvec>>> P_drmdp_05 = matGen(P_est, S, A);
        auto [ V_drmdp_05, policy_drmdp_05 ] = VI_drmdp(P_drmdp_05, S, A, R_sa, gamma, 0.5);
        numvec R_record_drmdp_05 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_05);
        drmdp_10perc_05.push_back(percentile(R_record_drmdp_05, 0.1));
        drmdp_20perc_05.push_back(percentile(R_record_drmdp_05, 0.2));
        drmdp_30perc_05.push_back(percentile(R_record_drmdp_05, 0.3));
        drmdp_avg_05.push_back(average(R_record_drmdp_05));
        
        
        auto [ obj_nmdp, u_nmdp ] = nmdp(P_est, R_sa, p0, gamma, S, A);
        // rsmdp tau ratio 0.9
        double tau_09 = 0.9 * obj_nmdp;
        auto [ obj_rsmdp_09, u_rsmdp_09 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_09);
        vector<numvec> policy_rsmdp_09 = inst.u_to_policy(u_rsmdp_09);
        numvec R_record_rsmdp_09 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_09);
        rsmdp_10perc_09.push_back(percentile(R_record_rsmdp_09, 0.1));
        rsmdp_20perc_09.push_back(percentile(R_record_rsmdp_09, 0.2));
        rsmdp_30perc_09.push_back(percentile(R_record_rsmdp_09, 0.3));
        rsmdp_avg_09.push_back(average(R_record_rsmdp_09));
        // rsmdp tau ratio 0.85
        double tau_085 = 0.85 * obj_nmdp;
        auto [ obj_rsmdp_085, u_rsmdp_085 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_085);
        vector<numvec> policy_rsmdp_085 = inst.u_to_policy(u_rsmdp_085);
        numvec R_record_rsmdp_085 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_085);
        rsmdp_10perc_085.push_back(percentile(R_record_rsmdp_085, 0.1));
        rsmdp_20perc_085.push_back(percentile(R_record_rsmdp_085, 0.2));
        rsmdp_30perc_085.push_back(percentile(R_record_rsmdp_085, 0.3));
        rsmdp_avg_085.push_back(average(R_record_rsmdp_085));
        auto end = std::chrono::system_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        cout << "It took " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " secs." << endl;
    }

   
}

// "Improvements on Percentiles" experiment under "grid world" environment in Section 5.1.
void exp1_GW(){
    cout << "This is the \"Improvements on Percentiles\" experiment under \"grid world\" environment in Section 5.1." << endl;
    cout << "The results are stored in the variables below:" << endl;
    cout << "\"nmdp_XXperc\" stores the XX percentiles of NMDPs under all sample sizes." << endl;
    cout << "\"rmdp_XXperc_YY\" stores the XX percentiles of RMDPs with radius YY under all sample sizes." << endl;
    cout << "\"drmdp_XXperc_YY\" stores the XX percentiles of DRMDPs with radius YY under all sample sizes." << endl;
    cout << "\"rsmdp_XXperc_YY\" stores the XX percentiles of RSMDPs with tau = YY * Z_N under all sample sizes." << endl;
    
    // parameters
    GW inst = genInst_GW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    int max_iter = 10000;
    int numOfTraj_eva = 5000;
    double tol = 1e-3;
    int MaxStePE = 100;                 // to evaluate policy
    int MaxStePE_Sam = 100;              // to read file (10 for RiSW and MR, 100 for GW)
    
    int num_of_sampSizes = 15;
    numvec nmdp_10perc; nmdp_10perc.reserve(num_of_sampSizes);                // ndmp, 10% percentiles
    numvec rmdp_10perc_12; rmdp_10perc_12.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 1.2
    numvec rmdp_10perc_10; rmdp_10perc_10.reserve(num_of_sampSizes);              // rmdp, 10% percentiles, radius 1.0
    numvec drmdp_10perc_02; drmdp_10perc_02.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 1.0
    numvec drmdp_10perc_04; drmdp_10perc_04.reserve(num_of_sampSizes);              // drmdp, 10% percentiles, radius 1.6
    numvec rsmdp_10perc_09; rsmdp_10perc_09.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.8
    numvec rsmdp_10perc_085; rsmdp_10perc_085.reserve(num_of_sampSizes);              // rsmdp, 10% percentiles, target ratio 0.85
    
    numvec nmdp_20perc; nmdp_20perc.reserve(num_of_sampSizes);                 // ndmp, 20% percentiles
    numvec rmdp_20perc_12; rmdp_20perc_12.reserve(num_of_sampSizes);             // rmdp, 20% percentiles, radius 1.2
    numvec rmdp_20perc_10; rmdp_20perc_10.reserve(num_of_sampSizes);              // rmdp, 20% percentiles, radius 1.0
    numvec drmdp_20perc_02; drmdp_20perc_02.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 1.0
    numvec drmdp_20perc_04; drmdp_20perc_04.reserve(num_of_sampSizes);             // drmdp, 20% percentiles, radius 1.6
    numvec rsmdp_20perc_09; rsmdp_20perc_09.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.8
    numvec rsmdp_20perc_085; rsmdp_20perc_085.reserve(num_of_sampSizes);             // rsmdp, 20% percentiles, target ratio 0.85
    
    numvec nmdp_30perc; nmdp_30perc.reserve(num_of_sampSizes);                // ndmp, 30% percentiles
    numvec rmdp_30perc_12; rmdp_30perc_12.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 1.2
    numvec rmdp_30perc_10; rmdp_30perc_10.reserve(num_of_sampSizes);             // rmdp, 30% percentiles, radius 1.0
    numvec drmdp_30perc_02; drmdp_30perc_02.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 1.0
    numvec drmdp_30perc_04; drmdp_30perc_04.reserve(num_of_sampSizes);             // drmdp, 30% percentiles, radius 1.6
    numvec rsmdp_30perc_09; rsmdp_30perc_09.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.8
    numvec rsmdp_30perc_085; rsmdp_30perc_085.reserve(num_of_sampSizes);             // rsmdp, 30% percentiles, target ratio 0.85
    
    numvec nmdp_avg; nmdp_avg.reserve(num_of_sampSizes);                // ndmp, average
    numvec rmdp_avg_12; rmdp_avg_12.reserve(num_of_sampSizes);             // rmdp, average, radius 1.2
    numvec rmdp_avg_10; rmdp_avg_10.reserve(num_of_sampSizes);             // rmdp, average, radius 1.0
    numvec drmdp_avg_02; drmdp_avg_02.reserve(num_of_sampSizes);             // drmdp, average, radius 1.0
    numvec drmdp_avg_04; drmdp_avg_04.reserve(num_of_sampSizes);             // drmdp, average, radius 1.6
    numvec rsmdp_avg_09; rsmdp_avg_09.reserve(num_of_sampSizes);            // rsmdp, average, target ratio 0.8
    numvec rsmdp_avg_085; rsmdp_avg_085.reserve(num_of_sampSizes);             // rsmdp, average, target ratio 0.85
    
    for (int sam_size = 1; sam_size < num_of_sampSizes+1; sam_size++){
        
        auto start = std::chrono::system_clock::now();    // timing
        cout << "GW. Now is sam_size = " << sam_size << endl;
        
        
        // read the estimated transition kernel
        string fileName = string("./exp1_samples_GW/SamSiz") +
        to_string(sam_size) + string("MaxSte") + to_string(MaxStePE_Sam) + string(".csv");
        vector<vector<numvec>> P_est = read_kernel(fileName, S, A);


        // nmdp
        auto [ V_nmdp_est, policy_nmdp_est ] = VI(P_est, S, A, R_sa, gamma, max_iter, tol);

        numvec R_record_nmdp = inst.EVA(numOfTraj_eva, policy_nmdp_est, MaxStePE, gamma);
        nmdp_10perc.push_back(percentile(R_record_nmdp, 0.1));
        nmdp_20perc.push_back(percentile(R_record_nmdp, 0.2));
        nmdp_30perc.push_back(percentile(R_record_nmdp, 0.3));
        nmdp_avg.push_back(average(R_record_nmdp));

        // rmdp radius 1.2
        auto [ V_rmdp_12, policy_rmdp_12 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 1.2);
        numvec R_record_rmdp_12 = inst.EVA(numOfTraj_eva, policy_rmdp_12, MaxStePE, gamma);
        rmdp_10perc_12.push_back(percentile(R_record_rmdp_12, 0.1));
        rmdp_20perc_12.push_back(percentile(R_record_rmdp_12, 0.2));
        rmdp_30perc_12.push_back(percentile(R_record_rmdp_12, 0.3));
        rmdp_avg_12.push_back(average(R_record_rmdp_12));

        // rmdp radius 1.0
        auto [ V_rmdp_10, policy_rmdp_10 ] = VI_rmdp(P_est, S, A, R_sa, gamma, 1.0);
        numvec R_record_rmdp_10 = inst.EVA(numOfTraj_eva, policy_rmdp_10, MaxStePE, gamma);
        rmdp_10perc_10.push_back(percentile(R_record_rmdp_10, 0.1));
        rmdp_20perc_10.push_back(percentile(R_record_rmdp_10, 0.2));
        rmdp_30perc_10.push_back(percentile(R_record_rmdp_10, 0.3));
        rmdp_avg_10.push_back(average(R_record_rmdp_10));

        // drmdp radius 0.2
        vector<vector<vector<numvec>>> P_drmdp_02 = matGen(P_est, S, A);
        auto [ V_drmdp_02, policy_drmdp_02 ] = VI_drmdp(P_drmdp_02, S, A, R_sa, gamma, 0.2);
        numvec R_record_drmdp_02 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_02);
        drmdp_10perc_02.push_back(percentile(R_record_drmdp_02, 0.1));
        drmdp_20perc_02.push_back(percentile(R_record_drmdp_02, 0.2));
        drmdp_30perc_02.push_back(percentile(R_record_drmdp_02, 0.3));
        drmdp_avg_02.push_back(average(R_record_drmdp_02));

        // drmdp radius 0.4
        vector<vector<vector<numvec>>> P_drmdp_04 = matGen(P_est, S, A);
        auto [ V_drmdp_04, policy_drmdp_04 ] = VI_drmdp(P_drmdp_04, S, A, R_sa, gamma, 0.4);
        numvec R_record_drmdp_04 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_drmdp_04);
        drmdp_10perc_04.push_back(percentile(R_record_drmdp_04, 0.1));
        drmdp_20perc_04.push_back(percentile(R_record_drmdp_04, 0.2));
        drmdp_30perc_04.push_back(percentile(R_record_drmdp_04, 0.3));
        drmdp_avg_04.push_back(average(R_record_drmdp_04));



        auto [ obj_nmdp, u_nmdp ] = nmdp(P_est, R_sa, p0, gamma, S, A);
        // rsmdp tau ratio 0.9
        double tau_09 = 0.9 * obj_nmdp;
        auto [ obj_rsmdp_09, u_rsmdp_09 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_09);
        vector<numvec> policy_rsmdp_09 = inst.u_to_policy(u_rsmdp_09);
        numvec R_record_rsmdp_09 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_09);
        rsmdp_10perc_09.push_back(percentile(R_record_rsmdp_09, 0.1));
        rsmdp_20perc_09.push_back(percentile(R_record_rsmdp_09, 0.2));
        rsmdp_30perc_09.push_back(percentile(R_record_rsmdp_09, 0.3));
        rsmdp_avg_09.push_back(average(R_record_rsmdp_09));
        // rsmdp tau ratio 0.85
        double tau_085 = 0.85 * obj_nmdp;
        auto [ obj_rsmdp_085, u_rsmdp_085 ] = rsmdp(P_est, R_sa, p0, gamma, S, A, w, tau_085);
        vector<numvec> policy_rsmdp_085 = inst.u_to_policy(u_rsmdp_085);
        numvec R_record_rsmdp_085 = inst.EVA(numOfTraj_eva, {}, MaxStePE, gamma, policy_rsmdp_085);
        rsmdp_10perc_085.push_back(percentile(R_record_rsmdp_085, 0.1));
        rsmdp_20perc_085.push_back(percentile(R_record_rsmdp_085, 0.2));
        rsmdp_30perc_085.push_back(percentile(R_record_rsmdp_085, 0.3));
        rsmdp_avg_085.push_back(average(R_record_rsmdp_085));
        auto end = std::chrono::system_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        cout << "It took " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " secs." << endl;
    }
    
    
}

// "Target-Oriented Feature" experiment under "river swim" environment in Section 5.2 (Table 1).
void exp2_RiSW(){
    // parameters
    RiSW inst = genInst_RiSW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_RiSw/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    
    // RMDP
    numvec radius_rmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_rmdp = radius_rmdp.size();
    numvec predictedReturns_rmdp; predictedReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec sampleReturns_rmdp; sampleReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec differences_rmdp; differences_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
//        cout << "V_rmdp is: " << endl;
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
//
//
        predictedReturns_rmdp.push_back(average(V_rmdp));
        sampleReturns_rmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rmdp.push_back(percentile(y, 0.5));
    }
    cout << "RMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rmdp);
    cout << "Differences are: ";
    printVec(differences_rmdp);
    
    
    // DRMDP
    numvec radius_drmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_drmdp = radius_drmdp.size();
    numvec predictedReturns_drmdp; predictedReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec sampleReturns_drmdp; sampleReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec differences_drmdp; differences_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
//
//
        predictedReturns_drmdp.push_back(average(V_drmdp));
        sampleReturns_drmdp.push_back(percentile(sampleReturns, 0.5));
        differences_drmdp.push_back(percentile(y, 0.5));
    }
    cout << "DRMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_drmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_drmdp);
    cout << "Differences are: ";
    printVec(differences_drmdp);

    
    // RSMDP
    numvec tau_coef = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    int numOfTau = tau_coef.size();
    numvec predictedReturns_rsmdp; predictedReturns_rsmdp.reserve(numOfTau);
    numvec sampleReturns_rsmdp; sampleReturns_rsmdp.reserve(numOfTau);
    numvec differences_rsmdp; differences_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
//
//
        predictedReturns_rsmdp.push_back(predictedReturn);
        sampleReturns_rsmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rsmdp.push_back(percentile(y, 0.5));
    }
    cout << "RSMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rsmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rsmdp);
    cout << "Differences are: ";
    printVec(differences_rsmdp);
    
}

// "Target-Oriented Feature" experiment under "machine replacement" environment in Section 5.2 (Table 3).
void exp2_MR(){
    // parameters
    MR inst = genInst_MR();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_MR/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    
    // RMDP
    numvec radius_rmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_rmdp = radius_rmdp.size();
    numvec predictedReturns_rmdp; predictedReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec sampleReturns_rmdp; sampleReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec differences_rmdp; differences_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
//        cout << "V_rmdp is: " << endl;
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
//
//
        predictedReturns_rmdp.push_back(average(V_rmdp));
        sampleReturns_rmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rmdp.push_back(percentile(y, 0.5));
    }
    cout << "RMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rmdp);
    cout << "Differences are: ";
    printVec(differences_rmdp);
    
    
    // DRMDP
    numvec radius_drmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_drmdp = radius_drmdp.size();
    numvec predictedReturns_drmdp; predictedReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec sampleReturns_drmdp; sampleReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec differences_drmdp; differences_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
//
//
        predictedReturns_drmdp.push_back(average(V_drmdp));
        sampleReturns_drmdp.push_back(percentile(sampleReturns, 0.5));
        differences_drmdp.push_back(percentile(y, 0.5));
    }
    cout << "DRMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_drmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_drmdp);
    cout << "Differences are: ";
    printVec(differences_drmdp);

    
    // RSMDP
    numvec tau_coef = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    int numOfTau = tau_coef.size();
    numvec predictedReturns_rsmdp; predictedReturns_rsmdp.reserve(numOfTau);
    numvec sampleReturns_rsmdp; sampleReturns_rsmdp.reserve(numOfTau);
    numvec differences_rsmdp; differences_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
//
//
        predictedReturns_rsmdp.push_back(predictedReturn);
        sampleReturns_rsmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rsmdp.push_back(percentile(y, 0.5));
    }
    cout << "RSMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rsmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rsmdp);
    cout << "Differences are: ";
    printVec(differences_rsmdp);
    
}

// "Target-Oriented Feature" experiment under "grid world" environment in Section 5.2 (Table 4).
void exp2_GW(){
    // parameters
    GW inst = genInst_GW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_GW/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    
    // RMDP
    numvec radius_rmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_rmdp = radius_rmdp.size();
    numvec predictedReturns_rmdp; predictedReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec sampleReturns_rmdp; sampleReturns_rmdp.reserve(numOfRadius_rmdp);
    numvec differences_rmdp; differences_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
//        cout << "V_rmdp is: " << endl;
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
//
//
        predictedReturns_rmdp.push_back(average(V_rmdp));
        sampleReturns_rmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rmdp.push_back(percentile(y, 0.5));
    }
    cout << "RMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rmdp);
    cout << "Differences are: ";
    printVec(differences_rmdp);
    
    
    // DRMDP
    numvec radius_drmdp = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5};
    int numOfRadius_drmdp = radius_drmdp.size();
    numvec predictedReturns_drmdp; predictedReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec sampleReturns_drmdp; sampleReturns_drmdp.reserve(numOfRadius_drmdp);
    numvec differences_drmdp; differences_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
//
//
        predictedReturns_drmdp.push_back(average(V_drmdp));
        sampleReturns_drmdp.push_back(percentile(sampleReturns, 0.5));
        differences_drmdp.push_back(percentile(y, 0.5));
    }
    cout << "DRMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_drmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_drmdp);
    cout << "Differences are: ";
    printVec(differences_drmdp);

    
    // RSMDP
    numvec tau_coef = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    int numOfTau = tau_coef.size();
    numvec predictedReturns_rsmdp; predictedReturns_rsmdp.reserve(numOfTau);
    numvec sampleReturns_rsmdp; sampleReturns_rsmdp.reserve(numOfTau);
    numvec differences_rsmdp; differences_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
//
//
        predictedReturns_rsmdp.push_back(predictedReturn);
        sampleReturns_rsmdp.push_back(percentile(sampleReturns, 0.5));
        differences_rsmdp.push_back(percentile(y, 0.5));
    }
    cout << "RSMDP: " << endl;
    cout << "Predicted returns are: ";
    printVec(predictedReturns_rsmdp);
    cout << "Sample returns are: ";
    printVec(sampleReturns_rsmdp);
    cout << "Differences are: ";
    printVec(differences_rsmdp);
    
}

// "Target-Oriented Feature" experiment under "river swim" environment in Section 5.2 (Figure 2).
void exp2_RiSW_write_xy(){
    cout << "\"x_rec_rmdp[0]\" and \"x_rec_rmdp[1]\" store the values of \"Level of Contamination\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_rmdp[0]\" and \"y_rec_rmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_drmdp[0]\" and \"x_rec_drmdp[1]\" store the values of \"Level of Contamination\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_drmdp[0]\" and \"y_rec_drmdp[1]\" store the values of \"Sample Return - Predicted Return\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_rsmdp[0]\" and \"x_rec_rsmdp[1]\" store the values of \"Level of Contamination\" for RSMDPs with tau = 0.8*Z_N and tau = 0.85*Z_N, respectively." << endl;
    cout << "\"y_rec_rsmdp[0]\" and \"y_rec_rsmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RSMDPs with tau = 0.8*Z_N and tau = 0.85*Z_N, respectively." << endl;
    // parameters
    RiSW inst = genInst_RiSW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_RiSw/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    
    // RMDP
    numvec radius_rmdp = {0.05, 0.1};
    int numOfRadius_rmdp = radius_rmdp.size();
    vector<numvec> x_rec_rmdp; x_rec_rmdp.reserve(numOfRadius_rmdp);
    vector<numvec> y_rec_rmdp; y_rec_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
        x_rec_rmdp.push_back(x);
        y_rec_rmdp.push_back(y);
    }

    
    
    // DRMDP
    numvec radius_drmdp = {0.05, 0.1};
    int numOfRadius_drmdp = radius_drmdp.size();
    vector<numvec> x_rec_drmdp; x_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<numvec> y_rec_drmdp; y_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
        x_rec_drmdp.push_back(x);
        y_rec_drmdp.push_back(y);
    }
    
    
    
    // RSMDP
    numvec tau_coef = {0.8, 0.85};
    int numOfTau = tau_coef.size();
    vector<numvec> x_rec_rsmdp; x_rec_rsmdp.reserve(numOfTau);
    vector<numvec> y_rec_rsmdp; y_rec_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
        x_rec_rsmdp.push_back(x);
        y_rec_rsmdp.push_back(y);
    }
    
    
}

// "Target-Oriented Feature" experiment under "machine replacement" environment in Section 5.2 (Figure 5).
void exp2_MR_write_xy(){
    cout << "\"x_rec_rmdp[0]\" and \"x_rec_rmdp[1]\" store the values of \"Level of Contamination\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_rmdp[0]\" and \"y_rec_rmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_drmdp[0]\" and \"x_rec_drmdp[1]\" store the values of \"Level of Contamination\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_drmdp[0]\" and \"y_rec_drmdp[1]\" store the values of \"Sample Return - Predicted Return\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_rsmdp[0]\" and \"x_rec_rsmdp[1]\" store the values of \"Level of Contamination\" for RSMDPs with tau = 0.9*Z_N and tau = 0.85*Z_N, respectively." << endl;
    cout << "\"y_rec_rsmdp[0]\" and \"y_rec_rsmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RSMDPs with tau = 0.9*Z_N and tau = 0.85*Z_N, respectively." << endl;
    // parameters
    MR inst = genInst_MR();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_MR/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    
    // RMDP
    numvec radius_rmdp = {0.05, 0.1};
    int numOfRadius_rmdp = radius_rmdp.size();
    vector<numvec> x_rec_rmdp; x_rec_rmdp.reserve(numOfRadius_rmdp);
    vector<numvec> y_rec_rmdp; y_rec_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
        x_rec_rmdp.push_back(x);
        y_rec_rmdp.push_back(y);
    }
    
    
    
    // DRMDP
    numvec radius_drmdp = {0.05, 0.1};
    int numOfRadius_drmdp = radius_drmdp.size();
    vector<numvec> x_rec_drmdp; x_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<numvec> y_rec_drmdp; y_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
        x_rec_drmdp.push_back(x);
        y_rec_drmdp.push_back(y);
    }
    
    
    
    // RSMDP
    numvec tau_coef = {0.9, 0.85};
    int numOfTau = tau_coef.size();
    vector<numvec> x_rec_rsmdp; x_rec_rsmdp.reserve(numOfTau);
    vector<numvec> y_rec_rsmdp; y_rec_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
        x_rec_rsmdp.push_back(x);
        y_rec_rsmdp.push_back(y);
    }
    
    
}

// "Target-Oriented Feature" experiment under "grid world" environment in Section 5.2 (Figure 6).
void exp2_GW_write_xy(){
    cout << "\"x_rec_rmdp[0]\" and \"x_rec_rmdp[1]\" store the values of \"Level of Contamination\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_rmdp[0]\" and \"y_rec_rmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_drmdp[0]\" and \"x_rec_drmdp[1]\" store the values of \"Level of Contamination\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"y_rec_drmdp[0]\" and \"y_rec_drmdp[1]\" store the values of \"Sample Return - Predicted Return\" for DRMDPs with radius = 0.05 and 0.1, respectively." << endl;
    cout << "\"x_rec_rsmdp[0]\" and \"x_rec_rsmdp[1]\" store the values of \"Level of Contamination\" for RSMDPs with tau = 0.9*Z_N and tau = 0.85*Z_N, respectively." << endl;
    cout << "\"y_rec_rsmdp[0]\" and \"y_rec_rsmdp[1]\" store the values of \"Sample Return - Predicted Return\" for RSMDPs with tau = 0.9*Z_N and tau = 0.85*Z_N, respectively." << endl;
    // parameters
    GW inst = genInst_GW();
    int S = inst.S;
    int A = inst.A;
    numvec R_sa = inst.R_sa;
    numvec p0(S, 1.0/double(S));
    double gamma = 0.85;
    numvec w(S, 1.0/double(S));
    double tol = 1e-3;
    int MaxStePE = 5000;                 // to evaluate policy
    int testSize = 1000;                 // number of perturbed kernels

    // read the unperturbed matrix
    string fileName = string("./exp2_samples_GW/train/MatID1.csv");
    vector<vector<numvec>> P_clean = read_kernel(fileName, S, A);       // the kernel that has not been perturbed
    cout << "P_clean[0][1] = " << endl;
    printVec(P_clean[0][1]);
    // RMDP
    numvec radius_rmdp = {0.05, 0.1};
    int numOfRadius_rmdp = radius_rmdp.size();
    vector<numvec> x_rec_rmdp; x_rec_rmdp.reserve(numOfRadius_rmdp);
    vector<numvec> y_rec_rmdp; y_rec_rmdp.reserve(numOfRadius_rmdp);
    for ( int i = 0; i < numOfRadius_rmdp; i++ ){
        cout << "Now is radius " << radius_rmdp[i] << endl;
        auto [ V_rmdp, policy_rmdp ] = VI_rmdp(P_clean, S, A, R_sa, gamma, radius_rmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, policy_rmdp, average(V_rmdp), gamma);
        x_rec_rmdp.push_back(x);
        y_rec_rmdp.push_back(y);
    }
    
    
    
    // DRMDP
    numvec radius_drmdp = {0.05, 0.1};
    int numOfRadius_drmdp = radius_drmdp.size();
    vector<numvec> x_rec_drmdp; x_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<numvec> y_rec_drmdp; y_rec_drmdp.reserve(numOfRadius_drmdp);
    vector<vector<vector<numvec>>> P_clean_drmdp; P_clean_drmdp.reserve(2);
    P_clean_drmdp.push_back(P_clean); P_clean_drmdp.push_back(P_clean);
    for ( int i = 0; i < numOfRadius_drmdp; i++ ){
        cout << "Now is radius " << radius_drmdp[i] << endl;
        auto [ V_drmdp, policy_drmdp ] = VI_drmdp(P_clean_drmdp, S, A, R_sa, gamma, radius_drmdp[i]);
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, average(V_drmdp), gamma, policy_drmdp);
        x_rec_drmdp.push_back(x);
        y_rec_drmdp.push_back(y);
    }
    
    
    
    // RSMDP
    numvec tau_coef = {0.9, 0.85};
    int numOfTau = tau_coef.size();
    vector<numvec> x_rec_rsmdp; x_rec_rsmdp.reserve(numOfTau);
    vector<numvec> y_rec_rsmdp; y_rec_rsmdp.reserve(numOfTau);
    auto [ obj_nmdp, u_nmdp ] = nmdp(P_clean, R_sa, p0, gamma, S, A);
    for ( int i = 0; i < numOfTau; i++ ){
        cout << "Now is tau_coef " << tau_coef[i] << endl;
        double tau = tau_coef[i] * obj_nmdp;
        auto [ obj_rsmdp, u_rsmdp ] = rsmdp(P_clean, R_sa, p0, gamma, S, A, w, tau);
        vector<numvec> policy_rsmdp = inst.u_to_policy(u_rsmdp);
        double predictedReturn = 0.0;
        for ( int s = 0; s < S; s++ ){
            for ( int a = 0; a < A; a++ ){
                predictedReturn += u_rsmdp[s][a] * R_sa[(s*A)+a];
            }
        }
        auto [ y, x, sampleReturns ] = inst.EVA_Obs(P_clean, testSize, {}, predictedReturn, gamma, policy_rsmdp);
        x_rec_rsmdp.push_back(x);
        y_rec_rsmdp.push_back(y);
    }
    
    
}


int main() {

    
    exp1_RiSW();
        
    exp1_MR();
        
    exp1_GW();
    
    exp2_RiSW();
    
    exp2_MR();
    
    exp2_GW();
    
    exp2_RiSW_write_xy();
    
    exp2_MR_write_xy();
    
    exp2_GW_write_xy();
    
    
    return 0;

}

