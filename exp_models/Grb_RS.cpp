#include "Grb_RS.h"
#include "randInstance.h"



pair<double, vector<numvec>> nmdp(const vector<vector<numvec>> P, const numvec &r, const numvec &p0, prec_t gamma, int S, int A){
    const double inf = numeric_limits<prec_t>::infinity();

    
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model = GRBModel(env);

    
    /*********************************************************************
             Create varables
             see https://support.gurobi.com/hc/en-us/community/posts/4414331624849-Create-multi-dimensional-variables-read-data-file-in-C-
    *********************************************************************/

    GRBVar** u;
    u = new GRBVar * [S];
    for (size_t s = 0; s < S; s++) {
        u[s] = model.addVars(A, GRB_CONTINUOUS);
        for (size_t a = 0; a < A; a++) {
            u[s][a].set(GRB_DoubleAttr_LB, 0.0);
            u[s][a].set(GRB_DoubleAttr_UB, inf);
        }
    }

    /*********************************************************************
                      Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    vector<GRBLinExpr> sum_u_s(S);             // e^\top u_s
    vector<GRBLinExpr> sum_p_hat_y_s(S);       // hat{p}^\top Q_s u

    for (size_t s = 0; s < S; s++) {

        for (size_t a = 0; a < A; a++) {
            objective += r[s * A + a] * u[s][a];
            sum_u_s[s] += u[s][a];

            for (size_t s2 = 0; s2 < S; s2++) {
                sum_p_hat_y_s[s2] += gamma * P[s][a][s2] * u[s][a];
            }
        }
    }

    for (size_t s = 0; s < S; s++) {
        model.addConstr(sum_u_s[s] - sum_p_hat_y_s[s] <= p0[s]);
    }

    // set objective
    model.setObjective(objective, GRB_MAXIMIZE);

    // run optimization
    model.optimize();

    // retrieve policy values
    vector<numvec> policy; policy.reserve(S);
    for (size_t s = 0; s < S; s++) {
        numvec pi_s; pi_s.reserve(A);
        for (size_t a = 0; a < A; a++) {
            pi_s.push_back(u[s][a].get(GRB_DoubleAttr_X));
        }
        policy.push_back(pi_s);
    }

    // retrieve the response values
    return { model.get(GRB_DoubleAttr_ObjVal), policy };


}


pair<double, vector<numvec>> rsmdp(const vector<vector<numvec>> P, const numvec &r, const numvec &d, prec_t gamma, int nStates, int nActions, numvec w, prec_t tau){
    
    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();

    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
//    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);


    /*********************************************************************
             Create varables
             see https://support.gurobi.com/hc/en-us/community/posts/4414331624849-Create-multi-dimensional-variables-read-data-file-in-C-
    *********************************************************************/

    GRBVar** u;
    u = new GRBVar * [nStates];
    for (size_t s = 0; s < nStates; s++) {
        u[s] = model.addVars(nActions, GRB_CONTINUOUS);
        for (size_t a = 0; a < nActions; a++) {
            u[s][a].set(GRB_DoubleAttr_LB, 0.0);
            u[s][a].set(GRB_DoubleAttr_UB, inf);
        }
    }

    GRBVar*** alpha;
    alpha = new GRBVar * *[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        alpha[s1] = new GRBVar * [nStates];
        for (size_t s2 = 0; s2 < nStates; s2++) {
            alpha[s1][s2] = model.addVars(nActions, GRB_CONTINUOUS);
            for (size_t a = 0; a < nActions; a++) {
                alpha[s1][s2][a].set(GRB_DoubleAttr_LB, -inf);
                alpha[s1][s2][a].set(GRB_DoubleAttr_UB, inf);
            }
        }
    }

    GRBVar**** beta;
    GRBVar**** y;
    GRBVar**** y_abs;
    beta = new GRBVar * **[nStates];
    y = new GRBVar * **[nStates];
    y_abs = new GRBVar * **[nStates];
    for (size_t s1 = 0; s1 < nStates; s1++) {
        beta[s1] = new GRBVar * *[nStates];
        y[s1] = new GRBVar * *[nStates];
        y_abs[s1] = new GRBVar * *[nStates];
        for (size_t s2 = 0; s2 < nStates; s2++) {
            beta[s1][s2] = new GRBVar * [nActions];
            y[s1][s2] = new GRBVar * [nActions];
            y_abs[s1][s2] = new GRBVar * [nActions];
            for (size_t a = 0; a < nActions; a++) {
                beta[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);
                y[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);
                y_abs[s1][s2][a] = model.addVars(nStates, GRB_CONTINUOUS);

                for (size_t s3 = 0; s3 < nStates; s3++) {
                    beta[s1][s2][a][s3].set(GRB_DoubleAttr_LB, 0.0);
                    beta[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                    y[s1][s2][a][s3].set(GRB_DoubleAttr_LB, -inf);
                    y[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                    y_abs[s1][s2][a][s3].set(GRB_DoubleAttr_LB, 0.0);
                    y_abs[s1][s2][a][s3].set(GRB_DoubleAttr_UB, inf);
                }
            }
        }
    }

    auto y_norm1 = unique_ptr<GRBVar[]>(model.addVars(numvec(nStates, 0.0).data(), nullptr,
        nullptr,
        vector<char>(nStates, GRB_CONTINUOUS).data(),
        nullptr, int(nStates)));


    /*********************************************************************
             Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    GRBLinExpr sum_r_u;
    vector<GRBLinExpr> sum_u_s(nStates);             // e^\top u_s
    vector<GRBLinExpr> sum_alpha_s(nStates);         // e^\top alpha_s
    vector<GRBLinExpr> sum_p_hat_y_s(nStates);       // hat{p}^\top y_s
    vector<GRBLinExpr> sum_y_abs_s(nStates);         // sum of y_abs

    // 1st for-loop to handle the "for every s constraint"
    for (size_t s = 0; s < nStates; s++) {

        for (size_t a = 0; a < nActions; a++) {
            sum_r_u += r[s*nActions+a] * u[s][a];
            sum_u_s[s] += u[s][a];
        }

        for (size_t s2 = 0; s2 < nStates; s2++) {
            for (size_t a = 0; a < nActions; a++) {
                sum_alpha_s[s] += alpha[s][s2][a];
                for (size_t s3 = 0; s3 < nStates; s3++) {
                    sum_p_hat_y_s[s] += P[s2][a][s3] * y[s][s2][a][s3];

                    if (s == s3) {
                        model.addConstr(y[s][s2][a][s3] == beta[s][s2][a][s3] - (gamma * u[s2][a]) - alpha[s][s2][a]);
                    }
                    else {
                        model.addConstr(y[s][s2][a][s3] == beta[s][s2][a][s3] - alpha[s][s2][a]);
                    }

                    model.addConstr( y[s][s2][a][s3] <= y_abs[s][s2][a][s3]);
                    model.addConstr(-y[s][s2][a][s3] <= y_abs[s][s2][a][s3]);

                    sum_y_abs_s[s] += y_abs[s][s2][a][s3];
                }
            }
        }

        model.addConstr(sum_u_s[s] - d[s] <= - sum_alpha_s[s] - sum_p_hat_y_s[s]);
        model.addConstr(sum_y_abs_s[s] <= y_norm1[s]);
        objective += w[s] * y_norm1[s];
    }

    model.addConstr(sum_r_u >= tau);

    // set objective
    model.setObjective(objective, GRB_MINIMIZE);
    
    // run optimization
    model.optimize();

    // retrieve policy values
    vector<numvec> u_star;
    for (size_t s = 0; s < nStates; s++) {
        numvec pi_s; 
        for (size_t a = 0; a < nActions; a++) {
            pi_s.push_back(u[s][a].get(GRB_DoubleAttr_X));
        }
        u_star.push_back(pi_s);
    }


    // retrieve the worst-case response values
    return { model.get(GRB_DoubleAttr_ObjVal), u_star };
}


pair<double, numvec> drmdp_srect(int S, int A, const vector<vector<numvec>> &P_est, const numvec &r, const numvec &V, double gamma, double theta, double k){
    
    
    int N = P_est.size();
    
    // general constants values
    const double inf = numeric_limits<prec_t>::infinity();

    GRBEnv env = GRBEnv(true);
    // GRBEnv env = GRBEnv();

    // make sure it is run in a single thread for a fair comparison
    env.set(GRB_IntParam_OutputFlag, 0);
//    env.set(GRB_IntParam_Threads, 1);
    env.start();
    // construct the LP model
    GRBModel model = GRBModel(env);


    /*********************************************************************
             Create varables
             see https://support.gurobi.com/hc/en-us/community/posts/4414331624849-Create-multi-dimensional-variables-read-data-file-in-C-
    *********************************************************************/
    

    GRBVar*** xiUp;
    GRBVar*** xiLow;
    xiUp = new GRBVar * *[N];
    xiLow = new GRBVar * *[N];
    for (size_t i = 0; i < N; i++) {
        xiUp[i] = new GRBVar * [A];
        xiLow[i] = new GRBVar * [A];
        for (size_t a = 0; a < A; a++) {
            xiUp[i][a] = model.addVars(S, GRB_CONTINUOUS);
            xiLow[i][a] = model.addVars(S, GRB_CONTINUOUS);
            for (size_t s = 0; s < S; s++) {
                xiUp[i][a][s].set(GRB_DoubleAttr_LB, 0.0);
                xiUp[i][a][s].set(GRB_DoubleAttr_UB, inf);
                xiLow[i][a][s].set(GRB_DoubleAttr_LB, 0.0);
                xiLow[i][a][s].set(GRB_DoubleAttr_UB, inf);
            }
        }
    }
    
    
    GRBVar zeta = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "t");
    
    
    GRBVar** eta;
    eta = new GRBVar * [N];
    for (size_t i = 0; i < N; i++) {
        eta[i] = model.addVars(A, GRB_CONTINUOUS);
        for (size_t a = 0; a < A; a++) {
            eta[i][a].set(GRB_DoubleAttr_LB, -inf);
            eta[i][a].set(GRB_DoubleAttr_UB, inf);
        }
    }
    
    
    auto pi = unique_ptr<GRBVar[]>(model.addVars(numvec(A, 0.0).data(), nullptr,
        nullptr,
        vector<char>(A, GRB_CONTINUOUS).data(),
        nullptr, int(A)));
    
    
    /*********************************************************************
             Build Model
    *********************************************************************/

    // objective
    GRBLinExpr objective;

    // constraints terms
    GRBLinExpr p_xiLow_xiUp;             // \sum_{i,a,s'}\hat{p}^i_{s,a,s'} * (xiLow^i_{a,s'} - xiUp^i_{a,s'})
    GRBLinExpr e_pi;                     // e^\top \pi
    GRBLinExpr pi_r;                     // pi ^\top r
    GRBLinExpr eta_sum;                  // \sum_{i,a} eta_{i,a}
    
    
    for ( int a = 0; a < A; a++ ){
        for ( int i = 0; i < N; i++ ){
            eta_sum += eta[i][a];
            for ( int sp = 0; sp < S; sp++ ){
                p_xiLow_xiUp += P_est[i][a][sp] * (xiLow[i][a][sp] - xiUp[i][a][sp]);
                model.addConstr( (pi[a] * gamma * (1.0/double(N)) * V[sp]) + xiUp[i][a][sp] - xiLow[i][a][sp] + eta[i][a] >= 0.0 );
                model.addConstr( -xiUp[i][a][sp] - xiLow[i][a][sp] + ((1.0/double(N)) * zeta) == 0.0 );
            }
        }
        e_pi += pi[a];
        pi_r += pi[a] * r[a];
    }
    model.addConstr( e_pi == 1.0 );
    
    
    objective += pi_r - (zeta * theta) - eta_sum + p_xiLow_xiUp;
    
    
    // set objective
    model.setObjective(objective, GRB_MAXIMIZE);
    
    // run optimization
    model.optimize();

    // retrieve policy values
    numvec pi_star; pi_star.reserve(A);


    for (size_t a = 0; a < A; a++) {
        pi_star.push_back(pi[a].get(GRB_DoubleAttr_X));
    }
    


//     retrieve the worst-case response values
    return { model.get(GRB_DoubleAttr_ObjVal), pi_star };
    
}


pair<numvec, vector<int>> VI(const vector<vector<numvec>> &P, int S, int A, numvec r, double gamma, int max_iter, double tol){
    numvec V(S, 1000.0);
    vector<int> policy; policy.reserve(S);
    
    for ( int iter = 0; iter < max_iter; iter++ ){
        numvec newV; newV.reserve(S);
        for ( int s = 0; s < S; s++ ){
            double BV_max = -99.0;
            for ( int a = 0; a < A; a++ ){
                double BV_a = r[(s * A) + a];
                for ( int s2 = 0; s2 < S; s2++ ){
                    BV_a += P[s][a][s2] * gamma * V[s2];
                }
                if ( BV_a > BV_max ){
                    BV_max = BV_a;
                }
            }
            newV.push_back(BV_max);
        }
        
        double Vdiff = 0.0;
        for ( int s = 0; s < S; s++ ){
            if ( abs(newV[s] - V[s]) > Vdiff ){
                Vdiff = abs(newV[s] - V[s]);
            }
        }
//        cout << "Iter " << iter << ", Vdiff = " << Vdiff << endl;
        V = newV;
//        cout << "Iter " << iter << "; Vdiff = " << Vdiff;
        if ( Vdiff < tol ){
            break;
        }
    }
    
    for ( int s = 0; s < S; s++ ){
        double BV_max = -99.0;     // value function coresponding to the "best" action
        int BV_argmax = 0;         // index of the "best" action
        for ( int a = 0; a < A; a++ ){
            double BV_a = r[(s * A) + a];
            for ( int s2 = 0; s2 < S; s2++ ){
                BV_a += P[s][a][s2] * gamma * V[s2];
            }
            if ( BV_a >  BV_max){
                BV_max = BV_a;
                BV_argmax = a;
            }
        }
        policy.push_back(BV_argmax);
    }
    
    return { V, policy };
    
}


numvec lp_raam(int S, const vector<size_t> &index, const numvec &q, double radius){
    int i = S - 1;
    numvec o = q;
    double eps = min(1 - q[index[0]], radius * 0.5);
    o[index[0]] = eps + q[index[0]];
    
    while (eps > 0 && i >= 0){
        o[index[i]] = o[index[i]] - min(eps, o[index[i]]);
        eps = eps - min(eps, q[index[i]]);
        i -= 1;
        
    }
    
    return o;
}


pair<numvec, vector<int>> VI_rmdp(const vector<vector<numvec>> &P, int S, int A, const numvec &r, double gamma, double radius, int max_iter, double tol){
    numvec V(S, 1000.0);                    // initialize values
    vector<int> policy; policy.reserve(S);
    numvec z; z.reserve(S);
    for ( int s = 0; s < S; s++ ){
        z.push_back(gamma * V[s]);
    }
    
    // compute the robust value function
    for ( int iter = 0; iter < max_iter; iter++ ){
        numvec newV; newV.reserve(S);
        vector<size_t> index = argsort(z);
        for ( int s = 0; s < S; s++ ){
            double BV_max = -99.0;
            for ( int a = 0; a < A; a++ ){
                numvec p = lp_raam(S, index, P[s][a], radius);      // solve the linear program by Marek's method in the RAAM paper
                double BV_a = r[s*A+a];
                for ( int s2 = 0; s2 < S; s2++ ){
                    BV_a += p[s2] * z[s2];
                }
                if ( BV_a > BV_max ){
                    BV_max = BV_a;
                }
            }
            newV.push_back(BV_max);
        }
        double Vdiff = 0.0;
        for ( int s = 0; s < S; s++ ){
            if ( abs(newV[s]-V[s]) > Vdiff ){
                Vdiff = abs(newV[s]-V[s]);
            }
        }
        V = newV;
        for ( int s = 0; s < S; s++ ){
            z[s] = gamma * V[s];
        }
        if ( Vdiff < tol ){
            break;
        }
    }
    
    
    // compute the policy based on the robust value function
    for ( int s = 0; s < S; s++ ){
        double BV_max = -99.0;
        int idx_max = 0;
        for ( int a = 0; a < A; a++ ){
            double BV_a = r[s*A+a];
            for ( int s2 = 0; s2 < S; s2++ ){
                BV_a += P[s][a][s2] * gamma * V[s2];
            }
            if ( BV_a > BV_max ){
                BV_max = BV_a;
                idx_max = a;
            }
        }
        policy.push_back(idx_max);
    }
    return { V, policy };
}


pair<numvec, vector<numvec>> VI_drmdp(const vector<vector<vector<numvec>>> &p_est, int S, int A, const numvec &r, double gamma, double theta, double k, int max_iter, double tol){
    int N = p_est.size();
    numvec V(S, 1000.0);                                        // initialize value function
    vector<numvec> policy; policy.reserve(S);
    for ( int s = 0; s < S; s++ ){
        policy.push_back(numvec(A,0.0));
    }
//    auto start_VI = std::chrono::system_clock::now();    // timing
    for ( int iter = 0; iter < max_iter; iter++ ){
        vector<numvec> policy_temp; policy_temp.reserve(S);
        numvec newV; newV.reserve(S);
        
        for ( int s = 0; s < S; s++ ){
            
            
            vector<vector<numvec>> p_est_s; p_est_s.reserve(N);                     // N samples of P[s,:,:] (s fixed)
            for ( int i = 0; i < N; i++ ){
                vector<numvec> p_est_s_i = p_est[i][s];
                p_est_s.push_back(p_est_s_i);
            }
            
            
            
            numvec r_s = { r.begin() + (s * A),  r.begin() + ((s+1) * A)};
            auto [newV_s, policy_temp_s] = drmdp_srect(S, A, p_est_s, r_s, V, gamma, theta, k);
            newV.push_back(newV_s);
            policy[s] = policy_temp_s;
        }
        
        double Vdiff = 0.0;
        for ( int s = 0; s < S; s++ ){
            if (abs(newV[s] - V[s]) > Vdiff){
                Vdiff = abs(newV[s] - V[s]);
            }
        }
        
        V = newV;
        if (Vdiff < tol){
            break;
        }
        
    }
//    auto end_VI = std::chrono::system_clock::now();
////
//    auto duration_VI = duration_cast<std::chrono::microseconds>(end_VI - start_VI);
//    cout << "VI took " << double(duration_VI.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " secs." << endl;
//    printVec(V);
//    printMat(policy);
    
    return { V, policy };
}


vector<vector<vector<numvec>>> matGen(const vector<vector<numvec>> &P_est, int S, int A, int genSize){
    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.0, 1.0);
    
    vector<vector<vector<numvec>>> P_N; P_N.reserve(genSize);
    
    for ( int k = 0; k < genSize; k++ ){
        
        // generate the perturbed kernel
        vector<vector<numvec>> P_perturbed; P_perturbed.reserve(S);
        for ( int s = 0; s < S; s++ ){
            vector<numvec> P_perturbed_s; P_perturbed_s.reserve(A);
            for ( int a = 0; a < A; a++ ){
                numvec P_perturbed_sa; P_perturbed_sa.reserve(S);
                for ( int s2 = 0; s2 < S; s2++ ){
                    P_perturbed_sa.push_back((0.95 * P_est[s][a][s2]) + (0.05 * distribution(generator)));
                }
                P_perturbed_s.push_back(P_perturbed_sa);
            }
            P_perturbed.push_back(P_perturbed_s);
        }
        
        
        
        // make the perturbed kernel satisfy sum(P(s,a,:)) = 1 for all (s,a)
        vector<vector<numvec>> P_k; P_k.reserve(S);
        for ( int s = 0; s < S; s++ ){
            vector<numvec> P_k_s; P_k_s.reserve(A);
            for ( int a = 0; a < A; a++ ){
                numvec P_k_sa; P_k_sa.reserve(S);
                double P_perturbed_sa_sum = 0.0;
                for ( int s2 = 0; s2 < S; s2++ ){
                    P_perturbed_sa_sum += P_perturbed[s][a][s2];
                }
                if ( P_perturbed_sa_sum > 0.0 ){
                    for ( int s2 = 0; s2 < S; s2++ ){
                        P_k_sa.push_back(P_perturbed[s][a][s2]/P_perturbed_sa_sum);
                    }
                }
                else{
                    P_k_sa = numvec(S, 1.0/(double(S)));
                }
                P_k_s.push_back(P_k_sa);
            }
            P_k.push_back(P_k_s);
        }
        
        P_N.push_back(P_k);
    }
    
    return P_N;
    
    
}





