#include "definitions.h"


// inf norm
bool compare(prec_t a, prec_t b) {
    return abs(a) < abs(b);
}

// infinity norm of vectors
prec_t linf_norm(vector<prec_t>& v) {
    vector<prec_t>::iterator itr = max_element(v.begin(), v.end(), compare);
    return abs(*itr);
}


// L2-norm of vectors
prec_t l2_norm(const vector<prec_t> & u) {
    prec_t accum = 0.0;
    for (size_t i = 0; i < u.size(); i++) {
        accum += u[i] * u[i];
    }
    return sqrt(accum);
}


// cdf of normal distribution
prec_t norm_cdf(prec_t x)
{
    // constants
    double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 = 1.421413741;
    double a4 = -1.453152027;
    double a5 = 1.061405429;
    double p = 0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x) / sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// inverse cdf of normal distribution
prec_t RationalApproximation(prec_t t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = { 2.515517, 0.802853, 0.010328 };
    double d[] = { 1.432788, 0.189269, 0.001308 };
    return t - ((c[2] * t + c[1]) * t + c[0]) /
        (((d[2] * t + d[1]) * t + d[0]) * t + 1.0);
}

// inverse cdf of normal distribution
prec_t norm_ppf(prec_t p)
{
    assert(p >= 0.0 && p <= 1.0);

    // See article above for explanation of this section.
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation(sqrt(-2.0 * log(p)));
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation(sqrt(-2.0 * log(1 - p)));
    }
}


// normal random number
prec_t normal_rand(prec_t mean, prec_t stddev){
    std::random_device rd;
    
    std::mt19937 gen(rd());     // if no seed, use this line
//    std::mt19937 gen(2);    // if use seed, use this line
    
    prec_t sample;
    std::normal_distribution<prec_t> distri(mean, stddev);
    sample = distri(gen);
    return sample;
}

// index of the max element in a numvec
size_t maxIdx(numvec v1){
    size_t maxElementIndex = std::max_element(v1.begin(),v1.end()) - v1.begin();
    
    return maxElementIndex;
}

// index of the max (absolute value) element in a numvec
size_t maxIdx_abs(numvec v){
    std::vector<double>::iterator result;
    result = std::max_element(v.begin(), v.end(), compare);
//    std::cout << "max element (absolute) at: " << std::distance(v.begin(), result) << '\n';
    
    return std::distance(v.begin(), result);
}



// index of the max (absolute value) element in a vector<numvec>
pair<size_t, size_t> maxIdx_abs_mtx(const vector<numvec> &v){
    size_t rowIdx = 0; size_t colIdx = 0;
    prec_t maxAbs = v[0][0];
    for (size_t r = 0; r < v.size(); r++){
        size_t maxColIdx_temp = maxIdx_abs(v[r]);
        if (abs(v[r][maxColIdx_temp]) > maxAbs){
            maxAbs = v[r][maxColIdx_temp];
            rowIdx = r;
            colIdx = maxColIdx_temp;
        }
    }
    
    return { rowIdx, colIdx };
}


// matrix transpose
vector<numvec> transpose(const vector<numvec> &Phi){
    size_t n_row = Phi.size();
    size_t n_col = Phi[0].size();
    vector<numvec> Phi_transpose; Phi_transpose.reserve(n_col);
    for (size_t j = 0; j < n_col; j++){
        numvec Phi_transpose_j; Phi_transpose_j.reserve(n_row);
        for (size_t i = 0; i < n_row; i++){
            Phi_transpose_j.push_back(Phi[i][j]);
        }
        Phi_transpose.push_back(Phi_transpose_j);
    }
    
    return Phi_transpose;
}


// print matrix
void printMat( const vector<numvec> &mat1 )
{
    size_t nrow = mat1.size();
    size_t ncol = mat1[0].size();
    for (size_t i = 0; i < nrow; i++){
        for (size_t j = 0; j < ncol; j++){
            cout << mat1[i][j] << " ";
        }
        cout << endl;
    }
}


void printVec(const numvec &vec1){
    size_t sz = vec1.size();
    for (size_t i = 0; i < sz; i++){
        cout << vec1[i] << endl;
    }
}


double average(const numvec & v){
    if(v.empty()){
        return 0;
    }
    int len = v.size();
    double sum_v = 0.0;
    for ( int i = 0; i < len; i++ ){
        sum_v += v[i];
    }
    double ave_v = sum_v/(double(len));
    return ave_v;
}


double percentile(const numvec &vec1, double perc){
    numvec vec_temp = vec1;
    std::sort(vec_temp.begin(), vec_temp.end());
    double percNum;
    int len = vec1.size();
    double b = 1.0 / (double(len) - 1.0);
    int k = floor(perc/b);
    if ( k == len ){
        percNum = vec_temp[len];
    }
    else {
        double eps = perc - (double(k) * b);
        percNum = vec_temp[k] + ((vec_temp[k+1] - vec_temp[k]) * (eps / b));
    }
    return percNum;
}


vector<vector<numvec>> read_kernel(string fileName, int S, int A){
    int vec_len = S * A * S;
    numvec vec; vec.reserve(vec_len);
    std::ifstream  ifile(fileName);

    std::string line; // we read the full line here
    while (std::getline(ifile, line)) // read the current line
    {
        std::istringstream iss{line}; // construct a string stream from line

        // read the tokens from current line separated by comma
        std::vector<std::string> tokens; // here we store the tokens
        std::string token; // current token
        while (std::getline(iss, token, ','))
        {
            tokens.push_back(token); // add the token to the vector
        }

        // map the tokens into our variables, this applies to your scenario
        prec_t w1 = stod(tokens[0]);
        vec.push_back(w1);

    }
    
    vector<vector<numvec>> P = tensor_kernel(vec, S, A);
    
    return P;
}


vector<vector<numvec>> tensor_kernel(const numvec &vec, int S, int A){
    vector<vector<numvec>> P; P.reserve(S);
    for ( int s = 0; s < S; s++ ){
        vector<numvec> P_s; P_s.reserve(A);
        for ( int a = 0; a < A; a++ ){
            numvec P_sa; P_sa.reserve(S);
            for ( int s2 = 0; s2 < S; s2++ ){
                P_sa.push_back(vec[(s*(A*S))+(a*S)+s2]);
            }
            P_s.push_back(P_sa);
        }
        P.push_back(P_s);
    }
    return P;
}


numvec read_kernel_vec(string fileName, int S, int A){
    int vec_len = S * A * S;
    numvec vec; vec.reserve(vec_len);
    std::ifstream  ifile(fileName);

    std::string line; // we read the full line here
    while (std::getline(ifile, line)) // read the current line
    {
        std::istringstream iss{line}; // construct a string stream from line

        // read the tokens from current line separated by comma
        std::vector<std::string> tokens; // here we store the tokens
        std::string token; // current token
        while (std::getline(iss, token, ','))
        {
            tokens.push_back(token); // add the token to the vector
        }

        // map the tokens into our variables, this applies to your scenario
        prec_t w1 = stod(tokens[0]);
        vec.push_back(w1);

    }

    
    return vec;
}


numvec dirichlet(const numvec &alpha, int k) {
    numvec randVec; randVec.reserve(k);
    using Gamma = std::gamma_distribution<double>;
    Gamma gamma;
    std::random_device rd;
    numvec y; y.reserve(k);
    double sum=0;
    for (int i=0; i<k; ++i) {
        double randNum = gamma(rd, Gamma::param_type(alpha[i], 1));
        y.push_back(randNum);
        sum += randNum;
    }
    for (int i=0; i<k; ++i) {
        randVec.push_back(y[i]/sum);
    }
    return randVec;
}


vector<int> discrete_int(const numvec &probabilities_in, const vector<int> &samples_in, int outputSize){
    int len = probabilities_in.size();
    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.0, 1.0);
    numvec breakPoints; breakPoints.reserve(len+1);
    breakPoints.push_back(0.0);
    for ( int i = 0; i < len-1; i++ ){
        double sum_temp = 0.0;
        for ( int j = 0; j < i+1; j++ ){
            sum_temp += probabilities_in[j];
        }
        breakPoints.push_back(sum_temp);
    }
    breakPoints.push_back(1.0);
    
    vector<int> randNums; randNums.reserve(outputSize);
    for ( int i = 0; i < outputSize; i++ ){
        double randNum_0to1 = distribution(generator);
//        cout << "randNum_0to1 = " << randNum_0to1 << endl;
        for ( int j = 0; j < len; j++ ){
            if (randNum_0to1 >= breakPoints[j] && randNum_0to1 < breakPoints[j+1]){
//                cout << breakPoints[j] << " <= " << randNum_0to1 << " < " << breakPoints[j+1] << ", sample is " << samples_in[j] << endl;
                randNums.push_back(samples_in[j]);
                break;
            }
        }
    }
    
    return randNums;
}



numvec uniform_vec(double lb, double ub, int len){
    default_random_engine generator;
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(lb, ub);
    
    numvec vec; vec.reserve(len);
    for ( int i = 0; i < len; i++ ){
        vec.push_back(distribution(generator));
    }
    
    return vec;
}








