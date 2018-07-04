#ifndef QST_WAVEFUNCTIONPOSITIVE_HPP
#define QST_WAVEFUNCTIONPOSITIVE_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <iostream>

namespace qst{

class WavefunctionPositive{

    int N_;            // Number of degrees of freedom (visible units)
    int npar_;         // Number of parameters
    int nparLambda_;   // Number of amplitude parameters
    Rbm rbmAm_;        // RBM for the amplitude

    const std::complex<double> I_; // Imaginary unit
    
    //Random number generator 
    std::mt19937 rgen_;
    
public:
    // Constructor 
    WavefunctionPositive(Parameters &par):rbmAm_(par),
                                  I_(0,1){
        npar_ = rbmAm_.Npar();  // Total number of parameters
        nparLambda_ = npar_;
        N_ = rbmAm_.Nvisible();
        std::random_device rd;
        //rgen_.seed(rd());
        rgen_.seed(13579);
    }
    
    // Private members access functions
    inline int N()const{
        return N_;
    }
    inline int Npar()const{
        return npar_;
    }
    inline int NparLambda()const{
        return nparLambda_;
    }
    inline int NparMu()const{
        return 0;
    }
    inline int Nchains(){
        return rbmAm_.Nchains();
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return rbmAm_.VisibleStateRow(s);
    }

    // Set the state of the wavefunction's degrees of freedom
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        rbmAm_.SetVisibleLayer(v);
    }
 
    // Initialize the wavefunction parameters    
    void InitRandomPars(int seed,double sigma){
        rbmAm_.InitRandomPars(seed,sigma);
    }
    
    // Amplitude
    double amplitude(const Eigen::VectorXd & v){
        return std::sqrt(rbmAm_.prob(v));//exp(0.5*rbmAm_.LogVal(v));         
    }
    // Phase
    double phase(const Eigen::VectorXd & v){
        return 0.0;
    }
    // Psi
    std::complex<double> psi(const Eigen::VectorXd & v){
        return amplitude(v);
    }

    //---- SAMPLING ----/

    // Perform k steps of Gibbs sampling
    void Sample(int steps){
        rbmAm_.Sample(steps);
    }
   
    //---- DERIVATIVES ----//
    
    //Compute gradient of effective energy wrt Lambda 
    Eigen::VectorXd LambdaGrad(const Eigen::VectorXd & v){
        return rbmAm_.VisEnergyGrad(v);
    }
    //Compute gradient of effective energy wrt all parameters
    Eigen::VectorXd Grad(const Eigen::VectorXd & v){
        return rbmAm_.VisEnergyGrad(v);
    }
    //Dummy function
    void rotatedGrad(const std::vector<std::string> & basis,
                            const Eigen::VectorXd & state,//VectorRbmT & gradR){
                            std::map<std::string,Eigen::MatrixXcd> & Unitaries,
                            Eigen::VectorXcd &gradR ){
    }

    //---- UTILITIES ----//

    //Get RBM parameters
    Eigen::VectorXd GetParameters(){
        return rbmAm_.GetParameters();
    }

    // Set RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        rbmAm_.SetParameters(pars);
    }
    
    void LoadWeights(std::string &fileName){
        std::ifstream fin(fileName);
        rbmAm_.LoadWeights(fin);
    }
};
}

#endif
