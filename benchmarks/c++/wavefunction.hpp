#ifndef QST_WAVEFUNCTION_HPP
#define QST_WAVEFUNCTION_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <iostream>

namespace qst{

class Wavefunction{

    int N_;        // Number of degrees of freedom (visible units)
    //int nh_;      // Number of hidden units
    int npar_;    // Number of parameters

    Rbm rbmAm_;   // RBM for the amplitude
    Rbm rbmPh_;   // RBM for the phases

    const std::complex<double> I_; // Imaginary unit
    
    //Random number generator 
    std::mt19937 rgen_;
    
public:
    // Constructor 
    Wavefunction(Parameters &par):rbmAm_(par),
                                  rbmPh_(par),
                                  I_(0,1){
        npar_ = rbmAm_.Npar() + rbmPh_.Npar();  // Total number of parameters
        N_ = rbmAm_.Nvisible();
        std::random_device rd;
        //rgen_.seed(rd());
        rgen_.seed(13579);
    }

    //typedef std::complex<double> StateType;
    
    // Private members access functions
    inline int N()const{
        return N_;
    }
    inline int Npar()const{
        return npar_;
    }
    inline int Nchains(){
        return rbmAm_.Nchains();
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return rbmAm_.VisibleStateRow(s);
    }

    // Set the visible layer state
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        rbmAm_.SetVisibleLayer(v);
    }
 
    // Initialize the network parameters    
    void InitRandomPars(int seed,double sigma){
        rbmAm_.InitRandomPars(seed,sigma);
        rbmPh_.InitRandomPars(seed,sigma);
    }
    
    //Compute derivative of log-probability
    Eigen::VectorXd LambdaGrad(const Eigen::VectorXd & v){
        //VectorXd der(npar_);
        //der<<rbmAm_.DerLog(v),rbmPh_.DerLog(v);
        //return der;
        return rbmAm_.VisEnergyGrad(v);
    }
    Eigen::VectorXd MuGrad(const Eigen::VectorXd & v){
        return rbmPh_.VisEnergyGrad(v);
    }
   
    Eigen::VectorXd Grad(const Eigen::VectorXd & v){
        Eigen::VectorXd der(npar_);
        der<<rbmAm_.VisEnergyGrad(v),rbmPh_.VisEnergyGrad(v);
        return der;
    }

    //Conditional Probabilities 
    void ProbHiddenGivenVisible(const Eigen::MatrixXd & v,Eigen::MatrixXd & probs){
        rbmAm_.ProbHiddenGivenVisible(v,probs);
    }
    void ProbVisibleGivenHidden(const Eigen::MatrixXd & h,Eigen::MatrixXd & probs){
        rbmAm_.ProbVisibleGivenHidden(h,probs);
    }

    // Amplitude
    double amplitude(const Eigen::VectorXd & v){
        return std::sqrt(rbmAm_.prob(v));//exp(0.5*rbmAm_.LogVal(v));         
    }
    // Phase
    double phase(const Eigen::VectorXd & v){
        return std::log(rbmPh_.prob(v));
    }
    // Wavefunction
    std::complex<double> psi(const Eigen::VectorXd & v){
        return amplitude(v)*exp(0.5*I_*phase(v));
    }

    // Sample the one layer 
    void SampleLayer(Eigen::MatrixXd & hv,const Eigen::MatrixXd & probs){
        std::uniform_real_distribution<double> distribution(0,1);
        for(int s=0;s<hv.rows();s++){
            for(int i=0;i<hv.cols();i++){
                hv(s,i)=distribution(rgen_)<probs(s,i);
            }
        }
    }
    // Perform k steps of Gibbs sampling
    void Sample(int steps){
        rbmAm_.Sample(steps);
    }
 
    //Get RBM parameters
    Eigen::VectorXd GetParameters(){
        Eigen::VectorXd pars(npar_);
        pars<<rbmAm_.GetParameters(),rbmPh_.GetParameters();
        return pars;
    }

    // Set RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        Eigen::VectorXd parsAm(npar_/2);
        Eigen::VectorXd parsPh(npar_/2);

        for(int i=0;i<npar_/2;i++){
            parsAm(i)=pars(i);
            parsPh(i)=pars(i+npar_/2);
        }
        rbmAm_.SetParameters(parsAm);
        rbmPh_.SetParameters(parsPh);
    }

    // Functions 
    inline void logistic(const Eigen::VectorXd & x,Eigen::VectorXd & y){
        for(int i=0;i<x.size();i++){
            y(i)=1./(1.+std::exp(-x(i)));;
        }
    }
    inline void logistic(const Eigen::MatrixXd & x,Eigen::MatrixXd & y){
        for(int i=0;i<x.rows();i++){
            for(int j=0;j<x.cols();j++){
                //y(i,j)=logistic(x(i,j));
                y(i,j)=1./(1.+std::exp(-x(i,j)));
            }
        }
    }
    inline double ln1pexp(double x){
        if(x>30){
            return x;
        }
        return std::log1p(std::exp(x));
    }
    void ln1pexp(const Eigen::VectorXd & x,Eigen::VectorXd & y){
        for(int i=0;i<x.size();i++){
            y(i)=ln1pexp(x(i));
        }
    }
};
}

#endif
