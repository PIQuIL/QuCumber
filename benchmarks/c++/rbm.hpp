#ifndef QST_RBM_HPP
#define QST_RBM_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>

namespace qst{

// RBM class
class Rbm{

    int nv_;                        // Number of visible units
    int nh_;                        // Number of hidden units
    int npar_;                      // Number of parameters
    int nchains_;                   // Number of sampling chains
    
    Eigen::MatrixXd v_;             // Visible states
    Eigen::MatrixXd h_;             // Hidden states
    Eigen::MatrixXd probv_given_h_; // Visible probabilities
    Eigen::MatrixXd probh_given_v_; // Hidden probabilities

    Eigen::MatrixXd W_;             // Weights
    Eigen::VectorXd b_;             // Visible fields
    Eigen::VectorXd c_;             // Hidden fields
    
    Eigen::VectorXd gamma_;         // Container for hidden contribution to quantities 
    
    std::mt19937 rgen_;             // Random number generator
    
public:
    // Contructor
    Rbm(Parameters &par):nv_(par.nv_),nh_(par.nh_){
        npar_=nv_+nh_+nv_*nh_;
        nchains_ = par.nc_;
        v_.setZero(nchains_,nv_);
        h_.setZero(nchains_,nh_);
        probv_given_h_.resize(nchains_,nv_);
        probh_given_v_.resize(nchains_,nh_);
        W_.resize(nh_,nv_);
        b_.resize(nv_);
        c_.resize(nh_);
        gamma_.resize(nh_);
        rgen_.seed(13579);
        //std::random_device rd;
        //rgen_.seed(rd());
    }

    // Private members access functions
    inline int Nvisible()const{
        return nv_;
    }
    inline int Nhidden()const{
        return nh_;
    }
    inline int Npar()const{
        return npar_;
    }
    inline int Nchains(){
        return nchains_;
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return v_.row(s);
    }
    
    // Set the visible layer state
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        v_=v;
    }
   
    // Initialize the network parameters
    void InitRandomPars(int seed,double sigma){
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(0,sigma);
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                W_(i,j)=distribution(generator);
            }
        }
        for(int j=0;j<nv_;j++){
            b_(j)=distribution(generator);
        }
        for(int i=0;i<nh_;i++){
            c_(i)=distribution(generator);
        }
    }

    // Compute derivative of the effective visible energy
    Eigen::VectorXd VisEnergyGrad(const Eigen::VectorXd & v){
        Eigen::VectorXd der(npar_);
        int p=0;
        logistic(W_*v+c_,gamma_);
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                der(p)=gamma_(i)*v(j);
                p++;
            }
        }
        for(int j=0;j<nv_;j++){
            der(p)=v(j);
            p++;
        }
        for(int i=0;i<nh_;i++){
            der(p)=gamma_(i);
            p++;
        } 
        return -der;
    }
   
    // Return the probability for state v
    inline double prob(const Eigen::VectorXd & v){
        return exp(LogVal(v));
    }
    
    // Value of the logarithm of the RBM probability
    inline double LogVal(const Eigen::VectorXd & v){
        ln1pexp(W_*v+c_,gamma_);
        return v.dot(b_)+gamma_.sum();
    }
    
    // Conditional Probabilities 
    void ProbHiddenGivenVisible(const Eigen::MatrixXd &v,Eigen::MatrixXd &probs){
        logistic((v*W_.transpose()).rowwise() + c_.transpose(),probs);
    }
    void ProbVisibleGivenHidden(const Eigen::MatrixXd &h,Eigen::MatrixXd &probs){
        logistic((h*W_).rowwise() + b_.transpose(),probs);
    }

    // Sample one layer 
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
        for(int k=0;k<steps;k++){
            ProbHiddenGivenVisible(v_,probh_given_v_);
            SampleLayer(h_,probh_given_v_);
            ProbVisibleGivenHidden(h_,probv_given_h_);
            SampleLayer(v_,probv_given_h_);
        }
    }
   
    // Get RBM parameters
    Eigen::VectorXd GetParameters(){
        Eigen::VectorXd pars(npar_);
        int p=0;
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                pars(p)=W_(i,j);
                p++;
            }
        }
        for(int j=0;j<nv_;j++){
            pars(p)=b_(j);
            p++;
        }
        for(int i=0;i<nh_;i++){
            pars(p)=c_(i);
            p++;
        }
        return pars;
    }
    
    // Set RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        int p=0;
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                W_(i,j)=pars(p);
                p++;
            }
        }
        for(int j=0;j<nv_;j++){
            b_(j)=pars(p);
            p++;
        }
        for(int i=0;i<nh_;i++){
            c_(i)=pars(p);
            p++;
        }
    }

    // Read weights from file 
    void LoadWeights(std::string & name){
        std::ifstream fin(name);
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                fin >> W_(i,j);
            }
        }
        for(int j=0;j<nv_;j++){
            fin >> b_(j);
        }
        for(int i=0;i<nh_;i++){
            fin >> c_(i);
        }
        fin.close(); 
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
