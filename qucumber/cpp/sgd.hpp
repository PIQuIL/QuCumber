#ifndef QST_SGD_HPP
#define QST_SGD_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>

namespace qst{

// Stochastic gradient descent
class Sgd{

    double eta_;                // Learning rate
    int npar_;                  // Number of parameters
    double l2reg_;              // L2 regularization
    Eigen::VectorXd deltaP_;    // Parameter change 
    std::mt19937 rgen_;         // Random number

public:

    Sgd(Parameters &par):eta_(par.lr_),l2reg_(par.l2_){
        npar_=-1;
    }
    
    // Set the number of parameter
    void SetNpar(int npar){
        npar_=npar;
        deltaP_.resize(npar_);
    }
    
    // Update the parameters
    void Update(Eigen::VectorXd & pars){
        assert(npar_>0);
        
        std::uniform_real_distribution<double> distribution(0,1);
        for(int i=0;i<npar_;i++){
            pars(i)=pars(i) - (deltaP_(i)+l2reg_*pars(i))*eta_;
        }
    }
    
    // Read update from gradient calculator
    void getUpdates(const Eigen::MatrixXd & derLog){
        deltaP_ = derLog;
    }
    
    void Reset(){
        
    }
};


}

#endif
