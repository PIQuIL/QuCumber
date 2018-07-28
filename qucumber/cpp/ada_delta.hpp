#ifndef QST_ADADELTA_HPP
#define QST_ADADELTA_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace qst{

class AdaDelta{

    int npar_;
    //decay constant
    double rho_;
    //small parameter
    double eps_;
    double dropout_p_;
    double l2reg_;

    Eigen::VectorXd Eg2_;
    Eigen::VectorXd Edx2_;

    std::mt19937 rgen_;
    Eigen::VectorXd deltaP_;    // Parameter change 

public:

    AdaDelta(Parameters &par,double rho=0.95,double eps=1.0e-6,double l2reg=0.0):rho_(rho),eps_(eps),l2reg_(l2reg){
        npar_=-1;
    }
    
    void SetNpar(int npar){
        npar_=npar;
        deltaP_.resize(npar_);
        //avg_grad.resize(npar);
        Eg2_.setZero(npar);
        Edx2_.setZero(npar);
    }
   
    void Update(Eigen::VectorXd & pars){
        assert(npar_>0);
        
        std::uniform_real_distribution<double> distribution(0,1);
        for(int i=0;i<npar_;i++){
            if(distribution(rgen_)>dropout_p_){
                pars(i)=pars(i) - deltaP_(i) ;
            }
        }
    }
 
    void getUpdates(const Eigen::VectorXd & grad){
        assert(npar_>0);
        
        Eg2_=rho_*Eg2_+(1.-rho_)*grad.cwiseAbs2();
        
        Eigen::VectorXd Dx(npar_);
        
        for(int i=0;i<npar_;i++){
            Dx(i)=-std::sqrt(Edx2_(i)+eps_)*grad(i);
            Dx(i)/=std::sqrt(Eg2_(i)+eps_);
            deltaP_(i)=-Dx(i);
        }
        Edx2_=rho_*Edx2_+(1.-rho_)*Dx.cwiseAbs2();
    }
    
    void Reset(){
      Eg2_=Eigen::VectorXd::Zero(npar_);
      Edx2_=Eigen::VectorXd::Zero(npar_);
    }
};


}

#endif
