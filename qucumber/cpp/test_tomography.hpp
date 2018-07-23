#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <string>
#include <map>

#ifndef QST_TESTTOMOGRPHY_HPP
#define QST_TESTTOMOGRPHY_HPP

namespace qst{

//Quantum State Tomography class
template<class NNState,class Observer> class Test{

    NNState & NNstate_;                   // Neural network representation of the state
    Observer &obs_;

    int N_;
    int npar_;                          // Number of variational parameters
    int nparLambda_;                    // Number of amplitude variational parameters
    int nparMu_;
    std::string basis_;
    Eigen::VectorXd grad_;         // Gradient 
    Eigen::VectorXcd rotated_grad_;
    std::mt19937 rgen_;                 // Random number generator
    std::map<std::string,Eigen::MatrixXcd> U_;
    Eigen::MatrixXd basis_states_;   
    Eigen::VectorXcd target_psi_;
    std::vector<Eigen::VectorXcd> rotated_wf_;
    std::vector<std::vector<std::string> > basisSet_;

public:
     
    Test(NNState & NNstate,Observer &obs,Parameters &par):NNstate_(NNstate),obs_(obs),N_(NNstate.N()){
            
        npar_=NNstate_.Npar();
        nparLambda_ = NNstate_.NparLambda();
        nparMu_ = npar_ - nparLambda_;
        grad_.resize(npar_);
        rotated_grad_.resize(npar_);
        basis_ = par.basis_;
        basis_states_.resize(1<<N_,N_);
        std::bitset<10> bit;
        // Create the basis of the Hilbert space
        for(int i=0;i<1<<N_;i++){
            bit = i;
            for(int j=0;j<N_;j++){
                basis_states_(i,j) = bit[N_-j-1];
            }
        }

    }

    // Test the derivatives of the KL divergence
    void DerKL(double eps=1.0e-4){
        auto pars = NNstate_.GetParameters();
        obs_.ExactPartitionFunction();
        Eigen::VectorXd derKL(npar_);
        Eigen::VectorXd ders(npar_);
        ders.setZero(npar_);

        //std::cout << NNstate_.LambdaGrad(basis_states_.row(150)) << std::endl;
        //-- ALGORITHMIC DERIVATIVES --//
        //Standard Basis
        for(int j=0;j<1<<N_;j++){
            //Positive phase - Lambda gradient in reference basis
            ders.head(nparLambda_) +=  norm(target_psi_(j))*NNstate_.LambdaGrad(basis_states_.row(j));
            //Negative phase - Lambda gradient in reference basis       
            ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        }
        std::cout <<ders << std::endl;
            
        //if (basis_.compare("std")!=0){
        //    //Rotated Basis
        //    for(int b=1;b<basisSet_.size();b++){
        //        for(int j=0;j<1<<N_;j++){
        //            NNstate_.rotatedGrad(basisSet_[b],basis_states_.row(j),U_,derKL);
        //            ////Positive phase - Lambda gradient in basis b
        //            //ders.head(nparLambda_) += norm(rotated_wf_[b-1](j))*derKL.head(nparLambda_).real();
        //            ////Positive phase - Mu gradient in basis b
        //            //ders.tail(nparMu_) -= norm(rotated_wf_[b-1](j))*derKL.tail(nparMu_).imag();
        //            ders += norm(rotated_wf_[b-1](j))*derKL; 
        //            //Negative phase - Lambda gradient in basis b (identical to the reference basis
        //            ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        //        }
        //    }
        //}
        ////-- NUMERICAL DERIVATIVES --//
        //for(int p=0;p<npar_;p++){
        //    pars(p)+=eps;
        //    NNstate_.SetParameters(pars);
        //    double valp=0.0;
        //    obs_.ExactPartitionFunction();
        //    obs_.ExactKL();
        //    valp = obs_.KL_;
        //    pars(p)-=2*eps;
        //    NNstate_.SetParameters(pars);
        //    double valm=0.0;
        //    obs_.ExactPartitionFunction();
        //    obs_.ExactKL();
        //    valm = obs_.KL_;
        //    pars(p)+=eps;
        //    double numder=(-valm+valp)/(eps*2);
        //    std::cout<<"Derivative wrt par "<<p<<". Grad =: "<<ders(p)<<" Numerical = : "<<numder<<std::endl;
        //}
    }

    // Test the derivatives of the KL divergence
    void DerNLL(Eigen::MatrixXd &data,double eps=1.0e-4){
        auto pars = NNstate_.GetParameters();
        obs_.ExactPartitionFunction();
        Eigen::VectorXcd derKL(npar_);
        Eigen::VectorXd ders(npar_);
        ders.setZero(npar_);

        //-- ALGORITHMIC DERIVATIVES --//
        //Standard Basis
        //std::cout << data.rows() <<std::endl;
        for(int j=0;j<data.rows();j++){
            //Positive phase - Lambda gradient in reference basis
            ders.head(nparLambda_) +=  NNstate_.LambdaGrad(data.row(j))/float(data.rows());
        }
        //for(int j=0;j<1<<N_;j++){ 
        //    ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        //    //std::cout << (NNstate_.LambdaGrad(basis_states_.row(j)).head(10)).transpose() << std::endl<<std::endl;
        //}
        //NNstate_.SetVisibleLayer(data);
        NNstate_.Sample(100);
        std::cout << NNstate_.VisibleStateRow(0) << std::endl;
        for(int k=0;k<NNstate_.Nchains();k++){ 
            ders.head(nparLambda_) -= NNstate_.LambdaGrad(NNstate_.VisibleStateRow(k))/double(NNstate_.Nchains());
        }
        //if (basis_.compare("std")!=0){
        //    //Rotated Basis
        //    for(int b=1;b<basisSet_.size();b++){
        //        for(int j=0;j<1<<N_;j++){
        //            NNstate_.rotatedGrad(basisSet_[b],basis_states_.row(j),U_,derKL);
        //            //Positive phase - Lambda gradient in basis b
        //            ders.head(nparLambda_) += norm(rotated_wf_[b-1](j))*derKL.head(nparLambda_).real();
        //            //Positive phase - Mu gradient in basis b
        //            ders.tail(nparMu_) -= norm(rotated_wf_[b-1](j))*derKL.tail(nparMu_).imag();
        //            //Negative phase - Lambda gradient in basis b (identical to the reference basis
        //            ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        //        }
        //    }
        //}
        //-- NUMERICAL DERIVATIVES --//
        for(int p=0;p<npar_;p++){
            pars(p)+=eps;
            NNstate_.SetParameters(pars);
            double valp=0.0;
            obs_.ExactPartitionFunction();
            obs_.NLL(data);
            valp = obs_.NLL_;
            pars(p)-=2*eps;
            NNstate_.SetParameters(pars);
            double valm=0.0;
            obs_.ExactPartitionFunction();
            obs_.NLL(data);
            valm = obs_.NLL_;
            pars(p)+=eps;
            double numder=(-valm+valp)/(eps*2);
            std::cout<<"Derivative wrt par "<<p<<". Grad =: "<<ders(p)<<" Numerical = : "<<numder<<std::endl;
        }
    }
    //Set the value of the target wavefunction
    void setBasisRotations(std::map<std::string,Eigen::MatrixXcd> & U){
        U_ = U;
    }
    void setBasis(std::vector<std::vector<std::string> > basis) {
        basisSet_ = basis;
    }
    //Set the value of the target wavefunction
    void setWavefunction(Eigen::VectorXcd & psi){
        target_psi_.resize(1<<N_);
        for(int i=0;i<1<<N_;i++){
            target_psi_(i) = psi(i);
        }
    }
    void setRotatedWavefunctions(std::vector<Eigen::VectorXcd> & psi){
        for(int b=0;b<psi.size();b++){
            rotated_wf_.push_back(psi[b]);
        }
    }
};
}

#endif
