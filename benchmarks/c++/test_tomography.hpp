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
        grad_.resize(npar_);
        rotated_grad_.resize(npar_);
        
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
        Eigen::VectorXcd derKL(npar_);
        Eigen::VectorXd ders(npar_);
        ders.setZero(npar_);

        //-- ALGORITHMIC DERIVATIVES --//
        //Standard Basis
        for(int j=0;j<1<<N_;j++){
            //Positive phase - Lambda gradient in reference basis
            ders.head(npar_/2) +=  norm(target_psi_(j))*NNstate_.LambdaGrad(basis_states_.row(j));
            //Negative phase - Lambda gradient in reference basis
            ders.head(npar_/2) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        }
        //Rotated Basis
        for(int b=1;b<basisSet_.size();b++){
            for(int j=0;j<1<<N_;j++){
                NNstate_.rotatedGrad(basisSet_[b],basis_states_.row(j),U_,derKL);
                //Positive phase - Lambda gradient in basis b
                ders.head(npar_/2) += norm(rotated_wf_[b-1](j))*derKL.head(npar_/2).real();
                //Positive phase - Mu gradient in basis b
                ders.tail(npar_/2) -= norm(rotated_wf_[b-1](j))*derKL.tail(npar_/2).imag();
                //Negative phase - Lambda gradient in basis b (identical to the reference basis
                ders.head(npar_/2) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
            }
        }

        //-- NUMERICAL DERIVATIVES --//
        for(int p=0;p<npar_;p++){
            pars(p)+=eps;
            NNstate_.SetParameters(pars);
            double valp=0.0;
            obs_.ExactPartitionFunction();
            obs_.ExactKL();
            valp = obs_.KL_;
            pars(p)-=2*eps;
            NNstate_.SetParameters(pars);
            double valm=0.0;
            obs_.ExactPartitionFunction();
            obs_.ExactKL();
            valm = obs_.KL_;
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
            std::cout<<rotated_wf_[b]<<std::endl;
        }
    }
};
}

#endif
