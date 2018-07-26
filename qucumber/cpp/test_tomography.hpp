#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <string>
#include <map>
#include <sstream>
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
    void DerKL(double eps,Eigen::VectorXd &alg_ders,Eigen::VectorXd &num_ders){
        auto pars = NNstate_.GetParameters();
        obs_.ExactPartitionFunction();
        Eigen::VectorXd derKL(npar_);
        alg_ders.setZero(npar_);
        num_ders.setZero(npar_);
        
        //-- ALGORITHMIC DERIVATIVES --//
        //Standard Basis
        for(int j=0;j<1<<N_;j++){
            //Positive phase - Lambda gradient in reference basis
            alg_ders.head(nparLambda_) +=  norm(target_psi_(j))*NNstate_.LambdaGrad(basis_states_.row(j));
            //Negative phase - Lambda gradient in reference basis       
            alg_ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
        }
          
        if (basis_.compare("std")!=0){
            //Rotated Basis
            for(int b=1;b<basisSet_.size();b++){
                for(int j=0;j<1<<N_;j++){
                    //Positive phase
                    NNstate_.rotatedGrad(basisSet_[b],basis_states_.row(j),U_,derKL);
                    alg_ders += norm(rotated_wf_[b-1](j))*derKL; 
                    //Negative phase - Lambda gradient in basis b (identical to the reference basis
                    alg_ders.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
                }
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
            num_ders(p)=(-valm+valp)/(eps*2);
        }
    }

    // Test the derivatives of the KL divergence
    void DerNLL(double eps,int nchains,Eigen::MatrixXd &data_samples,std::vector<std::vector<std::string> > &data_bases,Eigen::VectorXd &alg_ders_nosample,Eigen::VectorXd &alg_ders_sample,Eigen::VectorXd &num_ders){
        auto pars = NNstate_.GetParameters();
        obs_.ExactPartitionFunction();
        Eigen::VectorXcd derKL(npar_);
        alg_ders_nosample.setZero(npar_);
        alg_ders_sample.setZero(npar_);
        num_ders.setZero(npar_);

        //-- ALGORITHMIC DERIVATIVES --//
        //Standard Basis
        for(int j=0;j<data_samples.rows();j++){
            //Positive phase - Lambda gradient in reference basis
            alg_ders_nosample.head(nparLambda_) +=  NNstate_.LambdaGrad(data_samples.row(j))/float(data_samples.rows());
            alg_ders_sample.head(nparLambda_) +=  NNstate_.LambdaGrad(data_samples.row(j))/float(data_samples.rows());
        }

        for(int j=0;j<1<<N_;j++){ 
            alg_ders_nosample.head(nparLambda_) -= NNstate_.LambdaGrad(basis_states_.row(j))*norm(NNstate_.psi(basis_states_.row(j)))/obs_.Z_;
            //std::cout << (NNstate_.LambdaGrad(basis_states_.row(j)).head(10)).transpose() << std::endl<<std::endl;
        }
        NNstate_.SetVisibleLayer(data_samples.topRows(nchains));
        NNstate_.Sample(100);
        //std::cout << NNstate_.VisibleStateRow(0) << std::endl;
        for(int k=0;k<NNstate_.Nchains();k++){ 
            alg_ders_sample.head(nparLambda_) -= NNstate_.LambdaGrad(NNstate_.VisibleStateRow(k))/double(NNstate_.Nchains());
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
            obs_.NLL(data_samples,data_bases);
            valp = obs_.NLL_;
            pars(p)-=2*eps;
            NNstate_.SetParameters(pars);
            double valm=0.0;
            obs_.ExactPartitionFunction();
            obs_.NLL(data_samples,data_bases);
            valm = obs_.NLL_;
            pars(p)+=eps;
            num_ders(p)=(-valm+valp)/(eps*2);
        }
    }

    void RunDerCheck(double n_hidden,Eigen::MatrixXd &data_samples,std::vector<std::vector<std::string> > &data_bases,double eps=1.0e-4,int nchains=100){
        Eigen::VectorXd alg_derKL(npar_);
        Eigen::VectorXd num_derKL(npar_);
        Eigen::VectorXd alg_derNLL_sample(npar_);
        Eigen::VectorXd alg_derNLL_nosample(npar_);
        Eigen::VectorXd num_derNLL(npar_);
        DerKL(eps,alg_derKL,num_derKL); 
        DerNLL(eps,nchains,data_samples,data_bases,alg_derNLL_nosample,alg_derNLL_sample,num_derNLL);
        
        std::cout<< "Running KL derivatives check.."<<std::endl<<std::endl;
        std::cout<< "Network: rbm_am (AMPLITUDE MACHINE)"<<std::endl<<std::endl;
        int p=0;
        std::cout<<"Weights:"<<std::endl;
        std::cout<<"Numerical KL\t\tAlgorithm KL"<<"\t\t\t\t"<<"Numerical NLL"<<"\t\t"<<"Alg NLL nosample"<<"\t"<<"Alg NLL sample"<<std::endl<<std::endl;
        for(int i=0;i<n_hidden;i++){
            for(int j=0;j<NNstate_.N();j++){
                std::cout<<std::setprecision(8)<<num_derKL(p) <<"\t\t"<<std::setprecision(8)<<alg_derKL(p)<<"\t\t\t\t";
                std::cout<<std::setprecision(8)<<num_derNLL(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_nosample(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_sample(p)<<std::endl; 
                p++;
            }
        }
        std::cout<<std::endl;
        std::cout<<"Visible Bias:"<<std::endl;
        std::cout<<"Numerical KL\t\tAlgorithm KL"<<"\t\t\t\t"<<"Numerical NLL"<<"\t\t"<<"Alg NLL nosample"<<"\t"<<"Alg NLL sample"<<std::endl<<std::endl;
        for(int j=0;j<NNstate_.N();j++){
            std::cout<<std::setprecision(8)<<num_derKL(p) <<"\t\t"<<std::setprecision(8)<<alg_derKL(p)<<"\t\t\t\t";
            std::cout<<std::setprecision(8)<<num_derNLL(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_nosample(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_sample(p)<<std::endl; 
            p++;
        }
        std::cout<<std::endl;
        std::cout<<"Hidden Bias:"<<std::endl;
        std::cout<<"Numerical KL\t\tAlgorithm KL"<<"\t\t\t\t"<<"Numerical NLL"<<"\t\t"<<"Alg NLL nosample"<<"\t"<<"Alg NLL sample"<<std::endl<<std::endl;
        for(int i=0;i<n_hidden;i++){
            std::cout<<std::setprecision(8)<<num_derKL(p) <<"\t\t"<<std::setprecision(8)<<alg_derKL(p)<<"\t\t\t\t";
            std::cout<<std::setprecision(8)<<num_derNLL(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_nosample(p)<<"\t\t"<<std::setprecision(8)<<alg_derNLL_sample(p)<<std::endl; 
            p++;
        }
        std::cout<<std::endl;
        if (p+1<npar_){
            std::cout<< "Network: rbm_ph (PHASE MACHINE)"<<std::endl<<std::endl;
            std::cout<<"Weights:"<<std::endl;
            for(int i=0;i<n_hidden;i++){
                for(int j=0;j<NNstate_.N();j++){
                    std::cout<<std::setprecision(8)<<num_derKL(p)<<std::setw(20)<<std::setprecision(8)<<alg_derKL(p)<<std::endl;
                    p++;
                }
            }
            std::cout<<std::endl;
            std::cout<<"Visible Bias:"<<std::endl;
            for(int j=0;j<NNstate_.N();j++){
                std::cout<<std::setprecision(8)<<num_derKL(p)<<std::setw(20)<<std::setprecision(8)<<alg_derKL(p)<<std::endl;
                p++;
            }
            std::cout<<std::endl;
            std::cout<<"Hidden Bias:"<<std::endl;
            for(int i=0;i<n_hidden;i++){
                std::cout<<std::setprecision(8)<<num_derKL(p)<<std::setw(20)<<std::setprecision(8)<<alg_derKL(p)<<std::endl;
                p++;
            }
            std::cout<<std::endl;
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
