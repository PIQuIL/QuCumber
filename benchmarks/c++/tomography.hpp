#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <string>
#include <map>

#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

//Quantum State Tomography class
template<class NNState,class Optimizer> class Tomography {

    NNState & NNstate_;                   // Neural network representation of the state
    Optimizer & opt_;                   // Optimizer
    int N_;
    //int nh_;
    int npar_;                          // Number of variational parameters
    int bs_;                            // Batch size
    int cd_;                            // Number of Gibbs stepos in contrastive divergence
    int epochs_;                        // Number of training iterations
    double lr_;                         // Learning rate
    double l2_;                         // L2 regularization constant

    Eigen::VectorXd grad_;              // Gradients 
    Eigen::VectorXcd rotated_grad_;     // Rotated gradients
    std::mt19937 rgen_;                 // Random number generator
    
    double negative_log_likelihood_;    // Negative log-likelihood
    double overlap_;                    // Overlap
    double Z_;                          // Partition function
    Eigen::MatrixXd basis_states_;      // Hilbert space basis
 
    Eigen::VectorXcd wf_;                               // Target wavefunction
    std::map<std::string,Eigen::MatrixXcd> U_;          // Structure containin the single unitary rotations

    //std::vector<std::vector<std::string> > basisSet_;   // Possible bases       
    //std::vector<Eigen::VectorXcd> rotated_wf_;
public:
    
    Tomography(Optimizer & opt,NNState & NNstate,Parameters &par):opt_(opt),NNstate_(NNstate),N_(NNstate.N()),bs_(par.bs_),cd_(par.cd_) {
            
        npar_=NNstate_.Npar();
        opt_.SetNpar(npar_);
        bs_ = par.bs_;
        cd_ = par.cd_;
        lr_ = par.lr_;
        l2_ = par.l2_;
        epochs_ = par.ep_;
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

    //Compute gradient of KL divergence 
    void Gradient(const Eigen::MatrixXd & batchSamples,const std::vector<std::vector<std::string> >& batchBases){ 
        grad_.setZero();
        Eigen::VectorXd tmp;

        int bID = 0;
        //Positive Phase
        for(int k=0;k<bs_;k++){
            bID = 0;
            for(int j=0;j<N_;j++){ // Check if the basis is the reference one
                if (batchBases[k][j]!="Z"){
                    bID = 1;
                    break;
                }
            }
            if (bID==0){ // Positive phase - Lambda gradient in the reference basis
                grad_.head(npar_/2) += NNstate_.LambdaGrad(batchSamples.row(k))/double(bs_);
            }
            else { // Positive phase - Lambda and Mu gradients for non-trivial bases
                getRotatedGradient(batchBases[k],batchSamples.row(k),rotated_grad_);
                grad_.head(npar_/2) += rotated_grad_.head(npar_/2).real()/double(bs_);
                grad_.tail(npar_/2) -= rotated_grad_.tail(npar_/2).imag()/double(bs_);
            }
        }
        
        //Negative Phase
        NNstate_.Sample(cd_);
        for(int k=0;k<NNstate_.Nchains();k++){
            grad_.head(npar_/2) -= NNstate_.LambdaGrad(NNstate_.VisibleStateRow(k))/double(NNstate_.Nchains());
        }
        opt_.getUpdates(grad_);
    }
    
    // Update rbm parameters
    void UpdateParameters(){
        auto pars=NNstate_.GetParameters();
        opt_.Update(pars);
        NNstate_.SetParameters(pars);
    }
    
    ////Run the tomography
    void Run(Eigen::MatrixXd & trainData,std::vector<std::vector<std::string> >& trainBases){
        //opt_.Reset();
        int index;
        int counter = 0;
        int trainSize = trainData.rows();
        int saveFrequency =  int(trainSize / bs_);
        Eigen::MatrixXd batch_samples;
        std::vector<std::vector<std::string> > batch_bases;
        std::uniform_int_distribution<int> distribution(0,trainSize-1);
        
        int epoch = 0;
        for(int i=0;i<epochs_;i++){

            // Initialize the visible layer to random data samples
            batch_samples.resize(NNstate_.Nchains(),N_);
            for(int k=0;k<NNstate_.Nchains();k++){
                index = distribution(rgen_);
                batch_samples.row(k) = trainData.row(index);
            }
            NNstate_.SetVisibleLayer(batch_samples);
            
            // Build the batch of data
            batch_samples.resize(bs_,N_); 
            batch_bases.resize(bs_,std::vector<std::string>(N_));

            for(int k=0;k<bs_;k++){
                index = distribution(rgen_);
                batch_samples.row(k) = trainData.row(index);
                batch_bases[k] = trainBases[index];
            }
            
            // Perform one step of optimization
            Gradient(batch_samples,batch_bases);
            UpdateParameters();
            //Compute stuff and print
            if (counter == saveFrequency){
                epoch += 1;
                ExactPartitionFunction();
                Overlap();
                std::cout<<"Epoch "<<epoch<<"\t";
                std::cout<<"Overlap = "<<std::setprecision(8)<<overlap_;
                std::cout<<std::endl;
                counter = 0;
            }
            counter++;
        }
    }
    
    // Compute the partition function by exact enumeration 
    void ExactPartitionFunction() {
        Z_ =0.0;
        for(int i=0;i<basis_states_.rows();i++){
            Z_ += norm(NNstate_.psi(basis_states_.row(i)));
        }
    }

    // Compute the overlap with the target wavefunction
    void Overlap(){
        overlap_ = 0.0;
        std::complex<double> tmp;
        for(int i=0;i<basis_states_.rows();i++){
            tmp += conj(wf_(i))*NNstate_.psi(basis_states_.row(i))/std::sqrt(Z_);
        }
        overlap_ = abs(tmp);
    }
 
    //Set the value of the target wavefunction
    void setWavefunction(Eigen::VectorXcd & psi){
        wf_.resize(1<<N_);
        for(int i=0;i<1<<N_;i++){
            wf_(i) = psi(i);
        }
    }

    //Set the value of the target wavefunction
    void setBasisRotations(std::map<std::string,Eigen::MatrixXcd> & U){
        U_ = U;
    }

    void getRotatedGradient(const std::vector<std::string> & basis,const Eigen::VectorXd & state,Eigen::VectorXcd &gradR){//VectorRbmT & gradR){
        int t=0,counter=0;
        std::complex<double> U=1.0,den=0.0;
        std::bitset<16> bit;
        std::bitset<16> st;
        std::vector<int> basisIndex;
        Eigen::VectorXd v(N_);
        Eigen::VectorXcd num(npar_);
        num.setZero(); 
        basisIndex.clear();
        
        // Extract the sites where the rotation is non-trivial
        for(int j=0;j<N_;j++){
            if (basis[j]!="Z"){
                t++;
                basisIndex.push_back(j);
            }
        }

        // Loop over the states of the local Hilbert space
        for(int i=0;i<1<<t;i++){
            counter =0;
            bit = i;
            v=state;
            for(int j=0;j<N_;j++){
                if (basis[j] != "Z"){
                    v(j) = bit[counter];
                    counter++;
                }
            }
            U=1.0;
            //Compute the product of the matrix elements of the unitary rotations
            for(int ii=0;ii<t;ii++){
                U = U * U_[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
            }
            num += U*NNstate_.Grad(v)*NNstate_.psi(v); 
            den += U*NNstate_.psi(v);
        }
        gradR = num/den;
    }
};
}

#endif
