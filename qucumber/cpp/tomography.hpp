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
template<class NNState,class Observer,class Optimizer> class Tomography {

    NNState & NNstate_;                         // Neural network representation of the state
    Optimizer & opt_;                           // Optimizer
    Observer &obs_;                             // Observer

    int N_;                                     // Number of physical degrees of freedom
    int npar_;                                  // Number of variational parameters
    int nparLambda_;                            // Number of amplitude variational parameters
    int nparMu_;                                // Number of phase variational parameters
    int bs_;                                    // Batch size
    int cd_;                                    // Number of Gibbs stepos in contrastive divergence
    int epochs_;                                // Number of training iterations
    double lr_;                                 // Learning rate
    double l2_;                                 // L2 regularization constant

    Eigen::VectorXd grad_;                      // Gradients 
    Eigen::VectorXd rotated_grad_;             // Rotated gradients
    std::mt19937 rgen_;                         // Random number generator
    std::map<std::string,Eigen::MatrixXcd> U_;  // Structure containin the single unitary rotations
public:
    
    Tomography(Optimizer & opt,NNState & NNstate,Observer &obs,Parameters &par):opt_(opt),NNstate_(NNstate),obs_(obs),N_(NNstate.N()),bs_(par.bs_),cd_(par.cd_) {
            
        npar_=NNstate_.Npar();
        nparLambda_ = NNstate_.NparLambda();
        nparMu_ = NNstate_.NparMu();
        opt_.SetNpar(npar_);
        bs_ = par.bs_;
        cd_ = par.cd_;
        lr_ = par.lr_;
        l2_ = par.l2_;
        epochs_ = par.ep_;
        grad_.resize(npar_);
        rotated_grad_.resize(npar_);
    }

    //Compute gradient of KL divergence 
    void ComputeGradient(const Eigen::MatrixXd & batchSamples,const std::vector<std::vector<std::string> >& batchBases){ 
        grad_.setZero();

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
                grad_.head(nparLambda_) += NNstate_.LambdaGrad(batchSamples.row(k))/double(bs_);
            }
            else { // Positive phase - Lambda and Mu gradients for non-trivial bases
                NNstate_.rotatedGrad(batchBases[k],batchSamples.row(k),U_,rotated_grad_);
                grad_ += rotated_grad_/double(bs_);
                //grad_.head(nparLambda_) += rotated_grad_.head(nparLambda_).real()/double(bs_);
                //grad_.tail(nparMu_) -= rotated_grad_.tail(nparMu_).imag()/double(bs_);
            }
        }
       
        //Negative Phase - Exact
        obs_.ExactPartitionFunction();
        for(int b=0;b<obs_.basisSet_.size();b++){
            for(int j=0;j<1<<N_;j++){
                grad_.head(nparLambda_) -= norm(obs_.target_psi_(j))*NNstate_.LambdaGrad(obs_.basis_states_.row(j))/obs_.Z_;
            }
        }

        ////Negative Phase - Sampled
        //NNstate_.Sample(cd_);
        //for(int k=0;k<NNstate_.Nchains();k++){
        //    grad_.head(nparLambda_) -= NNstate_.LambdaGrad(NNstate_.VisibleStateRow(k))/double(NNstate_.Nchains());
        //}
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
            // Randomize a batch and set the visible layer to a data point 
            SetUpTrainingStep(trainData,batch_samples,trainBases,batch_bases,distribution); 
            // Perform one step of optimization
            ComputeGradient(batch_samples,batch_bases);
            UpdateParameters();
            //Compute stuff and print
            if (counter == saveFrequency){
                epoch += 1;
                obs_.Scan(epoch);
                //std::cout << "Epoch: " << epoch << std::endl;
                counter = 0;
            }
            counter++;
        }
    }
    
    //Set the value of the target wavefunction
    void setBasisRotations(std::map<std::string,Eigen::MatrixXcd> & U){
        U_ = U;
    }
    // Setup the training batch and visible layer initial configuration
    void SetUpTrainingStep(Eigen::MatrixXd & trainData,
                           Eigen::MatrixXd &batch_samples,
                           std::vector<std::vector<std::string> >& trainBases,
                           std::vector<std::vector<std::string> > &batch_bases,
                           std::uniform_int_distribution<int> & distribution){
            int index;
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
    }
};
}

#endif
