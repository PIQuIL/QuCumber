#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>
#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

//Quantum State Tomography class
class Tomography {

    Rbm & rbm_;                         // Rbm

    int npar_;                          // Number of variational parameters
    int bs_;                            // Batch size
    int cd_;                            // Number of Gibbs stepos in contrastive divergence
    int epochs_;                        // Number of training iterations
    double lr_;                         // Learning rate
    double l2_;                         // L2 regularization constant

    Eigen::VectorXd grad_;              // Gradient
    
    std::mt19937 rgen_;                 // Random number generator
    
    double negative_log_likelihood_;    // Negative log-likelihood
    double overlap_;                    // Overlap
    double Z_;                          // Partition function
    Eigen::MatrixXd basis_states_;      // Hilbert space basis

public:
    // Contructor 
    Tomography(Rbm &rbm,Parameters &par): rbm_(rbm){
        npar_=rbm_.Npar();
        bs_ = par.bs_;
        cd_ = par.cd_;
        lr_ = par.lr_;
        l2_ = par.l2_;
        epochs_ = par.ep_;
        grad_.resize(npar_);
        basis_states_.resize(1<<rbm_.Nvisible(),rbm_.Nvisible());
        std::bitset<10> bit;
        for(int i=0;i<1<<rbm_.Nvisible();i++){
            bit = i;
            for(int j=0;j<rbm_.Nvisible();j++){
                basis_states_(i,j) = bit[rbm_.Nvisible()-j-1];
            }
        }
    }

    // Compute gradients of negative Log-Likelihood on a batch 
    void Gradient(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        
        //Positive Phase driven by the data
        for(int s=0;s<bs_;s++){
            grad_ -= rbm_.DerLog(batch.row(s))/double(bs_);
        }
        //Negative Phase driven by the model
        rbm_.Sample(cd_);
        for(int s=0;s<rbm_.Nchains();s++){
            grad_ += rbm_.DerLog(rbm_.VisibleStateRow(s))/double(rbm_.Nchains());
        }
    }

    // Update rbm parameters
    void UpdateParameters(){
        auto pars=rbm_.GetParameters();
        for(int i=0;i<npar_;i++){
            pars(i) -= lr_*(grad_(i)+l2_*pars(i));
        }
        rbm_.SetParameters(pars);
    }

    // Run the tomography
    void Run(Eigen::MatrixXd &dataSet,Eigen::VectorXd &target_psi){
        
        // Initialization
        int index;
        int saveFrequency = 1000;
        int counter = 0;
        int trainSize = int(0.9*dataSet.rows());
        int validSize = int(0.1*dataSet.rows());
        Eigen::MatrixXd trainSet(1<<rbm_.Nvisible(),trainSize);
        Eigen::MatrixXd validSet(1<<rbm_.Nvisible(),validSize);
        Eigen::MatrixXd batch_bin;
        std::uniform_int_distribution<int> distribution(0,trainSize-1);
        
        trainSet = dataSet.topRows(trainSize);
        validSet = dataSet.bottomRows(validSize);

        // Training
        for(int i=0;i<epochs_;i++){

            // Initialize the visible layer to random data samples
            batch_bin.resize(rbm_.Nchains(),rbm_.Nvisible());
            for(int k=0;k<rbm_.Nchains();k++){
                index = distribution(rgen_);
                batch_bin.row(k) = trainSet.row(index);
            }
            rbm_.SetVisibleLayer(batch_bin);
            
            // Build the batch of data
            batch_bin.resize(bs_,rbm_.Nvisible()); 
            for(int k=0;k<bs_;k++){
                index = distribution(rgen_);
                batch_bin.row(k) = trainSet.row(index);
            }
            
            // Perform one step of optimization
            Gradient(batch_bin);
            UpdateParameters();
            
            //Compute stuff and print
            if (counter == saveFrequency){
                ExactPartitionFunction();
                Overlap(target_psi);
                NLL(validSet);
                std::cout<<"Epoch "<<i<<"\t";
                std::cout<<"NLL = "<<std::setprecision(8)<<negative_log_likelihood_<<"\t    ";
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
            Z_ += rbm_.p(basis_states_.row(i));
        }
    }
 
    // Compute the overlap with the target wavefunction
    void Overlap(Eigen::VectorXd &target_psi){
        overlap_ = 0.0;
        for(int i=0;i<basis_states_.rows();i++){
            overlap_ += target_psi(i)*std::sqrt(rbm_.p(basis_states_.row(i)))/std::sqrt(Z_);
        }
    }

    // Compute the average negative log-likelihood
    void NLL(Eigen::MatrixXd & samples) {
        negative_log_likelihood_=0.0;
        for (int k=0;k<samples.rows();k++){
            negative_log_likelihood_ += std::log(Z_) - std::log(rbm_.p(samples.row(k)));
        }
        negative_log_likelihood_ /= double(samples.rows());
    }
 
    // Derivatives check
    void DerKLTest(Eigen::MatrixXd &trainSet,Eigen::VectorXd &target_psi){
        double eps = 0.0001;
        auto pars = rbm_.GetParameters();
        double KL;
      
        std::cout<<"Running the derivative check..."<<std::endl<<std::endl;
        // Compute the algorithmic derivatives
        ExactPartitionFunction();
        Eigen::VectorXd ders(npar_);
        ders.setZero(npar_);
        for(int j=0;j<1<<rbm_.Nvisible();j++){
            ders -= target_psi(j)*target_psi(j)*rbm_.DerLog(basis_states_.row(j));
            ders += rbm_.DerLog(basis_states_.row(j))*rbm_.p(basis_states_.row(j))/Z_;
        }
           
        // Compute the numerical derivatives
        for(int p=0;p<npar_;p++){
            pars(p)+=eps;
            rbm_.SetParameters(pars);
            double valp=0.0;
            KL=0.0;
            ExactPartitionFunction();
            for(int j=0;j<1<<rbm_.Nvisible();j++){
                KL += target_psi(j)*target_psi(j)*2*log(target_psi(j));
                KL += target_psi(j)*target_psi(j)*log(Z_);
                KL -= target_psi(j)*target_psi(j)*log(rbm_.p(basis_states_.row(j)));
            }
            valp = KL;
            pars(p)-=2*eps;
            rbm_.SetParameters(pars);
            double valm=0.0;
            KL=0.0;
            ExactPartitionFunction();
            for(int j=0;j<1<<rbm_.Nvisible();j++){
                KL += target_psi(j)*target_psi(j)*2*log(target_psi(j));
                KL +=target_psi(j)*target_psi(j)*log(Z_);
                KL -=target_psi(j)*target_psi(j)*log(rbm_.p(basis_states_.row(j)));
            }
            valm = KL;
            pars(p)+=eps;
            double numder=(-valm+valp)/(eps*2);
            std::cout<<"Derivative wrt par "<<p<<" :  Algorithmic gradient = "<<ders(p)<<"  \tNumerical gradient = "<<numder<<std::endl;
        }
        std::cout << std::endl;
    }
};
}
#endif
