#ifndef QST_OBSERVERWAVEFUNCTION_HPP
#define QST_OBSERVERWAVEFUNCTION_HPP

#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>

namespace qst{

class ObserverPSI{

    Wavefunction &PSI_;

    int N_;
    int npar_;
    Eigen::VectorXcd target_psi_;
    std::vector<Eigen::VectorXcd> rotated_wf_;
    Eigen::MatrixXd basis_states_;      // Hilbert space basis
    std::map<std::string,Eigen::MatrixXcd> U_;
    std::vector<std::vector<std::string> > basisSet_;

public:

    double KL_;
    double overlap_;
    double Z_;

    ObserverPSI(Wavefunction &PSI):PSI_(PSI){ 
        
        std::cout<<"- Initializing observer module"<<std::endl;
        N_ = PSI_.N();
        npar_ = PSI_.Npar();

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
 
    //Compute different estimators for the training performance
    void Scan(int i){//,Eigen::MatrixXd &nll_test,std::ofstream &obs_out){
        ExactPartitionFunction();
        ExactKL(); 
        Overlap();
        PrintStats(i);
    }

    //Compute the partition function by exact enumeration 
    void ExactPartitionFunction() {
        Z_ = 0.0;
        for(int i=0;i<basis_states_.rows();i++){
            Z_ += norm(PSI_.psi(basis_states_.row(i)));
        }
    }

    // Compute the overlap with the target wavefunction
    void Overlap(){
        overlap_ = 0.0;
        std::complex<double> tmp;
        for(int i=0;i<basis_states_.rows();i++){
            tmp += conj(target_psi_(i))*PSI_.psi(basis_states_.row(i))/std::sqrt(Z_);
        }
        overlap_ = abs(tmp);
    }
    
    //Compute KL divergence exactly
    void ExactKL(){
        Eigen::VectorXcd rotated_psi(1<<N_);
        //KL in the standard basis
        KL_ = 0.0;
        for(int i=0;i<1<<N_;i++){
            if (norm(target_psi_(i))>0.0){
                KL_ += norm(target_psi_(i))*log(norm(target_psi_(i)));
            }
            KL_ -= norm(target_psi_(i))*log(norm(PSI_.psi(basis_states_.row(i))));
            KL_ += norm(target_psi_(i))*log(Z_);
        }
        //KL in the rotated bases
        for (int b=1;b<basisSet_.size();b++){
            rotateRbmWF(basisSet_[b],rotated_psi);
            for(int i=0;i<1<<N_;i++){
                if (norm(rotated_wf_[b-1](i))>0.0){
                    KL_ += norm(rotated_wf_[b-1](i))*log(norm(rotated_wf_[b-1](i)));
                }
                KL_ -= norm(rotated_wf_[b-1](i))*log(norm(rotated_psi(i)));
                KL_ += norm(rotated_wf_[b-1](i))*log(Z_);
            }
        }
    }
    
    //Print observer
    void PrintStats(int i){
        std::cout << "Epoch: " << i << "\t";     
        std::cout << "KL = " << std::setprecision(10) << KL_ << "\t";
        std::cout << "Overlap = " << std::setprecision(10) << overlap_<< "\t";//<< Fcheck_;
        std::cout << std::endl;
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
    //Set the value of the target wavefunction
    void setBasisRotations(std::map<std::string,Eigen::MatrixXcd> & U){
        U_ = U;
    }
    void setBasis(std::vector<std::vector<std::string> > basis) {
        basisSet_ = basis;
    }

    void rotateRbmWF(const std::vector<std::string> & basis, Eigen::VectorXcd &psiR){//VectorRbmT & psiR){
        int t,counter;
        std::complex<double> U,Upsi;
        std::bitset<16> bit;
        std::bitset<16> st;
        std::bitset<16> tmp;
        std::vector<int> basisIndex;
        Eigen::VectorXd state(N_);
        Eigen::VectorXd v(N_);
    
        for(int x=0;x<1<<N_;x++){
            U = 1.0;
            Upsi=0.0;
            basisIndex.clear();
            t = 0;
            st = x;
            for (int j=0;j<N_;j++){
                state(j) = st[N_-1-j];
            }
            for(int j=0;j<N_;j++){
                if (basis[j]!="Z"){
                    t++;
                    basisIndex.push_back(j);
                }
            }
            for(int i=0;i<1<<t;i++){
                counter  =0;
                bit = i;
                v=state;
                for(int j=0;j<N_;j++){
                    if (basis[j] != "Z"){
                        v(j) = bit[counter];
                        counter++;
                    }
                }
                U=1.0;
                for(int ii=0;ii<t;ii++){
                    U = U * U_[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
                }
                for(int j=0;j<N_;j++){
                    tmp[j]=v(N_-1-j);
                }
                Upsi += U*PSI_.psi(v);
            }
            psiR(x) = Upsi;
        }
    }
};
}

#endif
