#ifndef __MPS_SAMPLER_H_
#define __MPS_SAMPLER_H_

#include <string>
#include <iostream>
#include <vector>
#include <iomanip>
#include <Eigen/Core>

using namespace itensor;

class MPSsampler {
    
    int N_;                 // Number of spins
    MPS psi_;               // mps state
    std::mt19937 rgen_;     // Random number generator
public:
    
    Eigen::VectorXd state_;
    std::vector<ITensor> partial_tensors_;
    std::vector<ITensor> trail_tensors_;
    //Default constructor
    MPSsampler(int N,MPS & psi):N_(N),psi_(psi) {
        state_.setZero(N_);
    } 
    
    // Store partial contraction of the mps
    void get_partial_tensors(){

        ITensor pt; //partial tensor tmp variable
        pt = psi_.A(1);
        pt *= dag(prime(psi_.A(1),Link));
        partial_tensors_.push_back(pt);
        for (int j=2; j<N_; j++){
            pt *= psi_.A(j);
            pt *= dag(prime(psi_.A(j),Link));
            partial_tensors_.push_back(pt);
        }
    }

    void sample(){
        
        double prob;
        double joint_prob= 1.0;
        std::uniform_real_distribution<double> distr(0,1);

        trail_tensors_.clear();
        ITensor rho;
        ITensor trail,trail_dag;
        ITensor tmp;
        
        // Sample spin at site N
        rho = partial_tensors_[N_-2];
        rho *= psi_.A(N_);
        tmp = dag(prime(psi_.A(N_),Site,Link));
        rho *= tmp;
        prob = rho.real(2,2);
        state_(N_-1) = distr(rgen_) < prob;
        if (state_(N_-1) == 1){
            joint_prob *= prob;
        }
        else {
            joint_prob *=(1.0-prob);
        }
        
        // Sample spin at site N-1
        rho = partial_tensors_[N_-3];
        rho = rho * psi_.A(N_-1);
        tmp = dag(prime(psi_.A(N_-1),Site,Link));
        rho = rho * tmp;
        
        trail = psi_.A(N_);
        trail *=s_tensor(int(state_(N_-1)),psi_.A(N_));
        trail *= dag(prime(psi_.A(N_),Site,Link));
        trail *= prime(s_tensor(int(state_(N_-1)),psi_.A(N_)),Site);
        rho *= trail;
        rho = rho / joint_prob;

        prob = rho.real(2,2);
        state_(N_-2) = distr(rgen_) < prob;
        if (state_(N_-2) == 1){
            joint_prob *= prob;
        }
        else {
            joint_prob *=(1.0-prob);
        }
        
        // Sample the other spins
        for(int j= N_-2; j>0; j--){
            if(j>1){
                rho = partial_tensors_[j-2];
                rho = rho * psi_.A(j);
            }
            else {
                rho= psi_.A(j);
            }
            tmp = dag(prime(psi_.A(j),Site,Link));
            rho = rho * tmp;
            trail *= psi_.A(j+1);
            trail *= s_tensor(int(state_(j)),psi_.A(j+1));
            trail *= dag(prime(psi_.A(j+1),Site,Link));
            trail *= prime(s_tensor(int(state_(j)),psi_.A(j+1)),Site);
            rho *= trail;
            rho = rho / joint_prob;
            prob = rho.real(2,2);
            state_(j-1) = distr(rgen_) < prob;
            if (state_(j-1) == 1){
                joint_prob *= prob;
            }
            else {
                joint_prob *=(1.0-prob);
            }
        }
    }

    double collapse_psi(const Eigen::VectorXd & sigma){
        ITensor psi_sigma;
        ITensor s;
        s = s_tensor(sigma[0],psi_.A(1));
        psi_sigma = psi_.A(1);
        psi_sigma *= s;

        for (int j=2;j<=N_;j++){
            s = s_tensor(sigma[j-1],psi_.A(j));
            psi_sigma *= psi_.A(j);
            psi_sigma *= s;
        }
        return psi_sigma.real();
    }
       
   
    ITensor s_tensor(const int & s_value, const ITensor & A){
        auto index = findtype(A,Site);
        ITensor s(index);
        if (s_value == 0){
            s.set(index(1),1.0);
            s.set(index(2),0.0);
        }
        else {
            s.set(index(1),0.0);
            s.set(index(2),1.0);
        }
        return s;
    }

    void test(){
        
        int ns = 10000;
        std::bitset<10> bit;
        Eigen::VectorXd conf;
        std::vector<double> psi;
        std::vector<int> hist;
        std::vector<double> prob_approx;
        int ind;
        hist.assign(1<<N_,0);
        conf.setZero(N_);
        for(int i=0; i<1<<N_;i++){
            bit = i;
            for (int j=0;j<N_;j++){
                conf(j)=bit[N_-1-j];
            }
            psi.push_back(collapse_psi(conf));
        }
        std::cout <<std::endl<<std::endl; 
        for (int i=0; i<ns;i++){
            sample();
            ind = 0;
            for(int j=0;j<N_;j++){
                ind += state_(N_-j-1)*pow(2,j);    
            }
            hist[ind]++;
        }
        
        for(int i=0;i<1<<N_;i++){
            bit = i;
            for (int j=0;j<N_;j++){
                conf(j)=bit[N_-1-j];
                std::cout<<conf(j)<< " ";
            }
            std::cout<< "      Psi = " << psi[i]*psi[i];
            std::cout<< "      Sampled = " <<double(hist[i])/double(ns);
            std::cout<<std::endl;
        }

    }

   
};

#endif 
