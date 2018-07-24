#ifndef __DMRG_H_
#define __DMRG_H_

#include <string>
#include <iostream>

using namespace itensor;

class DMRG {
    
    int N_;
    double h_;
    MPO H_;
    SiteSet sites_;
    std::mt19937 rgen_;
    
public:
    
    MPS psi_;
    double gs_energy_;

    DMRG(int N,double h):N_(N),h_(h) {} 
    
    inline MPS GetPsi() {
        return psi_;
    }
    inline SiteSet GetSites() {
        return sites_;
    }
    
    void BuildHamiltonian(){
        sites_ = SpinHalf(N_);
        auto ampo = AutoMPO(sites_);
        for(int j = 1; j < N_; ++j) {
            ampo += -4.0,"Sz",j,"Sz",j+1;
            ampo += -2.0*h_,"Sx",j;
        }
        ampo += -2.0*h_,"Sx",N_;
        ampo += -4.0,"Sz",N_,"Sz",1;
        H_ = MPO(ampo);
    }

    void InitializeMPS(){
        auto state = InitState(sites_);
        std::uniform_int_distribution<int> distribution(0,1);
        for(int i = 1; i <= N_; ++i) {
            auto ran  = distribution(rgen_);
            if (ran == 1)
                state.set(i,"Up");
            else
                state.set(i,"Dn");
        }
        psi_ = MPS(state);
    }

    void run_dmrg(){
        auto sweeps = Sweeps(25);
        sweeps.maxm() = 10,20,100,100,200,200,200,200,400;
        sweeps.cutoff() = 1E-10;
        sweeps.niter() = 2;
        sweeps.noise() = 1E-7,1E-8,0.0;
        //println(sweeps);
    
        // Begin the DMRG calculation
        auto energy = dmrg(psi_,H_,sweeps,"Quiet");
        gs_energy_ = energy; 
        printfln("\nGround State Energy = %.10f",energy/float(N_));
    }
};

#endif 
