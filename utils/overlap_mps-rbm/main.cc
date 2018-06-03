#include "itensor/all.h"
#include <vector>
#include <fstream>
#include <random>
#include <string>
#include <iomanip>
#include "mps_sampler.h"
#include "parameters.h"
#include "rbm.h"
#include "dmrg.h"
#include  <boost/format.hpp>
#include <ctime>

using namespace itensor;
    
int main(int argc, char* argv[]) {

    Parameters par;
    par.ReadParameters(argc,argv);
    par.PrintParameters(); 
    
    // DEFINE THE RBM STATE
    Rbm rbm(par);
    std::string rbm_parameters_filename;
    rbm_parameters_filename = "weights.txt";
    //rbm_parameters_filename = "PATH_TO_FILENAME/FILENAME";
    rbm.LoadWeights(rbm_parameters_filename);
   
    // DMRG
    DMRG solver(par.nv_,par.h_); 
    solver.BuildHamiltonian();
    solver.InitializeMPS();
    solver.run_dmrg();
    MPS psi_mps = solver.GetPsi();
    SiteSet sites = solver.GetSites();
    
    // SAMPLER
    MPSsampler sampler(par.nv_,psi_mps);
    sampler.get_partial_tensors();
    //sampler.test();

    // DEFINE THE MPS STATE
    //SpinHalf sites;
    //MPS psi_mps;
    //std::string mps_sites_name,mps_tensor_name;
    //mps_sites_name  = "PATH_TO_MPS/SITES";
    //mps_tensor_name = "PATH_TO_MPS/TENSOR"; 
    //readFromFile(mps_sites_name,sites);
    //readFromFile(mps_tensor_name,psi_mps);
    
    // COMPUTE THE OVERLAP
    double O,F;     // Overlap and fidelity=O^2
    
    // SAMPLE THE RBM
    double O1 = 0.0;
    int steps = int(par.Nmc_/par.nc_); 
    for(int i=1; i<steps+1; i++){
        rbm.Sample(par.cd_);
        for(int k=0;k<par.nc_;k++){
            O1 += sampler.collapse_psi(rbm.VisibleStateRow(k))/sqrt(rbm.p(rbm.VisibleStateRow(k)))/double(par.Nmc_);
        }

    }
    
    // SAMPLE THE MPS
    double O2 = 0.0;
    steps = int(par.Nmps_); 
    for(int i=1; i<par.Nmps_+1; i++){
        sampler.sample();
        O2 += sqrt(rbm.p(sampler.state_))/sampler.collapse_psi(sampler.state_)/double(par.Nmps_);

    }
    F = O1 * O2;
    O = sqrt(F);
    std::cout << std::endl;   
    std::cout << "True overlap = 0.98603424" << std::endl <<std::endl;
    std::cout << "Overlap from sampling= " << O <<std::endl; //" +- " << O_err << endl << endl;
    std::cout << std::endl; 
    //std::string outputName = "overlap_rbm-mps_tfim1d_N10_cd";
    //outputName += boost::str(boost::format("%d") % par.cd_);
    //outputName += "_n" + boost::str(boost::format("%d") % par.Nmc_);
    //outputName += ".txt";
    //std::ofstream fout(outputName);
    //fout << O << std::endl;
    //fout.close(); 

    //// TEST
    ////Compute exact overlap
    //double O_exact = 0.0;
    //std::bitset<10> bit;
    //Eigen::VectorXd conf;
    //conf.setZero(par.nv_);
    //double Z = 0.0;
    //for(int i=0; i<1<<par.nv_;i++){
    //    bit  = i;
    //    for (int j=0;j<par.nv_;j++){
    //        conf(j)=bit[par.nv_-1-j];
    //    }
    //    O_exact += sqrt(rbm.p(conf))*sampler.collapse_psi(conf);
    //    Z += rbm.p(conf);
    //}
    //O_exact /= sqrt(Z);
    //std::cout << "Exact Overlap = " << O_exact <<std::endl; //" +- " << O_err << endl << endl;
} 
