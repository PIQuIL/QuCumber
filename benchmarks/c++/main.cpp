#include <iostream>
#include "qst.hpp"

int main(int argc, char* argv[]){

    //---- PARAMETERS ----//
    qst::Parameters par;
    //Read simulation parameters from command line
    par.ReadParameters(argc,argv);
    par.PrintParameters();
    
    //---- DATA ----//
    Eigen::MatrixXd training_data(par.ns_,par.nv_);
    std::ifstream data_file("training_data.txt");
    for(int i=0;i<par.ns_;i++){
        for(int j=0;j<par.nv_;j++){
            data_file >> training_data(i,j);
        }
    }
    Eigen::VectorXd target_psi(1<<par.nv_);
    std::ifstream psi_file("target_psi.txt");
    for(int i=0;i<1<<par.nv_;i++){
        psi_file >> target_psi(i);
    }
    
    //---- RBM ----//
    qst::Rbm rbm(par);
    rbm.InitRandomPars(12345,par.w_);

    //---- TOMOGRAPHY ----//
    qst::Tomography tomo(rbm,par);  
    tomo.Run(training_data,target_psi); 
    
    //---- DERIVATIVES TEST ---//
    //tomo.DerKLTest(training_data,target_psi);
}
