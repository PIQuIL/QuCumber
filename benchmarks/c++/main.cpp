#include <iostream>
#include "qst.hpp"

int main(int argc, char* argv[]){

    //---- PARAMETERS ----//
    qst::Parameters par;    //Set default initial parameters

    //Read simulation parameters from command line
    par.ReadParameters(argc,argv);    //Read parameters from the command line
    par.PrintParameters();            //Print parameter on screen
   
    //Load the data
    std::string fileName; 
    std::string baseName = "data/2qubits_";
    Eigen::VectorXcd target_psi(1<<par.nv_);                //Target wavefunction
    std::vector<Eigen::VectorXcd> rotated_target_psi;       //Vector with the target wavefunctions in different basis
    std::vector<std::vector<std::string> > basisSet;        //Set of bases available
    std::map<std::string,Eigen::MatrixXcd> UnitaryRotations;//Container of the of 1-local unitary rotations
    Eigen::MatrixXd training_samples(par.ns_,par.nv_);      //Training samples matrix
    std::vector<std::vector<std::string> > training_bases;  //Training bases matrix

    //Load data
    qst::GenerateUnitaryRotations(UnitaryRotations);        //Generate the unitary rotations
    fileName = baseName + "psi.txt";            
    qst::LoadWavefunction(par,fileName,target_psi,rotated_target_psi);
    fileName = baseName + "bases.txt";
    qst::LoadBasesConfigurations(par,fileName,basisSet);                //Load training samples
    qst::LoadTrainingData(baseName,par,training_samples,training_bases);//Load training bases
    
   
    //---- OPTIMIZER ----//
    typedef qst::Sgd Optimizer; //Stochastic gradient descent
    Optimizer opt(par);         //Construc the optimizer object

    //---- NEURAL NETWORK STATE ----//
    typedef qst::Wavefunction NNState;
    typedef qst::ObserverPSI Observer;

    
    NNState nn(par);
    nn.InitRandomPars(12345,par.w_);
    Observer obs(nn);
    obs.setWavefunction(target_psi);
    obs.setBasisRotations(UnitaryRotations);
    obs.setBasis(basisSet);
    obs.setRotatedWavefunctions(rotated_target_psi);
    
    //---- TOMOGRAPHY ----//
    //qst::Tomography<NNState,Observer,Optimizer> tomo(opt,nn,obs,par);
    //tomo.setBasisRotations(UnitaryRotations);
    //tomo.Run(training_samples,training_bases);
    
    //---- TEST ----// 
    qst::Test<NNState,Observer> test(nn,obs,par);
    test.setWavefunction(target_psi);
    test.setBasisRotations(UnitaryRotations);
    test.setBasis(basisSet);
    test.setRotatedWavefunctions(rotated_target_psi);
    test.DerKL(0.000001);
}
