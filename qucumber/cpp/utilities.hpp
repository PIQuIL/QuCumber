#ifndef QST_UTILITIES_HPP
#define QST_UTILITIES_HPP
#include <Eigen/Core>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <boost/format.hpp>
#include <fstream>
#include "parameters.hpp"

namespace qst{

void LoadTrainingData(std::string &baseName, Parameters &par,Eigen::MatrixXd & trainSamples,std::vector<std::vector<std::string> >& trainBases) 
{
    int trainSize = par.nb_ * par.ns_;
    std::string fileName;

    trainSamples.resize(trainSize,par.nv_);
    trainBases.resize(trainSize,std::vector<std::string>(par.nv_));

    fileName = baseName + "train_samples.txt";
    std::ifstream fin_samples(fileName);
    
    for (int n=0; n<trainSize; n++) {
        for (int j=0; j<par.nv_; j++) {
            fin_samples>> trainSamples(n,j);
        }
    }
    if (par.basis_.compare("std")!=0){
        fileName = baseName + "train_bases.txt";
        std::ifstream fin_bases(fileName);
        
        for (int n=0; n<trainSize; n++) {
            for (int j=0; j<par.nv_; j++) {
                fin_samples>> trainSamples(n,j);
                fin_bases>> trainBases[n][j];
            }
        }
    }
    else {
        for (int n=0; n<trainSize; n++) {
            for (int j=0; j<par.nv_; j++) {
                trainBases[n][j] = "Z";
            }
        }
    }
}

void LoadWavefunction(Parameters & par,std::string &wf_fileName,Eigen::VectorXd & wf){
    std::ifstream fin(wf_fileName);
    wf.resize(1<<par.nv_);
    for(int i=0;i<1<<par.nv_;i++){
        fin >> wf(i);
    }
}

void LoadWavefunction(Parameters & par,std::string &wf_fileName,Eigen::VectorXcd & wf,std::vector<Eigen::VectorXcd> & rotated_wf){

    std::ifstream fin(wf_fileName);
    wf.resize(1<<par.nv_);
    double x_in;
   
    if (par.basis_.compare("std")!=0){
        for(int i=0;i<1<<par.nv_;i++){
            fin >> x_in;
            wf.real()(i)=x_in;
            fin >> x_in;
            wf.imag()(i)=x_in;
        }
        Eigen::VectorXcd tmp(1<<par.nv_);
        for(int b=1;b<par.nb_;b++){
            for(int i=0;i<1<<par.nv_;i++){
                fin >> x_in;
                tmp.real()(i)=x_in;
                fin >> x_in;
                tmp.imag()(i)=x_in;
            }
            rotated_wf.push_back(tmp);
        }
    }
    else {
        for(int i=0;i<1<<par.nv_;i++){
            fin >> x_in;
            wf.real()(i)=x_in;
            wf.imag()(i)=0.0;
        }


    }
}
void LoadBasesConfigurations(Parameters &par,std::string &basis_name,std::vector<std::vector<std::string> > &basis) {
    
    std::ifstream fin(basis_name);
    basis.resize(par.nb_,std::vector<std::string>(par.nv_));
    for (int b=0;b<par.nb_;b++){
        for(int j=0;j<par.nv_;j++){
            fin >> basis[b][j];
        }
    }
}

void SetNumberOfBases(Parameters &par){
    if (par.basis_ == "std")    par.nb_ = par.nv_;
    if (par.basis_ == "xy1")    par.nb_ = 2*par.nv_+1;
}
void GenerateUnitaryRotations(std::map<std::string,Eigen::MatrixXcd> & U){

    double oneDivSqrt2 = 1.0/sqrt(2.0);
    
    //X rotation
    U["X"].setZero(2,2);
    U["X"].real()<< oneDivSqrt2,oneDivSqrt2,
                    oneDivSqrt2,-oneDivSqrt2;
    //Y rotation
    U["Y"].resize(2,2);
    U["Y"].real()<< oneDivSqrt2,0.0,
                    oneDivSqrt2,0.0;
    U["Y"].imag()<< 0.0,-oneDivSqrt2,
                    0.0,oneDivSqrt2;

}

}

#endif
