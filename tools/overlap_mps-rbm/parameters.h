#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <stdio.h>
#include <stdlib.h>


// Parameter Class
class Parameters{

public:
   
    int nh_;     // Number of hidden units 
    int nv_;     // Number of visible units
    int cd_;     // Order of contrastive divergence
    int nc_;     // Number of chains
    int Nmc_;     // Number of Monte Carlo samples for the RBM
    int Nmps_;   // Number of mps samples
    int h_;     // Magnetic field

    // Constructor
    Parameters() {
        // Default values
        nv_ = 10;
        nh_ = 10;
        cd_ = 50;
        nc_ = 100;
        Nmc_ = 10000;
        Nmps_= 10000;
        h_ = 1.0;
    }
    
    // Read parameters from the command line
    void ReadParameters(int argc,char** argv){
        std::string flag;
        
        flag = "-nv";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) nv_=atoi(argv[i+1]);
        }
        flag = "-nh";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) nh_=atoi(argv[i+1]);
        }
        flag = "-h";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) h_=double(atof(argv[i+1]));
        }
        flag = "-nc";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) nc_=atoi(argv[i+1]);
        }
        flag = "-cd";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) cd_=atoi(argv[i+1]);
        }
        flag = "-Nmc";
        for(int i=1;i<argc;i++){
           if(flag==argv[i]) Nmc_=atoi(argv[i+1]);
        }
        Nmps_ = Nmc_;
    }
    
    // Print the parameters
    void PrintParameters(){
        std::cout << "\nNeural-Network Quantum State Tomography\n\n";
        std::cout << " Number of chains: " << nc_<< std::endl;
        std::cout << " Steps of block Gibbs sampling: " << cd_<< std::endl;
        std::cout << " Number of samples: " << Nmc_<< std::endl;
        std::cout << std::endl<<std::endl;
    }
};
#endif
