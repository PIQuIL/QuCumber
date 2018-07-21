#ifndef QST_PARAMETERS_HPP
#define QST_PARAMETERS_HPP
#include <stdio.h>
#include <stdlib.h>

namespace qst{

// Parameter Class
class Parameters{

public:
   
    int nh_;            // Number of hidden units 
    int nv_;            // Number of visible units
    double w_;          // Width of normal distribution for initial parameters
    int cd_;            // Number of sampling steps in contrastive divergence
    int nc_;            // Number of sampling chains
    double lr_;         // Learning rate
    double l2_;         // L2 normalization
    int bs_;            // Batch size
    int ep_;            // Training epochs 
    int ns_;            // Number of training samples
    int nb_;            // Number of bases
    std::string basis_; // Basis code

    // Constructor
    Parameters() {
        // Default values
        nv_ = 10;
        nh_ = 10;
        w_  = 0.01;
        cd_ = 10;
        nc_ = 100;
        lr_ = 0.01;
        l2_ = 0.0001;
        bs_ = 100;
        ep_ = 1000;
        ns_ = 10000;
        nb_ = 1;
        basis_ = "std";
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
        flag = "-w";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) w_=double(atof(argv[i+1]));
        }
        flag = "-nc";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) nc_=atoi(argv[i+1]);
        }
        flag = "-cd";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) cd_=atoi(argv[i+1]);
        }
        flag = "-lr";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) lr_=double(atof(argv[i+1]));
        }
        flag = "-l2";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) l2_=double(atof(argv[i+1]));
        }
        flag = "-bs";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) bs_=atoi(argv[i+1]);
        }
        flag = "-ns";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) ns_=atoi(argv[i+1]);
        }
        flag = "-ep";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) ep_=atoi(argv[i+1]);
        }
        flag = "-basis";
        for(int i=1;i<argc;i++){
            if(flag==argv[i]) basis_=argv[i+1];
        }
        nb_ = 2*nv_+1;
    }
    
    // Print the parameters
    void PrintParameters(){
        std::cout << "\nNeural-Network Quantum State Tomography\n\n";
        std::cout << " Number of visible units: " << nv_ << std::endl;
        std::cout << " Number of hidden units: " << nh_<< std::endl;
        std::cout << " Initial distribution width: " << w_<< std::endl;
        std::cout << " Number of chains: " << nc_<< std::endl;
        std::cout << " Steps of contrastive divergence: " << cd_<< std::endl;
        std::cout << " Learning rate: " << lr_<< std::endl;
        std::cout << " L2 regularization: " << l2_<< std::endl;
        std::cout << " Batch size: " << bs_<< std::endl;
        std::cout << " Number of training samples: " << ns_<< std::endl;
        std::cout << " Number of training iterations " << ep_<< std::endl;
        std::cout << std::endl<<std::endl;
    }
};
}
#endif
