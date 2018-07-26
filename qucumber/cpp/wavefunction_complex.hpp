#ifndef QST_WAVEFUNCTIONCOMPLEX_HPP
#define QST_WAVEFUNCTIONCOMPLEX_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <iostream>

namespace qst{

class WavefunctionComplex{

    int N_;            // Number of degrees of freedom (visible units)
    int npar_;         // Number of parameters
    int nparLambda_;   // Number of amplitude parameters
    int nparMu_;       // Number of phase parameters
    Rbm rbmAm_;        // RBM for the amplitude
    Rbm rbmPh_;        // RBM for the phases

    const std::complex<double> I_; // Imaginary unit
    
    //Random number generator 
    std::mt19937 rgen_;
    
public:
    // Constructor 
    WavefunctionComplex(Parameters &par):rbmAm_(par),
                                  rbmPh_(par),
                                  I_(0,1){
        npar_ = rbmAm_.Npar() + rbmPh_.Npar();  // Total number of parameters
        nparLambda_ = rbmAm_.Npar();
        nparMu_ = rbmPh_.Npar();
        N_ = rbmAm_.Nvisible();
        std::random_device rd;
        //rgen_.seed(rd());
        rgen_.seed(13579);
    }
    
    // Private members access functions
    inline int N()const{
        return N_;
    }
    inline int Npar()const{
        return npar_;
    }
    inline int NparLambda()const{
        return nparLambda_;
    }
    inline int NparMu()const{
        return nparMu_;
    }
    inline int Nchains(){
        return rbmAm_.Nchains();
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return rbmAm_.VisibleStateRow(s);
    }

    // Set the state of the wavefunction's degrees of freedom
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        rbmAm_.SetVisibleLayer(v);
    }
 
    // Initialize the wavefunction parameters    
    void InitRandomPars(int seed,double sigma){
        rbmAm_.InitRandomPars(seed,sigma);
        rbmPh_.InitRandomPars(seed,sigma);
    }
    
    // Amplitude
    double amplitude(const Eigen::VectorXd & v){
        return std::sqrt(rbmAm_.prob(v));//exp(0.5*rbmAm_.LogVal(v));         
    }
    // Phase
    double phase(const Eigen::VectorXd & v){
        return std::log(rbmPh_.prob(v));
    }
    // Psi
    std::complex<double> psi(const Eigen::VectorXd & v){
        return amplitude(v)*exp(0.5*I_*phase(v));
    }

    //---- SAMPLING ----/
    
    // Perform k steps of Gibbs sampling
    void Sample(int steps){
        rbmAm_.Sample(steps);
    }
   
    //---- DERIVATIVES ----//
    
    //Compute gradient of effective energy wrt Lambda 
    Eigen::VectorXd LambdaGrad(const Eigen::VectorXd & v){
        return rbmAm_.VisEnergyGrad(v);
    }
    //Compute gradient of effective energy wrt all parameters
    Eigen::VectorXd Grad(const Eigen::VectorXd & v){
        Eigen::VectorXd der(npar_);
        der<<rbmAm_.VisEnergyGrad(v),rbmPh_.VisEnergyGrad(v);
        return der;
    }
    //Compute the gradient of the effective energy in an arbitrary basis given by U
    void rotatedGrad(const std::vector<std::string> & basis,
                            const Eigen::VectorXd & state,
                            std::map<std::string,Eigen::MatrixXcd> & Unitaries,
                            Eigen::VectorXd &rotated_gradient ){
        int t=0,counter=0;
        std::complex<double> U=1.0;
        std::complex<double> rotated_psi = 0.0;
        std::bitset<16> bit;
        std::bitset<16> st;
        std::vector<int> basisIndex;
        Eigen::VectorXd v(N_);
        Eigen::VectorXcd rotated_psiGrad(npar_);
        rotated_psiGrad.setZero(); 
        basisIndex.clear();
        // Extract the sites where the rotation is non-trivial
        for(int j=0;j<N_;j++){
            //std::cout<<basis[j]<<std::endl;
            if (basis[j]!="Z"){
                t++;
                basisIndex.push_back(j);
            }
        }

        // Loop over the states of the local Hilbert space
        for(int i=0;i<1<<t;i++){
            counter =0;
            bit = i;
            v=state;
            for(int j=0;j<N_;j++){
                if (basis[j] != "Z"){
                    v(j) = bit[counter];
                    counter++;
                }
            }
            U=1.0;
            //Compute the product of the matrix elements of the unitary rotations
            for(int ii=0;ii<t;ii++){
                //std::cout<<basis[basisIndex[ii]]<<std::endl;
                U = U * Unitaries[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
            }
            //std::cout<<U<<std::endl; 
            rotated_psiGrad += U*Grad(v)*psi(v);
            rotated_psi += U*psi(v);
            //int p=0;
            //for(int i=0;i<2;i++){
            //    for(int j=0;j<2;j++){
            //        std::cout<< rotated_psiGrad(p) << " ";
            //        p++;
            //    }
            //    std::cout<<std::endl;
            //}
        }
        ////std::cout<<std::endl;
        int p=0;
        //for(int i=0;i<2;i++){
        //    for(int j=0;j<2;j++){
        //        std::cout<< Grad(v)(p) << " ";
        //        p++;
        //    }
        //    std::cout<<std::endl;

        //}
        //std::cout<<std::endl;


        rotated_gradient.head(nparLambda_)=(rotated_psiGrad.head(nparLambda_)/rotated_psi).real();
        rotated_gradient.tail(nparMu_)=-(rotated_psiGrad.tail(nparMu_)/rotated_psi).imag();
        p=0;
        //for(int i=0;i<2;i++){
        //    for(int j=0;j<2;j++){
        //        std::cout<< rotated_gradient(p) << " ";
        //        p++;
        //    }
        //    std::cout<<std::endl;
        //}
        ////std::cout<<rotated_gradient<<std::endl<<std::endl<<std::endl; 
    }

    
    //---- UTILITIES ----//

    //Get RBM parameters
    Eigen::VectorXd GetParameters(){
        Eigen::VectorXd pars(npar_);
        pars<<rbmAm_.GetParameters(),rbmPh_.GetParameters();
        return pars;
    }

    // Set RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        Eigen::VectorXd parsAm(nparLambda_);
        Eigen::VectorXd parsPh(npar_-nparLambda_);

        for(int i=0;i<nparLambda_;i++){
            parsAm(i)=pars(i);
        }
        for(int i=0;i<npar_-nparLambda_;i++){
            parsPh(i)=pars(nparLambda_+i);
        }
        rbmAm_.SetParameters(parsAm);
        rbmPh_.SetParameters(parsPh);
    }

    void LoadWeights(std::string &fileName){
        std::ifstream fin(fileName);
        rbmAm_.LoadWeights(fin);
        rbmPh_.LoadWeights(fin);
    }
    void PrintParameters(){
        rbmAm_.PrintParameters();
        rbmPh_.PrintParameters();
    }
};
}

#endif
