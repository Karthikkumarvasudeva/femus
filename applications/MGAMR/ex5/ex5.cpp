/** \file Ex11.cpp
 *  \brief This example shows how to set and solve the weak form
 *   of the Boussinesq appoximation of the Navier-Stokes Equation
 *
 *  \f{eqnarray*}
 *  && \mathbf{V} \cdot \nabla T - \nabla \cdot\alpha \nabla T = 0 \\
 *  && \mathbf{V} \cdot \nabla \mathbf{V} - \nabla \cdot \nu (\nabla \mathbf{V} +(\nabla \mathbf{V})^T)
 *  +\nabla P = \beta T \mathbf{j} \\
 *  && \nabla \cdot \mathbf{V} = 0
 *  \f}
 *  in a unit box domain (in 2D and 3D) with given temperature 0 and 1 on
 *  the left and right walls, respectively, and insulated walls elsewhere.
 *  \author Eugenio Aulisa
 */

#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "LinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "adept.h"
#include <cstdlib>

#include "../include/LinearImplicitSystemForSSC.hpp"


#include "MyVector.hpp"
#include "MyMatrix.hpp"

using namespace femus;

bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int faceIndex, const double time) {
  bool dirichlet = true; //dirichlet
  value = 0.;
  
  //BEGIN: to use for 3D
  if(faceIndex == 2) dirichlet = false;
  //END
  
  return dirichlet;
}



bool SetRefinementFlag(const std::vector < double >& x, const int& elemgroupnumber, const int& level) {

  bool refine = false;

 //BEGIN: refinement strategy for 2D
//   if(elemgroupnumber == 7 || elemgroupnumber == 9 ){
//     refine = true;
//   }
  //END
  

  //BEGIN: refinement strategy with circle
  double radius = 0.25/(level);
  
  if(x[0]*x[0]+x[1]*x[1] < radius*radius) refine=true;
  //END


  
  return refine;

}



void AssembleTemperature_AD(MultiLevelProblem& ml_prob);    //, unsigned level, const unsigned &levelMax, const bool &assembleMatrix );


int main(int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  //mlMsh.ReadCoarseMesh("./input/cube_hex.neu","seventh",scalingFactor);
  //mlMsh.ReadCoarseMesh("./input/adaptiveRef4.neu", "seventh", scalingFactor);
  
  //BEGIN mesh for 2D tests
//   mlMsh.ReadCoarseMesh("./input/adaptiveRef4Mixed.neu", "seventh", scalingFactor);
  //END 
  
  //BEGIN mesh for 3D tests
  mlMsh.ReadCoarseMesh("./input/cube_mixed_Two_boundary_groups_Four_volume_groups_AMR.neu", "seventh", scalingFactor); ///@todo pick from library!!!
  //END 
    
  //mlMsh.ReadCoarseMesh("./input/triAMR.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh("./input/quadAMR.neu", "seventh", scalingFactor);
  /* "seventh" is the order of accuracy that is used in the gauss integration scheme
     probably in the furure it is not going to be an argument of this function   */
  unsigned dim = mlMsh.GetDimension();

//   unsigned numberOfUniformLevels = 3;
//   unsigned numberOfSelectiveLevels = 0;
//   mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);

  unsigned numberOfUniformLevels = 1;
  unsigned numberOfSelectiveLevels =3;
  mlMsh.RefineMesh(numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels , SetRefinementFlag);

  // erase all the coarse mesh levels
  //mlMsh.EraseCoarseLevels(numberOfUniformLevels - 3);


  MultiLevelSolution mlSol(&mlMsh);

  // add variables to mlSol
  mlSol.AddSolution("T", LAGRANGE, SECOND);//FIRST);;

  mlSol.Initialize("All");

  // attach the boundary condition function and generate boundary data
  mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
  mlSol.GenerateBdc("All");

  // define the multilevel problem attach the mlSol object to it
  MultiLevelProblem mlProb(&mlSol);

  // add system Poisson in mlProb as a Linear Implicit System
  LinearImplicitSystemForSSC & system = mlProb.add_system < LinearImplicitSystemForSSC > ("Poisson");

  // add solution "u" to system
  system.AddSolutionToSystemPDE("T");

  system.SetLinearEquationSolverType(FEMuS_DEFAULT); // GMRES
  //system.SetMgSmoother(ASM_SMOOTHER); // Additive Swartz Method
  // attach the assembling function to system
  system.SetAssembleFunction(AssembleTemperature_AD);

//   
  system.SetMaxNumberOfLinearIterations(200);
  system.SetAbsoluteLinearConvergenceTolerance(1.e-20);



  system.SetMgType(V_CYCLE);

  // initilaize and solve the system

  system.init();

  system.SetSolverFineGrids(RICHARDSON);

  //system.SetPreconditionerFineGrids(IDENTITY_PRECOND);
  system.SetPreconditionerFineGrids(ILU_PRECOND);
  //system.SetPreconditionerFineGrids(JACOBI_PRECOND);
  //system.SetPreconditionerFineGrids(SOR_PRECOND);
  
  system.SetTolerances(1.e-50, 1.e-80, 1.e+50, 10, 10); //GMRES tolerances 
  
  unsigned simulation = 3;
  double scale = 1.;
  
  
  if (simulation  == 0){ //our theory
    system.SetSscLevelSmoother(true); 
    system.SetFactorAndScale(true, scale); 
    system.SetSSCType(SYMMETRIC1111);
  }
  else if (simulation  == 1){ //our reduced symmetric
    system.SetSscLevelSmoother(true); 
    system.SetFactorAndScale(false, scale); 
    system.SetSSCType(SYMMETRIC1111);
  }
  else if (simulation  == 2){ //our reduced asymmetric
    system.SetSscLevelSmoother(true); 
    system.SetFactorAndScale(false, scale); 
    system.SetSSCType(ASYMMETRIC0101);
  }
  else  if(simulation == 3) { //JK
    system.SetSscLevelSmoother(false); 
    system.SetFactorAndScale(true, scale); 
  }
  else if (simulation  == 4){ //BPWX
    system.SetSscLevelSmoother(false); 
    system.SetFactorAndScale(false, scale);
  }
  
  
  system.SetNumberPreSmoothingStep(1); //number of pre and post smoothing
  system.SetNumberPostSmoothingStep(1);
  
  system.ClearVariablesToBeSolved();
  system.AddVariableToBeSolved("All");
  
  system.SetNumberOfSchurVariables(1);
  system.SetElementBlockNumber(4);


  system.SetPrintSolverInfo(false);

  // ====== BEGIN part to re-implement for SSC MGAMR!!! ================
  system.SetOuterSolver(PREONLY);
  system.MGsolve();
  // ====== END part to re-implement for SSC MGAMR!!! ================


  // print solutions
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back("All");

  VTKWriter vtkIO(&mlSol);
  vtkIO.SetDebugOutput(true);
  vtkIO.Write(Files::_application_output_directory, fe_fams_for_files[ FILES_CONTINUOUS_QUADRATIC ], variablesToBePrinted);
  vtkIO.Write(Files::_application_output_directory, fe_fams_for_files[ FILES_CONTINUOUS_LINEAR ], variablesToBePrinted);

  mlMsh.PrintInfo();
  
  return 0;
}





void AssembleTemperature_AD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled

  //  extract pointers to the several objects that we are going to use
  LinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<LinearImplicitSystem> ("Poisson");   // pointer to the linear implicit system named "Poisson"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*           msh         = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*           el          = msh->GetMeshElements();  // pointer to the elem object in msh (level)

  MultiLevelSolution*   mlSol         = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*   sol         = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object


  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level];  // pointer to the equation (level) object
  SparseMatrix*   KK          = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*  RES         = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  bool assembleMatrix = mlPdeSys->GetAssembleMatrix();
  adept::Stack& s = FemusInit::_adeptStack;
  if(assembleMatrix) s.continue_recording();
  else s.pause_recording();

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27
  
  //solution variable
  unsigned solTIndex;
  solTIndex = mlSol->GetIndex("T");    // get the position of "T" in the ml_sol object
  unsigned solTType = mlSol->GetSolutionType(solTIndex);    // get the finite element type for "T"

  std::vector < unsigned > solVIndex(dim);

  unsigned solTPdeIndex;
  solTPdeIndex = mlPdeSys->GetSolPdeIndex("T");    // get the position of "T" in the pdeSys object

  std::vector < adept::adouble >  solT; // local solution

  std::vector < adept::adouble > aResT; // local redidual vector

  std::vector < std::vector < double > > coordX(dim);    // local coordinates
  unsigned coordXType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  solT.reserve(maxSize);
  aResT.reserve(maxSize);

  for(unsigned  k = 0; k < dim; k++) {
    coordX[k].reserve(maxSize);
  }

  std::vector <double> phiT;  // local test function
  std::vector <double> phiT_x; // local test function first order partial derivatives
  std::vector <double> phiT_xx; // local test function second order partial derivatives

  phiT.reserve(maxSize);
  phiT_x.reserve(maxSize * dim);
  phiT_xx.reserve(maxSize * dim2);

  double weight; // gauss point weight

  std::vector < int > sysDof; // local to global pdeSys dofs
  sysDof.reserve(maxSize);

  std::vector < double > Res; // local redidual vector
  Res.reserve(maxSize);

  std::vector < double > Jac;
  Jac.reserve(maxSize * maxSize);

  if(assembleMatrix)
    KK->zero(); // Set to zero all the entries of the Global Matrix

  // element loop: each process loops only on the elements that owns
  for(int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);
    
    short unsigned ielGroup = msh->GetElementGroup(iel);
    //double K = ( ielGroup == 7 || ielGroup == 9 )?  1:0.1;
    
   
    //BEGIN: K for 2D simulations
//     double K = ( ielGroup == 6 || ielGroup == 8 ) ?  0.1 * (rand()%((15 - 5) + 1) + 5) :  (rand()%((15 - 5) + 1) + 5) ;
    //END
    
    
//     if(ielGroup == 7){
//       std::cout << K <<" ";
//     }
    
    unsigned nDofsT = msh->GetElementDofNumber(iel, solTType);    // number of solution element dofs
    unsigned nDofsX = msh->GetElementDofNumber(iel, coordXType);    // number of coordinate element dofs

    // resize local arrays
    sysDof.resize(nDofsT);

    solT.resize(nDofsT);

    for(unsigned  k = 0; k < dim; k++) {
      coordX[k].resize(nDofsX);
    }

    aResT.resize(nDofsT);    //resize
    std::fill(aResT.begin(), aResT.end(), 0);    //set aRes to zero

    // local storage of global mapping and solution
    for(unsigned i = 0; i < nDofsT; i++) {
      unsigned solTDof = msh->GetSolutionDof(i, iel, solTType);    // global to global mapping between solution node and solution dof
      solT[i] = (*sol->_Sol[solTIndex])(solTDof);      // global extraction and local storage for the solution
      sysDof[i] = pdeSys->GetSystemDof(solTIndex, solTPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dofs
    }

    // local storage of coordinates
    for(unsigned i = 0; i < nDofsX; i++) {
      unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);    // global to global mapping between coordinates node and coordinate dof
      for(unsigned k = 0; k < dim; k++) {
        coordX[k][i] = (*msh->GetTopology()->_Sol[k])(coordXDof);      // global extraction and local storage for the element coordinates
      }
    }
    
    
    //BEGIN: K for circle simulations
    
    double xg[3];
    for(unsigned k = 0; k < dim; k++) {
      xg[k] = coordX[k][nDofsX-1];
    }
    
    double r = xg[0] * xg[0] + xg[1] * xg[1] ; 
    
    double K = (1. / r) * (1.+.01 * (rand()%((100 - 0) + 1) + 0)) ; // 1/r * a where a is from 1 to 2 random
    
    //END
    
    // start a new recording of all the operations involving adept::adouble variables
    if(assembleMatrix) s.new_recording();

    // *** Gauss point loop ***
    for(unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solTType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][solTType]->Jacobian(coordX, ig, weight, phiT, phiT_x, phiT_xx);

      // evaluate the solution, the solution derivatives and the coordinates in the gauss point
      adept::adouble solT_gss = 0;
      std::vector < adept::adouble > gradSolT_gss(dim, 0.);

      for(unsigned i = 0; i < nDofsT; i++) {
        solT_gss += phiT[i] * solT[i];

        for(unsigned j = 0; j < dim; j++) {
          gradSolT_gss[j] += phiT_x[i * dim + j] * solT[i];
        }
      }

      // *** phiT_i loop ***
      for(unsigned i = 0; i < nDofsT; i++) {
        adept::adouble gradTgradphiT = 0.;

        for(unsigned j = 0; j < dim; j++) {
          gradTgradphiT +=  phiT_x[i * dim + j] * gradSolT_gss[j];
        }

        
        
        aResT[i] -= (phiT[i] - K * gradTgradphiT) * weight;
      } // end phiT_i loop

    } // end gauss point loop

    // } // endif single element not refined or fine grid loop

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store them in RES
    Res.resize(nDofsT);    //resize

    for(int i = 0; i < nDofsT; i++) {
      Res[i] = -aResT[i].value();
    }

    RES->add_vector_blocked(Res, sysDof);

    //Extarct and store the Jacobian
    if(assembleMatrix) {
      Jac.resize(nDofsT * nDofsT);
      // define the dependent variables
      s.dependent(&aResT[0], nDofsT);

      // define the independent variables
      s.independent(&solT[0], nDofsT);

      // get the and store jacobian matrix (row-major)
      s.jacobian(&Jac[0] , true);
      KK->add_matrix_blocked(Jac, sysDof, sysDof);

      s.clear_independents();
      s.clear_dependents();
    }
  } //end element loop for each process

  RES->close();
  if(assembleMatrix)
    KK->close();

  // ***************** END ASSEMBLY *******************
}










