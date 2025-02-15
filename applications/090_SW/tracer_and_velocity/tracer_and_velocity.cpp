/** tutorial/Ex1
 * This example shows how to:
 * initialize a femus application;
 * define the multilevel-mesh object mlMsh;
 * read from the file ./input/square.neu the coarse-level mesh and associate it to mlMsh;
 * add in mlMsh uniform refined level-meshes;
 * define the multilevel-solution object mlSol associated to mlMsh;
 * add in mlSol different types of finite element solution variables;
 * initialize the solution varables;
 * define vtk and gmv writer objects associated to mlSol;
 * print vtk and gmv binary-format files in ./output directory.
 **/

#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "PetscMatrix.hpp"

#include "TransientSystem.hpp"
#include "LinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"

#include "slepceps.h"
#include <slepcmfn.h>

using namespace femus;

double dt = 1./10.; //= dx / maxWaveSpeed * 0.85;

double ni_h = 100.; // 0.1, 1, 10, 100, 200

double ni_v = 0.0001;

double k_v = 0.0001;

double pi = acos ( -1. );
double k_h = 1/(10*pi);

const unsigned NumberOfLayers = 5;

const double hRest[10]={1,1,1,1,1};
//const double hRest[10]={2,2,2,2,2,2,2,2,2,2};
//const double hRest[20] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

double InitialValueV ( const std::vector < double >& x ) {
  return 1.;
}

double InitialValueH ( const std::vector < double >& x ) {
  return hRest[0];
}


double InitialValueT ( const std::vector < double >& x ) {
  double pi = acos ( -1. );
//   return 17.5 + 25/pi * atan(x[0]/100.);
//   if ( x[0] < 0 ) return 5;
//   else return 30;
  return (- sin(pi*x[0]));
}


double InitialValueB ( const std::vector < double >& x ) {
  return 5; //( H_shelf + H_0 / 2 * (1 + tanh(hh / phi)) );
}


bool SetBoundaryCondition ( const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time ) {
  bool dirichlet = false; //dirichlet
  if ( facename == 1 || facename == 2 ) dirichlet = true;
  value = 0.;
  return dirichlet;
}


void ETD ( MultiLevelProblem& ml_prob );
void ETD2 ( MultiLevelProblem& ml_prob );


int main ( int argc, char** args ) {

  SlepcInitialize ( &argc, &args, PETSC_NULLPTR, PETSC_NULLPTR );

  // init Petsc-MPI communicator
  FemusInit mpinit ( argc, args, MPI_COMM_WORLD );

  // define multilevel mesh
  MultiLevelMesh mlMsh;
  double scalingFactor = 1.;

  unsigned numberOfUniformLevels = 1;
  unsigned numberOfSelectiveLevels = 0;

  unsigned nx = static_cast<unsigned> ( floor ( pow ( 2.,/*11*/6 ) + 0.5 ) ); //Grid cell size = 0.03125 m

  double length = 2; //2 * 1465700.;

  mlMsh.GenerateCoarseBoxMesh ( nx, 0, 0, -length / 2, length / 2, 0., 0., 0., 0., EDGE3, "seventh" );

  //mlMsh.RefineMesh(numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels , NULL);
  mlMsh.PrintInfo();

  // define the multilevel solution and attach the mlMsh object to it
  MultiLevelSolution mlSol ( &mlMsh );

  for ( unsigned i = 0; i < NumberOfLayers; i++ ) {
    char name[10];
    sprintf ( name, "h%d", i );
    mlSol.AddSolution ( name, DISCONTINUOUS_POLYNOMIAL, ZERO, 2);
    sprintf ( name, "v%d", i );
    mlSol.AddSolution ( name, LAGRANGE, FIRST, 2 );
    sprintf ( name, "T%d", i );
    mlSol.AddSolution ( name, DISCONTINUOUS_POLYNOMIAL, ZERO, 2);
    sprintf ( name, "HT%d", i );
    mlSol.AddSolution ( name, DISCONTINUOUS_POLYNOMIAL, ZERO, 2);
  }

  mlSol.AddSolution ( "b", DISCONTINUOUS_POLYNOMIAL, ZERO, 1, false );

  mlSol.AddSolution ( "eta", DISCONTINUOUS_POLYNOMIAL, ZERO, 1, false );

  mlSol.Initialize ( "All" );

  for ( unsigned i = 0; i < NumberOfLayers; i++ ) {
    char name[10];
    sprintf ( name, "v%d", i );
    mlSol.Initialize ( name, InitialValueV );
  }

  for ( unsigned i = 0; i < NumberOfLayers; i++ ) {
    char name[10];
    sprintf ( name, "h%d", i );
    mlSol.Initialize ( name, InitialValueH );
  }

  for ( unsigned i = 0; i < NumberOfLayers; i++ ) {
    char name[10];
    sprintf ( name, "T%d", i );
    mlSol.Initialize ( name, InitialValueT );
  }

  mlSol.Initialize ( "b", InitialValueB );

  mlSol.AttachSetBoundaryConditionFunction ( SetBoundaryCondition );
  mlSol.GenerateBdc ( "All" );

  MultiLevelProblem ml_prob ( &mlSol );

  // ******* Add FEM system to the MultiLevel problem *******
  TransientLinearImplicitSystem& system = ml_prob.add_system < TransientLinearImplicitSystem > ( "SWv" );
  TransientLinearImplicitSystem& system2 = ml_prob.add_system < TransientLinearImplicitSystem > ( "SWt" );
  for ( unsigned i = 0; i < NumberOfLayers; i++ ) {
    char name[10];
//     sprintf ( name, "h%d", i );
//     system.AddSolutionToSystemPDE ( name );
    sprintf ( name, "v%d", i );
    system.AddSolutionToSystemPDE ( name );
    sprintf ( name, "HT%d", i );
    system2.AddSolutionToSystemPDE ( name );
  }
  system.init();
  system2.init();
  
  mlSol.SetWriter ( VTK );
  std::vector<std::string> print_vars;
  print_vars.push_back ( "All" );
  //mlSol.GetWriter()->SetDebugOutput(true);
  mlSol.GetWriter()->Write ( Files::_application_output_directory, fe_fams_for_files[ FILES_CONTINUOUS_LINEAR ], print_vars, 0 );

  unsigned numberOfTimeSteps = 100; //17h=1020 with dt=60, 17h=10200 with dt=6
  for ( unsigned i = 0; i < numberOfTimeSteps; i++ ) {
    system.CopySolutionToOldSolution();
//     dt = 60.;
    ETD ( ml_prob );
//     dt = 60.;
    ETD2 ( ml_prob );
    mlSol.GetWriter()->Write ( Files::_application_output_directory, fe_fams_for_files[ FILES_CONTINUOUS_LINEAR ], print_vars, ( i + 1 ) / 1 );
  }
  return 0;
}


void ETD ( MultiLevelProblem& ml_prob ) {

  const unsigned& NLayers = NumberOfLayers;

  adept::Stack& s = FemusInit::_adeptStack;

  TransientLinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<TransientLinearImplicitSystem> ( "SWv" ); // pointer to the linear implicit system named "Poisson"

  unsigned level = ml_prob._ml_msh->GetNumberOfLevels() - 1u;

  Mesh* msh = ml_prob._ml_msh->GetLevel ( level ); // pointer to the mesh (level) object
  elem* el = msh->GetMeshElements();  // pointer to the elem object in msh (level)

  MultiLevelSolution* mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution* sol = ml_prob._ml_sol->GetSolutionLevel ( level ); // pointer to the solution (level) object

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object

  SparseMatrix* KK = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector* RES = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)
  NumericVector* EPS = pdeSys->_EPS; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned    nprocs = msh->n_processors(); // get the process_id (for parallel computation)


  //solution variable
  std::vector < unsigned > solIndexh ( NLayers );
  std::vector < unsigned > solPdeIndexh ( NLayers );

  std::vector < unsigned > solIndexv ( NLayers );
  std::vector < unsigned > solPdeIndexv ( NLayers );

  //std::vector < unsigned > solIndexHT ( NLayers );
  //std::vector < unsigned > solPdeIndexHT ( NLayers );


  std::vector < unsigned > solIndexT ( NLayers );

  std::vector < int > l2GMapRow; // local to global mapping
  std::vector < int > l2GMapColumn; // local to global mapping

  for ( unsigned i = 0; i < NLayers; i++ ) {
    char name[10];
    sprintf ( name, "h%d", i );
    solIndexh[i] = mlSol->GetIndex ( name ); // get the position of "hi" in the sol object
    //solPdeIndexh[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "hi" in the pdeSys object

    sprintf ( name, "v%d", i );
    solIndexv[i] = mlSol->GetIndex ( name ); // get the position of "vi" in the sol object
    solPdeIndexv[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "vi" in the pdeSys object

    //sprintf ( name, "HT%d", i );
    //solIndexHT[i] = mlSol->GetIndex ( name ); // get the position of "Ti" in the sol object
    //solPdeIndexHT[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "Ti" in the pdeSys object

    sprintf ( name, "T%d", i );
    solIndexT[i] = mlSol->GetIndex ( name ); // get the position of "Ti" in the sol object

  }

  unsigned solTypeh = mlSol->GetSolutionType ( solIndexh[0] ); // get the finite element type for "hi"
  unsigned solTypev = mlSol->GetSolutionType ( solIndexv[0] ); // get the finite element type for "vi"
  unsigned solTypeT = mlSol->GetSolutionType ( solIndexT[0] ); // get the finite element type for "Ti"

  KK->zero();
  RES->zero();

  MatSetOption ( ( static_cast<PetscMatrix*> ( KK ) )->mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE );

  unsigned start = msh->_dofOffset[solTypev][iproc] + 1;
  unsigned end = msh->_dofOffset[solTypev][iproc + 1] - 1;
  for ( unsigned i =  start; i <  end; i++ ) {

    std::vector < double > solhm ( NLayers );
    std::vector < double > solhp ( NLayers );
    std::vector < adept::adouble > solvm ( NLayers ); // local coordinates
    std::vector < adept::adouble > solv ( NLayers ); // local coordinates
    std::vector < adept::adouble > solvp ( NLayers ); // local coordinates
    std::vector < double > solTm ( NLayers ); // local coordinates
    std::vector < double > solTp ( NLayers ); // local coordinates

    //vector< adept::adouble > aResh ( NLayers );
    std::vector < adept::adouble > aResv ( NLayers );
    //vector< adept::adouble > aResHT ( NLayers );


    l2GMapRow.resize ( NLayers );
    l2GMapColumn.resize ( 3 * NLayers );

    std::fill ( aResv.begin(), aResv.end(), 0 ); //set aRes to zero

    for ( unsigned j = 0; j < NLayers; j++ ) {

      solv[j] = ( *sol->_Sol[solIndexv[j]] ) ( i );
      l2GMapRow[j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 0, i );
      l2GMapColumn[j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 0, i );

      solvm[j] = ( *sol->_Sol[solIndexv[j]] ) ( i - 1 );
      solvp[j] = ( *sol->_Sol[solIndexv[j]] ) ( i + 1 );
      l2GMapColumn[1 * NLayers + j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 0, i - 1 );
      l2GMapColumn[2 * NLayers + j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 1, i );


      solhm[j] = ( *sol->_Sol[solIndexh[j]] ) ( i - 1 );
      solTm[j] = ( *sol->_Sol[solIndexT[j]] ) ( i - 1 );

      //l2GMapColumn[3 * NLayers + j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i - 1 );
      //l2GMapColumn[4 * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i - 1 );

      solhp[j]  = ( *sol->_Sol[solIndexh[j]] ) ( i );
      solTp[j] = ( *sol->_Sol[solIndexT[j]] ) ( i );

      //l2GMapColumn[4 * NLayers + j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i );
      //l2GMapColumn[6 * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i );

    }
    s.new_recording();

    std::vector <double> xm ( 2 );
    std::vector <double> xp ( 2 );
    for ( unsigned j = 0; j < 2; j++ ) {
      unsigned xDofm  = msh->GetSolutionDof ( j, i - 1, 2 ); // global to global mapping between coordinates node and coordinate dof
      xm[j] = ( *msh->GetTopology()->_Sol[0] ) ( xDofm ); // global extraction and local storage for the element coordinates

      unsigned xDofp  = msh->GetSolutionDof ( j, i, 2 ); // global to global mapping between coordinates node and coordinate dof
      xp[j] = ( *msh->GetTopology()->_Sol[0] ) ( xDofp ); // global extraction and local storage for the element coordinates
    }
    double dxm = xm[1] - xm[0];
    double dxp = xp[1] - xp[0];

    double bm = 20; //( H_shelf + H_0 / 2 * (1 + tanh(hhm / phi)) );

    double bp = 20; //( H_shelf + H_0 / 2 * (1 + tanh(hhp / phi)) );

    double hTotm = 0.;
    double hTotp = 0.;

    std::vector < double > hALEm ( NLayers, 0. );
    std::vector < double > hALEp ( NLayers, 0. );

    hALEm[0] = hRest[0] + ( hTotm - bm );
    hALEp[0] = hRest[0] + ( hTotp - bp );
    for ( unsigned k = 1; k < NLayers; k++ ) {
      hALEm[k] = hRest[k];
      hALEp[k] = hRest[k];
    }

//     std::vector < double > wm(NLayers+1, 0.);
//     std::vector < double > wp(NLayers+1, 0.);
//
//     for(unsigned k = NLayers; k>1; k--){
//       wm[k-1] = wm[k] -  solhm[k-1].value() * (solv[k-1].value() - solvm[k-1].value() )/dxm - ( hALEm[k-1] - solhm[k-1].value()) / dt;
//       wp[k-1] = wp[k] -  solhp[k-1].value() * (solvp[k-1].value() - solv[k-1].value() )/dxp - ( hALEp[k-1] - solhp[k-1].value()) / dt;
//     }

//     double dx = 0.5 * ( dxm + dxp );
//     for ( unsigned k = NLayers; k > 1; k-- ) {
//       w[k - 1] = w[k] - ( solhp[k - 1].value() * ( 0.5 * ( solv[k - 1].value() + solvp[k - 1].value() ) )
//                           - solhm[k - 1].value() * ( 0.5 * ( solv[k - 1].value() + solvm[k - 1].value() ) ) ) / dx
//                  - ( 0.5 * ( hALEm[k - 1] + hALEp[k - 1] )  - 0.5 * ( solhm[k - 1].value() + solhp[k - 1].value() ) ) / dt;
//       //std::cout<<"w in u equation"<<w[k-1]<<std::endl;
// 
//     }

    std::vector < double > w ( NLayers + 1, 0. );
    
    std::vector < double > zMidm ( NLayers );
    std::vector < double > zMidp ( NLayers );
    for ( unsigned k = 0; k < NLayers; k++ ) {
      zMidm[k] = -bm + solhm[k] / 2.;
      zMidp[k] = -bp + solhp[k] / 2.;
      for ( unsigned i = k + 1; i < NLayers; i++ ) {
        zMidm[k] += solhm[i];
	zMidp[k] += solhp[i];
      }
    }
        
    for ( unsigned k = NLayers; k > 1; k-- ) {
      w[k-1] = - 0.1 * 0.5 * (zMidm[k-1]+zMidp[k-1]);
    }
    

    for ( unsigned k = 0; k < NLayers; k++ ) {
    //BEGIN SIN FOR VELOCITY  
      aResv[k] += sin(solv[k]) /*- w[k]*solv[k]*/;    
      
//       if ( k < NLayers - 1 ) {
//         aResv[k] += w[k + 1] * 0.5 * ( solv[k] + solv[k + 1] );
//       }
//       if ( k > 0 ) {
//         aResv[k] -= w[k] * 0.5 * ( solv[k - 1] + solv[k] );
//       }
      //END SIN FOR VELOCITY
      
//       adept::adouble vMidm = 0.5 * ( solvm[k] + solv[k] );
//       adept::adouble fvm = 0.5 * vMidm * vMidm /*+ Pm[k]*/;
//       aResv[k] +=  fvm / dxm;
// 
//       adept::adouble vMidp = 0.5 * ( solv[k] + solvp[k] );
//       adept::adouble fvp = 0.5 * vMidp * vMidp /*+ Pp[k]*/;
//       aResv[k] -=  fvp / dxp;
 
      //if ( i > start ) {
        //aResv[k] += solv[k] * solvm[k]  / dxm; //second order
        if ( solvm[k] > 0 ) {
          aResv[k] += solvm[k] / dxm;
        }
        else {
          aResv[k] += solv[k] / dxm;
        }
      //}
      //if ( i < end - 1 ) {
        //aResv[k] -= solv[k] * solvp[k]  / dxp; //second order
        if ( solvp[k] > 0 ) {
          aResv[k] -= solv[k] / dxp; //first order upwind
        }
        else {
          aResv[k] -= solvp[k] / dxp; //first order upwind
        }
      //}
      adept::adouble deltaZt = 0.;
      adept::adouble deltaZb = 0.;
      adept::adouble ht = 0.;
      adept::adouble hb = 0.;
      if ( k > 0 ) {
        ht = ( solhm[k - 1] + solhm[k] + solhp[k - 1] + solhp[k] ) / 4.;
        deltaZt = ( solv[k - 1] - solv[k] ) / ht;
        aResv[k] -= 0.5 * w[k] * deltaZt;
      }
      else {
        ht = 0.5 * ( solhm[k] + solhp[k] );
        deltaZt = 0.* ( 0. - solv[k] ) / ht;
      }
      if ( k < NLayers - 1 ) {
        hb = ( solhm[k] + solhm[k + 1] + solhp[k] + solhp[k + 1] ) / 4.;
        deltaZb = ( solv[k] - solv[k + 1] ) / hb;
        aResv[k] -= 0.5 * w[k + 1] * deltaZb;
      }
      else {
        hb = 0.5 * ( solhm[k] + solhp[k] );
        deltaZb = 0.* ( solv[k] - 0. ) / hb;
      }
      
//       aResv[k] += ni_h * ( solvm[k] - solv[k] ) / ( dxm * 0.5 * ( dxm + dxp ) ); // horizontal diffusion
//       aResv[k] += ni_h * ( solvp[k] - solv[k] ) / ( dxp * 0.5 * ( dxm + dxp ) ); // horizontal diffusion
// 
//       aResv[k] += ni_v * ( deltaZt - deltaZb ) / ( ( ht + hb ) / 2. ); // vertical diffusion
    }

    std::vector < double > Res ( NLayers ); // local redidual vector
    for ( unsigned k = 0; k < NLayers; k++ ) {
      Res[k] =  aResv[k].value();
      //std::cout<< "Res["<<k<<"] = " << Res[k] <<std::endl;
    }

    RES->add_vector_blocked ( Res, l2GMapRow );

    s.dependent ( &aResv[0], NLayers );

    // define the independent variables
    s.independent ( &solv[0], NLayers );
    s.independent ( &solvm[0], NLayers );
    s.independent ( &solvp[0], NLayers );
    //s.independent ( &solhm[0], NLayers );
    //s.independent ( &solHTm[0], NLayers );
    //s.independent ( &solhp[0], NLayers );
    //s.independent ( &solHTp[0], NLayers );

    // get the jacobian matrix (ordered by row major )
    std::vector < double > Jac ( NLayers * NLayers * 3 );
    s.jacobian ( &Jac[0], true );

    //store K in the global matrix KK
    KK->add_matrix_blocked ( Jac, l2GMapRow, l2GMapColumn );

    s.clear_independents();
    s.clear_dependents();

  }


  RES->close();
  KK->close();



//   PetscViewer    viewer;
//   PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,900,900,&viewer);
//   PetscObjectSetName((PetscObject)viewer,"FSI matrix");
//   PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);
//   MatView((static_cast<PetscMatrix*>(KK))->mat(),viewer);
//   double a;
//   std::cin>>a;
//

//  abort();
  MFN mfn;
  Mat A = ( static_cast<PetscMatrix*> ( KK ) )->mat();
  FN f, f1, f2, f3 , f4;

  //std::cout << "dt = " << dt << " dx = "<< dx << " maxWaveSpeed = "<<maxWaveSpeed << std::endl;
  std::cout << "dt = " << dt << std::endl;

  //dt = 100.;

  Vec v = ( static_cast< PetscVector* > ( RES ) )->vec();
  Vec y = ( static_cast< PetscVector* > ( EPS ) )->vec();

  MFNCreate ( PETSC_COMM_WORLD, &mfn );

  MFNSetOperator ( mfn, A );
  MFNGetFN ( mfn, &f );

  FNPhiSetIndex ( f, 1 );
  FNSetType ( f, FNPHI );
// FNView(f,PETSC_VIEWER_STDOUT_WORLD);

  FNSetScale ( f, dt, dt );
  MFNSetFromOptions ( mfn );

  MFNSolve ( mfn, v, y );
  MFNDestroy ( &mfn );

  sol->UpdateSol ( mlPdeSys->GetSolPdeIndex(), EPS, pdeSys->KKoffset );

  unsigned solIndexeta = mlSol->GetIndex ( "eta" );
  unsigned solIndexb = mlSol->GetIndex ( "b" );
  sol->_Sol[solIndexeta]->zero();
  for ( unsigned k = 0; k < NumberOfLayers; k++ ) {
    sol->_Sol[solIndexeta]->add ( *sol->_Sol[solIndexh[k]] );
  }
  sol->_Sol[solIndexeta]->add ( -1, *sol->_Sol[solIndexb] );

//   for ( unsigned k = 0; k < NumberOfLayers; k++ ) {
//     for ( unsigned i =  msh->_dofOffset[solTypeHT][iproc]; i <  msh->_dofOffset[solTypeHT][iproc + 1]; i++ ) {
//       double valueHT = ( *sol->_Sol[solIndexHT[k]] ) ( i );
//       double valueH = ( *sol->_Sol[solIndexh[k]] ) ( i );
// 
//       double valueT = valueHT / valueH;
// 
//       sol->_Sol[solIndexT[k]]->set ( i, valueT );
//     }
// 
//     sol->_Sol[solIndexT[k]]->close();
// 
//   }


}


void ETD2 ( MultiLevelProblem& ml_prob ) {

  const unsigned& NLayers = NumberOfLayers;

  adept::Stack& s = FemusInit::_adeptStack;

  TransientLinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<TransientLinearImplicitSystem> ( "SWt" ); // pointer to the linear implicit system named "Poisson"

  unsigned level = ml_prob._ml_msh->GetNumberOfLevels() - 1u;

  Mesh* msh = ml_prob._ml_msh->GetLevel ( level ); // pointer to the mesh (level) object
  elem* el = msh->GetMeshElements();  // pointer to the elem object in msh (level)

  MultiLevelSolution* mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution* sol = ml_prob._ml_sol->GetSolutionLevel ( level ); // pointer to the solution (level) object

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object

  SparseMatrix* KK = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector* RES = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)
  NumericVector* EPS = pdeSys->_EPS; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned    nprocs = msh->n_processors(); // get the process_id (for parallel computation)


  //solution variable
  std::vector < unsigned > solIndexh ( NLayers );
  //std::vector < unsigned > solPdeIndexh ( NLayers );

  std::vector < unsigned > solIndexv ( NLayers );
  //std::vector < unsigned > solPdeIndexv ( NLayers );

  std::vector < unsigned > solIndexHT ( NLayers );
  std::vector < unsigned > solPdeIndexHT ( NLayers );

  std::vector < unsigned > solIndexT ( NLayers );

  std::vector < int > l2GMapRow; // local to global mapping
  std::vector < int > l2GMapColumn; // local to global mapping

  for ( unsigned i = 0; i < NLayers; i++ ) {
    char name[10];
    sprintf ( name, "h%d", i );
    solIndexh[i] = mlSol->GetIndex ( name ); // get the position of "hi" in the sol object
    //solPdeIndexh[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "hi" in the pdeSys object

    sprintf ( name, "v%d", i );
    solIndexv[i] = mlSol->GetIndex ( name ); // get the position of "vi" in the sol object
    //solPdeIndexv[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "vi" in the pdeSys object

    sprintf ( name, "HT%d", i );
    solIndexHT[i] = mlSol->GetIndex ( name ); // get the position of "Ti" in the sol object
    solPdeIndexHT[i] = mlPdeSys->GetSolPdeIndex ( name ); // get the position of "Ti" in the pdeSys object

    sprintf ( name, "T%d", i );
    solIndexT[i] = mlSol->GetIndex ( name ); // get the position of "Ti" in the sol object

  }

  unsigned solTypeh = mlSol->GetSolutionType ( solIndexh[0] ); // get the finite element type for "hi"
  unsigned solTypev = mlSol->GetSolutionType ( solIndexv[0] ); // get the finite element type for "vi"
  unsigned solTypeHT = mlSol->GetSolutionType ( solIndexHT[0] ); // get the finite element type for "Ti"

  KK->zero();
  RES->zero();

  MatSetOption ( ( static_cast<PetscMatrix*> ( KK ) )->mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE );

  for ( unsigned k = 0; k < NumberOfLayers; k++ ) {
    for ( unsigned i =  msh->_dofOffset[solTypeHT][iproc]; i <  msh->_dofOffset[solTypeHT][iproc + 1]; i++ ) {
      double valueT = ( *sol->_SolOld[solIndexT[k]] ) ( i );
      double valueH = ( *sol->_Sol[solIndexh[k]] ) ( i );

      double valueHT = valueT * valueH;

      sol->_Sol[solIndexHT[k]]->set ( i, valueHT );
    }
    sol->_Sol[solIndexHT[k]]->close();
  }

  unsigned start = msh->_dofOffset[solTypeHT][iproc];
  unsigned end = msh->_dofOffset[solTypeHT][iproc + 1];
  for ( unsigned i =  start; i <  end; i++ ) {

    std::vector < double > solhm ( NLayers );
    std::vector < double > solh ( NLayers ); // local coordinates
    std::vector < double > solhp ( NLayers );
    std::vector < double > solvm ( NLayers ); // local coordinates
    std::vector < double > solvp ( NLayers ); // local coordinates
    std::vector < adept::adouble > solHTm ( NLayers ); // local coordinates
    std::vector < adept::adouble > solHT ( NLayers ); // local coordinates
    std::vector < adept::adouble > solHTp ( NLayers ); // local coordinates
    
    std::vector < adept::adouble > solHTmm(NLayers);    // local coordinates
    std::vector < adept::adouble > solHTpp(NLayers);    // local coordinates

    //vector< adept::adouble > aResh ( NLayers );
    //vector< adept::adouble > aResv ( NLayers );
    std::vector < adept::adouble > aResHT ( NLayers );

    unsigned bc1 = ( i == start ) ? 0 : 1;
    unsigned bc2 = ( i == end - 1 ) ? 0 : 1;
    
    unsigned bc3 = ( i > start + 1) ? 1 : 0;
    unsigned bc4 = ( i < end - 2 ) ? 1 : 0;

    l2GMapRow.resize ( NLayers );
    l2GMapColumn.resize ( ( 1 + bc1 + bc2 ) * NLayers );
    //l2GMapColumn.resize ( ( 1 + bc1 + bc2 + bc3 + bc4) * NLayers );

    //std::fill ( aResh.begin(), aResh.end(), 0 ); //set aRes to zero
    std::fill ( aResHT.begin(), aResHT.end(), 0 ); //set aRes to zero

    for ( unsigned j = 0; j < NLayers; j++ ) {

      solh[j] = ( *sol->_Sol[solIndexh[j]] ) ( i );
      solHT[j] = ( *sol->_Sol[solIndexHT[j]] ) ( i );
      //l2GMapRow[j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i );
      l2GMapRow[/*NLayers +*/ j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i );

      //l2GMapColumn[j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i );
      l2GMapColumn[/*NLayers +*/ j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i );

      solvm[j] = ( *sol->_Sol[solIndexv[j]] ) ( i );
      solvp[j] = ( *sol->_Sol[solIndexv[j]] ) ( i + 1 );

      //l2GMapColumn[2 * NLayers + j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 0, i );
      //l2GMapColumn[3 * NLayers + j] = pdeSys->GetSystemDof ( solIndexv[j], solPdeIndexv[j], 1, i );

      if ( i > start ) {
        solhm[j] = ( *sol->_Sol[solIndexh[j]] ) ( i - 1 );
        solHTm[j] = ( *sol->_Sol[solIndexHT[j]] ) ( i - 1 );

        //l2GMapColumn[4 * NLayers + j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i - 1 );
        l2GMapColumn[NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i - 1 );

      }

      if ( i < end - 1 ) {
        solhp[j] = ( *sol->_Sol[solIndexh[j]] ) ( i + 1 );
        solHTp[j] = ( *sol->_Sol[solIndexHT[j]] ) ( i + 1 );

        //l2GMapColumn[ ( 4 + 2 * bc1 ) * NLayers + j] = pdeSys->GetSystemDof ( solIndexh[j], solPdeIndexh[j], 0, i + 1 );
        l2GMapColumn[ ( 1 + bc1) * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i + 1 );
      }
      
//       if ( i > start + 1 ) {
//         solHTmm[j] = ( *sol->_Sol[solIndexHT[j]] ) ( i - 2 );
//         if (i == end - 1) l2GMapColumn[( 1 + bc1 ) * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i - 2 );
//         else l2GMapColumn[( (1 + bc1) + bc3 ) * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i - 2 );
//       }
//     
//       if ( i < end - 2 ) {
//         solHTpp[j] = ( *sol->_Sol[solIndexHT[j]] ) ( i + 2 );
//         l2GMapColumn[( (1 + bc1) + bc3 + bc4 ) * NLayers + j] = pdeSys->GetSystemDof ( solIndexHT[j], solPdeIndexHT[j], 0, i + 2 );
//       }
      
    }
    
    s.new_recording();

    std::vector < double > x ( 2 ); // local coordinates
    for ( unsigned j = 0; j < 2; j++ ) {
      unsigned xDof  = msh->GetSolutionDof ( j, i, 2 ); // global to global mapping between coordinates node and coordinate dof
      x[j] = ( *msh->GetTopology()->_Sol[0] ) ( xDof ); // global extraction and local storage for the element coordinates
    }
    double dx = x[1] - x[0];

    double b = 5; //( H_shelf + H_0 / 2 * (1 + tanh(hh / phi)) );

    double hTot = 0.;
    for ( unsigned k = 0; k < NLayers; k++ ) {
      hTot += solh[k]/*.value()*/;
    }

    std::vector < double > hALE ( NLayers, 0. );

    hALE[0] = hRest[0] + ( hTot - b );
    for ( unsigned k = 1; k < NLayers; k++ ) {
      hALE[k] = hRest[k];
    }

    std::vector < double > w ( NLayers + 1, 0. );
    
//     NEW w
//     for ( unsigned k = NLayers; k > 1; k-- ) {
//       w[k - 1] = w[k] - ( hALE[k - 1] - solh[k - 1]) / dt; 
//       if(bc2){ 
//         w[k - 1] -=   0.5 * ( solh[k - 1] + solhp[k - 1] ) * solvp[k - 1] /dx;
//       }
//       else{
//         w[k - 1] -=   solh[k - 1] * 1 /dx;  
//       }
//       if(bc1){
//         w[k - 1] +=   0.5 * ( solh[k - 1] + solhm[k - 1] ) * solvm[k - 1] /dx;
//       }
//       else{
//         w[k - 1] +=   solh[k - 1] * 1 /dx;   
//       }
//       //std::cout<< w[k-1] << " ";
//     }
//     //std::cout<<std::endl;
    
//     OLD w     
//     for(unsigned k = 1; k < NLayers; k++){
//       w[k] = w[k-1] + (  0.5 * ( solh[k-1].value() + solhp[k-1].value() ) * solvp[k-1].value()
// 		       - 0.5 * ( solh[k-1].value() + solhm[k-1].value() ) * solvm[k-1].value() )/dx
// 		    + ( hALE[k-1] - solh[k-1].value()) / dt;//TODO
// 		    //std::cout<<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"<<w[k-1]<<std::endl;
//     }
    
    std::vector < double > zMid ( NLayers );
    for ( unsigned k = 0; k < NLayers; k++ ) {
      zMid[k] = -b + solh[k] / 2.;
      for ( unsigned i = k + 1; i < NLayers; i++ ) {
        zMid[k] += solh[i];
      }
    }
    
    for ( unsigned k = NLayers; k > 1; k-- ) {
      w[k-1] = - 0.1 * zMid[k-1];
    }
    

    for ( unsigned k = 0; k < NLayers; k++ ) {

      //BEGIN FIRST ORDER
      if ( i > start ) {
        //aResHT[k] += 0.5 * (solHTm[k] + solHT[k]) * solvm[k]  / dx; //second order
        if ( solvm[k] > 0 ) {
          aResHT[k] += solHTm[k] * solvm[k] / dx;
        }
        else {
          aResHT[k] += solHT[k] * solvm[k] / dx;
        }
      }
      if ( i < end - 1 ) {
        //aResHT[k] -= 0.5 * (solHT[k] + solHTp[k]) * solvp[k]  / dx; //second order
        if ( solvp[k] > 0 ) {
          aResHT[k] -= solHT[k] * solvp[k] / dx; //first order upwind
        }
        else {
          aResHT[k] -= solHTp[k] * solvp[k] / dx; //first order upwind
        }
      }
//       else{
//         aResHT[k] -= solHT[k] /*.value()*/  / dx; //first order upwind
//       }
      //END

      //BEGIN THIRD ORDER
//       if ( i > start ) {
//         aResHT[k] += 0.5 * ( solHTm[k].value() + solHT[k].value() ) * solvm[k] / dx;
//         if ( solvm[k] > 0 ) {
//           if ( i > start + 1 ) {
//             aResHT[k] += - 1. / 6. * ( solHT[k].value() - 2.*solHTm[k].value() + solHTmm[k].value() ) * solvm[k]  / dx;
//           }
//         }
//         else {
//           if ( i < end - 1 ) {
//             aResHT[k] += - 1. / 6. * ( solHTp[k].value() - 2.*solHT[k].value() + solHTm[k].value() ) * solvm[k]  / dx;
//           }
//         }
//       }
//       if ( i < end - 1 ) {
//         aResHT[k] -= 0.5 * ( solHTp[k].value() + solHT[k].value() ) * solvp[k] / dx;
//         if ( solvp[k] > 0 ) {
//           if (i > start) {
//             aResHT[k] -= - 1. / 6. * ( solHTp[k].value() - 2.*solHT[k].value() + solHTm[k].value() ) * solvp[k]  / dx;
//           }
//         }
//         else {
//           if ( i < end - 2 ) {
//             aResHT[k] -= - 1. / 6. * ( solHTpp[k].value() - 2.*solHTp[k].value() + solHT[k].value() ) * solvp[k]  / dx;
//           }
//         }
//       }
      //END

      if ( k < NLayers - 1 ) {
        aResHT[k] += w[k + 1] * 0.5 * ( solHT[k].value() / solh[k] + solHT[k + 1].value() / solh[k + 1] );
      }
      if ( k > 0 ) {
        aResHT[k] -= w[k] * 0.5 * ( solHT[k - 1].value() / solh[k - 1] + solHT[k].value() / solh[k] );
      }
      
      adept::adouble deltaZt = 0.;
      adept::adouble deltaZb = 0.;
      adept::adouble ht = 0.;
      adept::adouble hb = 0.;
      if ( k > 0 ){
	ht = (solhm[k-1] + solhm[k] + solhp[k-1] + solhp[k]) / 4.;
	deltaZt = ( solHT[k-1] - solHT[k] ) / ht;
	//aResv[k] -= 0.5 * w[k] * deltaZt;
      }
      else{
	ht = 0.5 * (solhm[k] + solhp[k]);
	deltaZt = 0.*( 0. - solHT[k] ) / ht;
      }
      if (k < NLayers - 1) {
	hb = (solhm[k] + solhm[k+1] + solhp[k] + solhp[k+1] ) / 4.;
	deltaZb = (solHT[k] - solHT[k+1]) / hb;
	//aResv[k] -= 0.5 * w[k+1] * deltaZb; 
      }
      else{
	hb = 0.5 * (solhm[k] + solhp[k]);
	deltaZb = 0.*(solHT[k] - 0.) / hb;
      }
      
      //std::cout<<"AAAAAAAAAAAAAAAAAAAAAAAAAA"<<deltaZt - deltaZb<<std::endl;
      
      aResHT[k] += solhm[k] * k_v * (deltaZt - deltaZb) / ( (ht + hb) / 2. ); // vertical diffusion

      //aResHT[k] += ((solhp[k] - solhm[k]) * k_v * (solHTp[k] - solHTm[k])) / (dx*dx); // horizontal diffusion
      aResHT[k] += k_h * solh[k] * (solHTm[k] - solHT[k])/(dx*dx); // horizontal diffusion
      aResHT[k] += k_h * solh[k] * (solHTp[k] - solHT[k])/(dx*dx); // horizontal diffusion

    }

    std::vector < double > Res ( NLayers ); // local redidual vector
    for ( unsigned k = 0; k < NLayers; k++ ) {
      //Res[k] =  aResh[k].value();
      Res[k] =  aResHT[k].value();
      //std::cout<< "Res["<<k<<"] = " << Res[k] <<std::endl;
      //std::cout<< "Res["<<NLayers+k<<"] = " << Res[NLayers+k] <<std::endl;
    }

    RES->add_vector_blocked ( Res, l2GMapRow );

    //s.dependent ( &aResh[0], NLayers );
    s.dependent ( &aResHT[0], NLayers );

    // define the independent variables
    //s.independent ( &solh[0], NLayers );
    s.independent ( &solHT[0], NLayers );
    //s.independent ( &solvm[0], NLayers );
    //s.independent ( &solvp[0], NLayers );
    if ( i > start ) {
      //s.independent ( &solhm[0], NLayers );
      s.independent ( &solHTm[0], NLayers );
    }
    if ( i < end - 1 ) {
      //s.independent ( &solhp[0], NLayers );
      s.independent ( &solHTp[0], NLayers );
    }
/*    if ( i > start + 1) {
      s.independent ( &solHTmm[0], NLayers );
    }
    if ( i < end - 2 ) {
      s.independent ( &solHTpp[0], NLayers );
    } */   
    
    // get the jacobian matrix (ordered by row major )
    std::vector < double > Jac ( NLayers * NLayers * ( 1 + bc1 + bc2 ) );
    //vector < double > Jac ( NLayers * NLayers * ( 1 + bc1 + bc2 + bc3 +bc4 ) );
    s.jacobian ( &Jac[0], true );

    //store K in the global matrix KK
    KK->add_matrix_blocked ( Jac, l2GMapRow, l2GMapColumn );

    s.clear_independents();
    s.clear_dependents();

  }

  RES->close();
  KK->close();



//   PetscViewer    viewer;
//   PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,900,900,&viewer);
//   PetscObjectSetName((PetscObject)viewer,"FSI matrix");
//   PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);
//   MatView((static_cast<PetscMatrix*>(KK))->mat(),viewer);
//   double a;
//   std::cin>>a;
// //

//  abort();
  MFN mfn;
  Mat A = ( static_cast<PetscMatrix*> ( KK ) )->mat();
  FN f, f1, f2, f3 , f4;

  //std::cout << "dt = " << dt << " dx = "<< dx << " maxWaveSpeed = "<<maxWaveSpeed << std::endl;
  std::cout << "dt = " << dt << std::endl;

  //dt = 100.;

  Vec v = ( static_cast< PetscVector* > ( RES ) )->vec();
  Vec y = ( static_cast< PetscVector* > ( EPS ) )->vec();

  MFNCreate ( PETSC_COMM_WORLD, &mfn );

  MFNSetOperator ( mfn, A );
  MFNGetFN ( mfn, &f );

  FNPhiSetIndex ( f, 1 );
  FNSetType ( f, FNPHI );
// FNView(f,PETSC_VIEWER_STDOUT_WORLD);

  FNSetScale ( f, dt, dt );
  MFNSetFromOptions ( mfn );

  MFNSolve ( mfn, v, y );
  MFNDestroy ( &mfn );

  sol->UpdateSol ( mlPdeSys->GetSolPdeIndex(), EPS, pdeSys->KKoffset );

  unsigned solIndexeta = mlSol->GetIndex ( "eta" );
  unsigned solIndexb = mlSol->GetIndex ( "b" );
  sol->_Sol[solIndexeta]->zero();
  for ( unsigned k = 0; k < NumberOfLayers; k++ ) {
    sol->_Sol[solIndexeta]->add ( *sol->_Sol[solIndexh[k]] );
  }
  sol->_Sol[solIndexeta]->add ( -1, *sol->_Sol[solIndexb] );

  for ( unsigned k = 0; k < NumberOfLayers; k++ ) {
    for ( unsigned i =  msh->_dofOffset[solTypeHT][iproc]; i <  msh->_dofOffset[solTypeHT][iproc + 1]; i++ ) {
      double valueHT = ( *sol->_Sol[solIndexHT[k]] ) ( i );
      double valueH = ( *sol->_Sol[solIndexh[k]] ) ( i );

      double valueT = valueHT / valueH;
      if (i == 0) valueT = 0.;
//       if (i == msh->_dofOffset[solTypeHT][iproc + 1] - 1 ) valueT = 0.;
      
      
      sol->_Sol[solIndexT[k]]->set ( i, valueT );
    }

    sol->_Sol[solIndexT[k]]->close();

  }


}

