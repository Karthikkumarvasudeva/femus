/*=========================================================================

 Program: FEMuS
 Module: MultiLevelProblem
 Authors: Eugenio Aulisa, Simone Bnà, Giorgio Bornia

 Copyright (c) FEMuS
 All rights reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

//----------------------------------------------------------------------------
// includes :
//----------------------------------------------------------------------------
#include "MultiLevelProblem.hpp"

#include "GeomElTypeEnum.hpp"

#include "MultiLevelSolution.hpp"

#include "TransientSystem.hpp"
#include "LinearImplicitSystem.hpp"
#include "NonLinearImplicitSystem.hpp"

#include <iostream>


namespace femus {




// ===  Constructors / Destructor - BEGIN =================
MultiLevelProblem::MultiLevelProblem():
				      _files(NULL),
				      _app_specs_ptr(NULL)
{
    

}


MultiLevelProblem::MultiLevelProblem( MultiLevelSolution *ml_sol):
				      _ml_sol(ml_sol),
				      _ml_msh(ml_sol->GetMLMesh()),
				      _gridn(_ml_msh->GetNumberOfLevels()),
				      _files(NULL),
				      _app_specs_ptr(NULL)
{

}



MultiLevelProblem::~MultiLevelProblem(){
  for( system_iterator iterator = _systems.begin(); iterator != _systems.end(); iterator++) {
    delete iterator->second;
  }
}
// ===  Constructors / Destructor - END =================




// ===  Systems - BEGIN =================

System & MultiLevelProblem::add_system (const std::string& sys_type,
				      const std::string& name)
{
  // If the user already built a system with this name, we'll
  // trust them and we'll use it.  That way they can pre-add
  // non-standard derived system classes, and if their restart file
  // has some non-standard sys_type we won't throw an error.
  if (_systems.count(name))
  {
      return this->get_system(name);
  }
  // Build a basic System
  else if (sys_type == "Basic")
    this->add_system<System> (name);

  // Build a Newmark system
//   else if (sys_type == "Newmark")
//     this->add_system<NewmarkSystem> (name);

  // build a transient implicit linear system
  else if ((sys_type == "Transient") ||
	   (sys_type == "TransientImplicit") ||
	   (sys_type == "TransientLinearImplicit"))
    this->add_system<TransientLinearImplicitSystem> (name);

  // build a transient implicit nonlinear system
  else if (sys_type == "TransientNonlinearImplicit")
    this->add_system<TransientNonlinearImplicitSystem> (name);

  // build a linear implicit system
  else if (sys_type == "LinearImplicit")
    this->add_system<LinearImplicitSystem> (name);

  // build a nonlinear implicit system
  else if (sys_type == "NonlinearImplicit")
    this->add_system<NonLinearImplicitSystem> (name);

  else
  {
    std::cerr << "ERROR: Unknown system type: " << sys_type << std::endl;
  }

  // Return a reference to the new system
  return this->get_system(name);
}




void MultiLevelProblem::clear ()
{

  // clear the systems.  We must delete them since we newed them!
  clear_systems();
}



void MultiLevelProblem::clear_systems() {

  while (!_systems.empty())
  {
    system_iterator pos = _systems.begin();

    System *sys = pos->second;
    delete sys;
    sys = NULL;

    _systems.erase (pos);
  }
    
}




// void MultiLevelProblem::init()
// {
//   const unsigned int n_sys = this->n_systems();
//
//   assert(n_sys != 0);
//
//   for (unsigned int i=0; i != this->n_systems(); ++i)
//     this->get_system(i).init();
// }

// ===  Systems - END =================



// ===  Mesh, Solution - BEGIN =================

 void MultiLevelProblem::SetMultiLevelMeshAndSolution(MultiLevelSolution * ml_sol) {
     
				      _ml_sol = ml_sol;
				      _ml_msh = ml_sol->GetMLMesh();
				       _gridn = _ml_msh->GetNumberOfLevels();
     
 }
 
// ===  Mesh, Solution - END =================

 
 
// ===  Quadrature - BEGIN =================

 void MultiLevelProblem::SetQuadratureRuleAllGeomElems(const std::string quadr_order_in) {

     std::vector< std::string > qrule_list(1); 
     qrule_list[0] = quadr_order_in;
     
     SetQuadratureRuleAllGeomElemsMultiple(qrule_list);
     
     
//   _qrule.resize(1);
//   _qrule[0].reserve(N_GEOM_ELS);
//   
//   for (int iel = 0; iel < femus::geom_elems.size(); iel++) {
//           Gauss qrule_temp(femus::geom_elems[iel].c_str(),quadr_order_in.c_str());
//          _qrule[0].push_back(qrule_temp);
//            }

   return;
}


 void MultiLevelProblem::SetQuadratureRuleAllGeomElemsMultiple(const std::vector<std::string> quadr_order_in_vec) {

  _qrule.resize(quadr_order_in_vec.size());
  
  for (int q = 0; q < _qrule.size(); q++) {
  _qrule[q].reserve(N_GEOM_ELS);
  
  for (int iel = 0; iel < femus::geom_elems.size(); iel++) {
          Gauss qrule_temp(femus::geom_elems[iel].c_str(),quadr_order_in_vec[q].c_str());
         _qrule[q].push_back(qrule_temp);
           }
  }
  
   return;
}

// ===  Quadrature - END =================


} //end namespace femus


