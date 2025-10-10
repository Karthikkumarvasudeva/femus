#ifndef __femus_biharmonic_HM_D_hpp__
#define __femus_biharmonic_HM_D_hpp__
 
#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
/**
 * Given the non linear problem
 *
 *      \Delta^2 u  = f(x),
 *      u(\Gamma) = 0
 *      \Delta u(\Gamma) = 0
 *
 * in the unit box \Omega centered in the origin with boundary \Gamma, where
 *
 *                      f(x) = \Delta^2 u_e ,
 *                    u_e = \cos ( \pi * x ) * \cos( \pi * y ),
 *
 * the following function assembles the system:
 *
 *      \Delta u = v
 *      \Delta v = f(x) = 4. \pi^4 u_e
 *      u(\Gamma) = 0
 *      v(\Gamma) = 0
 *
 * using automatic differentiation
 **/

using namespace femus;


namespace karthik {
  
  class biharmonic_HM {
    
  public:




//========= BOUNDARY_IMPLEMENTATION_U - BEGIN ==================

static void natural_loop_1dU(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_u,
                     const unsigned solFEType_u,
                     std::vector< double > & Res
                    ) {

     double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_u_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann

                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_u_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dU(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_u,
                       const unsigned solFEType_u,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_u_bdry;
  std::vector <double> phi_u_x_bdry;

  phi_u_bdry.reserve(max_size);
  phi_u_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_u = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);

// dof-based - BEGIN
     std::vector< double > grad_u_dot_n_at_dofs(n_dofs_face_u);


    for (unsigned i_bdry = 0; i_bdry < grad_u_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_u_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_u.c_str(), grad_u_dot_n_at_dofs_temp, face, 0.);
     grad_u_dot_n_at_dofs[i_bdry] = grad_u_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
//      elem_all[ielGeom_bdry][solType_coords]->compute_normal(Jac_iqp_bdry, normal);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_u ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_u_bdry, phi_u_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_u_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_u_bdry.size(); i_bdry ++) {
           grad_u_dot_n_qp +=  grad_u_dot_n_at_dofs[i_bdry] * phi_u_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_u; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_u_dot_n_qp /*grad_u_dot_n*/  * phi_u_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_U - END ==================


//========= BOUNDARY_IMPLEMENTATION_Sxx - BEGIN ==================

static void natural_loop_1dSxx(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_sxx,
                     const unsigned solFEType_sxx,
                     std::vector< double > & Res
                    ) {

     double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_sxx_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_sxx_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dSxx(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_sxx,
                       const unsigned solFEType_sxx,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_sxx_bdry;
  std::vector <double> phi_sxx_x_bdry;

  phi_sxx_bdry.reserve(max_size);
  phi_sxx_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_sxx = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);

// dof-based - BEGIN
     std::vector< double > grad_sxx_dot_n_at_dofs(n_dofs_face_sxx);


    for (unsigned i_bdry = 0; i_bdry < grad_sxx_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_sxx_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_sxx.c_str(), grad_sxx_dot_n_at_dofs_temp, face, 0.);
     grad_sxx_dot_n_at_dofs[i_bdry] = grad_sxx_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_sxx ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_sxx_bdry, phi_sxx_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_sxx_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_sxx_bdry.size(); i_bdry ++) {
           grad_sxx_dot_n_qp +=  grad_sxx_dot_n_at_dofs[i_bdry] * phi_sxx_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_sxx; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_sxx_dot_n_qp /*grad_u_dot_n*/  * phi_sxx_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_Sxx - END ==================



//========= BOUNDARY_IMPLEMENTATION_Sxy - BEGIN ==================

static void natural_loop_1dSxy(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_sxy,
                     const unsigned solFEType_sxy,
                     std::vector< double > & Res
                    ) {

     double grad_sxy_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxy.c_str(), grad_sxy_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_sxy_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxy);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_sxy_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dSxy(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_sxy,
                       const unsigned solFEType_sxy,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_sxy_bdry;
  std::vector <double> phi_sxy_x_bdry;

  phi_sxy_bdry.reserve(max_size);
  phi_sxy_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_sxy_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxy.c_str(), grad_sxy_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_sxy = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxy);

// dof-based - BEGIN
     std::vector< double > grad_sxy_dot_n_at_dofs(n_dofs_face_sxy);


    for (unsigned i_bdry = 0; i_bdry < grad_sxy_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_sxy_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_sxy.c_str(), grad_sxy_dot_n_at_dofs_temp, face, 0.);
     grad_sxy_dot_n_at_dofs[i_bdry] = grad_sxy_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_sxy ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_sxy_bdry, phi_sxy_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_sxy_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_sxy_bdry.size(); i_bdry ++) {
           grad_sxy_dot_n_qp +=  grad_sxy_dot_n_at_dofs[i_bdry] * phi_sxy_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_sxy; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_sxy_dot_n_qp /*grad_u_dot_n*/  * phi_sxy_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_Sxy - END ==================



//========= BOUNDARY_IMPLEMENTATION_Syy - BEGIN ==================

static void natural_loop_1dSyy(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_syy,
                     const unsigned solFEType_syy,
                     std::vector< double > & Res
                    ) {

     double grad_syy_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_syy.c_str(), grad_syy_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_syy_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_syy);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_syy_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dSyy(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_syy,
                       const unsigned solFEType_syy,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_syy_bdry;
  std::vector <double> phi_syy_x_bdry;

  phi_syy_bdry.reserve(max_size);
  phi_syy_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_syy_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_syy.c_str(), grad_syy_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_syy = msh->GetElementFaceDofNumber(iel, jface, solFEType_syy);

// dof-based - BEGIN
     std::vector< double > grad_syy_dot_n_at_dofs(n_dofs_face_syy);


    for (unsigned i_bdry = 0; i_bdry < grad_syy_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_syy_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_syy.c_str(), grad_syy_dot_n_at_dofs_temp, face, 0.);
     grad_syy_dot_n_at_dofs[i_bdry] = grad_syy_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_syy ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_syy_bdry, phi_syy_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_syy_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_syy_bdry.size(); i_bdry ++) {
           grad_syy_dot_n_qp +=  grad_syy_dot_n_at_dofs[i_bdry] * phi_syy_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_syy; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_syy_dot_n_qp /*grad_u_dot_n*/  * phi_syy_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_Syy - END ==================






static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use

  LinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<LinearImplicitSystem> (ml_prob.get_app_specs_pointer()->_system_name);

  const unsigned level = mlPdeSys->GetLevelToAssemble();


  // Geometry, Global - BEGIN
  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  // Geometry, Global - END


  // Solution, Global - BEGIN
    MultiLevelSolution*  ml_sol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
    Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object
  // Solution, Global - END


  // Equation, Global - BEGIN
  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  SparseMatrix*             JAC = pdeSys->_KK;

  RES->zero();
  KK->zero(); // Set to zero all the entries of the Global Matrix
  // Equation, Global - END


  // Solution, Global, u - BEGIN
  const std::string solname_u = ml_sol->GetSolName_string_vec()[0];
  const unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());    // get the position of "u" in the ml_sol object
  const unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);    // get the finite element type for "u"
  const unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());    // get the position of "u" in the pdeSys object

  // Solution, Global, u - END


  // Solution, Global, sxx - BEGIN
  const std::string solname_sxx = ml_sol->GetSolName_string_vec()[1];
  const unsigned solsxxIndex = ml_sol->GetIndex(solname_sxx.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_sxx = ml_sol->GetSolutionType(solsxxIndex);    // get the finite element type for "v"
  const unsigned solsxxPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxx.c_str());    // get the position of "v" in the pdeSys object

  // Solution, Global, sxx - END

  // Solution, Global, sxy - BEGIN
  const std::string solname_sxy = ml_sol->GetSolName_string_vec()[2];
  const unsigned solsxyIndex = ml_sol->GetIndex(solname_sxy.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_sxy = ml_sol->GetSolutionType(solsxyIndex);    // get the finite element type for "v"
  const unsigned solsxyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxy.c_str());    // get the position of "v" in the pdeSys object

  // Solution, Global, sxy - END

  // Solution, Global, syy - BEGIN
  const std::string solname_syy = ml_sol->GetSolName_string_vec()[3];
  const unsigned solsyyIndex = ml_sol->GetIndex(solname_syy.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_syy = ml_sol->GetSolutionType(solsyyIndex);    // get the finite element type for "v"
  const unsigned solsyyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_syy.c_str());    // get the position of "v" in the pdeSys object
  // Solution, Global, syy - END



  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27


  // Geometry, Local - BEGIN
  std::vector < std::vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  for (unsigned i = 0; i < dim; i++)    x[i].reserve(maxSize);
  // Geometry, Local - END


  // Solution, Local - BEGIN
  std::vector < adept::adouble >  solu; // local solution
  std::vector < adept::adouble >  solsxx; // local solution
  std::vector < adept::adouble >  solsxy; // local solution
  std::vector < adept::adouble >  solsyy; // local solution
  // reserve memory for the local standar vectors
  solu.reserve(maxSize);
  solsxx.reserve(maxSize);

  solsxy.reserve(maxSize);
  solsyy.reserve(maxSize);
  // Solution, Local - END

  // Solution, Local, phi - BEGIN
  std::vector <double> phi;  // local test function
  std::vector <double> phi_x; // local test function first order partial derivatives
  std::vector <double> phi_xx; // local test function second order partial derivatives

  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);
// // //   unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
    unsigned dim2 = (6 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)

  phi_xx.reserve(maxSize * dim2);

  // Solution, Local, phi - END

  // Equation, Local - BEGIN
  std::vector < int > sysDof; // local to global pdeSys dofs
  sysDof.reserve(4 * maxSize);

  std::vector < double > Res; // local redidual vector

  std::vector < adept::adouble > aResu;
  std::vector < adept::adouble > aRessxx;
  std::vector < adept::adouble > aRessxy;
  std::vector < adept::adouble > aRessyy;

  Res.reserve(4 * maxSize);

  aResu.reserve(maxSize);
  aRessxx.reserve(maxSize);
  aRessxy.reserve(maxSize);
  aRessyy.reserve(maxSize);

  std::vector < double > Jac; // local Jacobian matrix (ordered by column, adept)
  Jac.reserve(4 * maxSize * 4 * maxSize);
  // Equation, Local - END

  double weight; // gauss point weight


double nu =  0.1 /* Poisson ratio value */;
double nu1 = (4.0 * (1.0 - nu)) / (1.0 + nu);
double nu2 = 2.0 / (1.0 + nu);
// // // double nu2 = 1. - nu;



  for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel); 

// // //     unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_u);    // number of solution element dofs
    unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_sxx);    // number of solution element dofs

    unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs

    std::vector<unsigned> Sol_n_el_dofs_Mat_vol(4, nDofs);

    // resize local arrays
    sysDof.resize(4 * nDofs);

    solu.resize(nDofs);
    solsxx.resize(nDofs);
    solsxy.resize(nDofs);
    solsyy.resize(nDofs);

    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofs2);
    }

    aResu.assign(nDofs, 0.);
    aRessxx.assign(nDofs, 0.);
    aRessxy.assign(nDofs, 0.);
    aRessyy.assign(nDofs, 0.);

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {

// // //       unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);    // global to global mapping between solution node and solution dof
      unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_sxx);    // global to global mapping between solution node and solution dof



      solu[i]          = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
      solsxx[i]        = (*sol->_Sol[solsxxIndex])(solDof);      // global extraction and local storage for the solution
      solsxy[i]        = (*sol->_Sol[solsxyIndex])(solDof);      // sxy  -> secondary row2, col2
      solsyy[i]        = (*sol->_Sol[solsyyIndex])(solDof);      // syy  -> secondary row1, col2

      sysDof[i]             = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[nDofs + i]     = pdeSys->GetSystemDof(solsxxIndex, solsxxPdeIndex, i, iel);
      sysDof[2 * nDofs + i] = pdeSys->GetSystemDof(solsxyIndex, solsxyPdeIndex, i, iel); // sxy
      sysDof[3 * nDofs + i] = pdeSys->GetSystemDof(solsyyIndex, solsyyPdeIndex, i, iel); // syy

    }

    // local storage of coordinates
    for (unsigned i = 0; i < nDofs2; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType); // global to global mapping between coordinates node and coordinate dof

      for (unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);  // global extraction and local storage for the element coordinates
      }
    }

    // start a new recording of all the operations involving adept::adouble variables
    s.new_recording();

    // *** Gauss point loop ***

    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
// *** get gauss point weight, test function and test function partial derivatives ***

      msh->_finiteElement[ielGeom][solFEType_sxx]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

      std::vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        for (unsigned jdim = 0; jdim < dim; jdim++) {
          xGauss[jdim] += x[jdim][i] * phi[i];
        }
      }

      // evaluate the solution, the solution derivatives in the gauss point
      adept::adouble soluGauss = 0.;
      std::vector < adept::adouble > soluGauss_x(dim, 0.);

      adept::adouble solsxxGauss = 0.;
      std::vector < adept::adouble > solsxxGauss_x(dim, 0.);

      adept::adouble solsxyGauss = 0.;
      std::vector < adept::adouble > solsxyGauss_x(dim, 0.);

      adept::adouble solsyyGauss = 0.;
      std::vector < adept::adouble > solsyyGauss_x(dim, 0.);


      for (unsigned i = 0; i < nDofs; i++) {
        soluGauss += phi[i] * solu[i];
        solsxxGauss += phi[i] * solsxx[i];

        solsxyGauss += phi[i] * solsxy[i];
        solsyyGauss += phi[i] * solsyy[i];

        for (unsigned jdim = 0; jdim < dim; jdim++) {
          soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
          solsxxGauss_x[jdim] += phi_x[i * dim + jdim] * solsxx[i];

          solsxyGauss_x[jdim] += phi_x[i * dim + jdim] * solsxy[i];
          solsyyGauss_x[jdim] += phi_x[i * dim + jdim] * solsyy[i];

        }
      }
      // *** phi_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {

        adept::adouble Laplace_u = 0.;
        adept::adouble Laplace_sxx = 0.;

        adept::adouble Laplace_sxy = 0.;
        adept::adouble Laplace_syy = 0.;


        adept::adouble Mxxxx_sxx = phi[i] * solsxxGauss;
        adept::adouble Mxyxy_sxy = 2. * phi[i] * solsxyGauss;
        adept::adouble Myyyy_syy = phi[i] * solsyyGauss;


        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * soluGauss_x[jdim];
          Laplace_sxx   +=  - phi_x[i * dim + jdim] * solsxxGauss_x[jdim];

          Laplace_sxy   +=  - phi_x[i * dim + jdim] * solsxyGauss_x[jdim];
          Laplace_syy   +=  - phi_x[i * dim + jdim] * solsyyGauss_x[jdim];

        }

        double pi = acos(-1.);

    adept::adouble Bxxu = 0.;
    adept::adouble Bxyu = 0.;
    adept::adouble Byyu = 0.;
    adept::adouble Bxxsxx = 0.;
    adept::adouble Bxysxy = 0.;
    adept::adouble Byysyy = 0.;

// // //     if (dim == 2) {

        Bxxu += phi_x[i * dim] * soluGauss_x[0];
        Bxyu +=   phi_x[i * dim + 1] * soluGauss_x[0] + phi_x[i * dim ] * soluGauss_x[1] ;
        Byyu +=  phi_x[i * dim + 1] * soluGauss_x[1];

        Bxxsxx += phi_x[i * dim] * solsxxGauss_x[0];
        Bxysxy +=( (phi_x[i * dim + 1 ] * solsxyGauss_x[0] + phi_x[i * dim ] * solsxyGauss_x[1]) );
        Byysyy +=  phi_x[i * dim + 1] * solsyyGauss_x[1];


// // //     }
        adept::adouble F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

        // System residuals - signs adjusted to match matrix form
     aResu[i] += ( Bxxsxx + Bxysxy + Byysyy + F_term ) * weight;
     aRessxx[i] += (Bxxu + Mxxxx_sxx ) * weight;
     aRessxy[i] += ( Bxyu + Mxyxy_sxy ) * weight;
     aRessyy[i] += (Byyu + Myyyy_syy) * weight;

      } // end phi_i loop

    } // end gauss point loop

    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store

   Res.resize(4 * nDofs, 0.);

    for (int i = 0; i < nDofs; i++) {
      Res[i]               = - aResu[i].value();
      Res[nDofs + i]       = - aRessxx[i].value();

      Res[2 * nDofs + i  ] = - aRessxy[i].value(); // sxy
      Res[3 * nDofs + i  ] = - aRessyy[i].value(); // syy

    }

    RES->add_vector_blocked(Res, sysDof);

    Jac.resize(16 * nDofs * nDofs);

    // define the independent variables
    s.independent(&solu[0], nDofs);
    s.independent(&solsxx[0], nDofs);

    s.independent(&solsxy[0], nDofs);
    s.independent(&solsyy[0], nDofs);

        // define the dependent variables
    s.dependent(&aResu[0], nDofs);
    s.dependent(&aRessxx[0], nDofs);
    s.dependent(&aRessxy[0], nDofs);
    s.dependent(&aRessyy[0], nDofs);

    // get the jacobian matrix (ordered by column)
    s.jacobian(&Jac[0], true);

    KK->add_matrix_blocked(Jac, sysDof, sysDof);

         constexpr bool print_algebra_local = false;
     if (print_algebra_local) {

         assemble_jacobian<double,double>::print_element_jacobian(iel, Jac, Sol_n_el_dofs_Mat_vol, 10, 5);
         assemble_jacobian<double,double>::print_element_residual(iel, Res, Sol_n_el_dofs_Mat_vol, 10, 5);

     }


    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();


}

  };
  
}

#endif
