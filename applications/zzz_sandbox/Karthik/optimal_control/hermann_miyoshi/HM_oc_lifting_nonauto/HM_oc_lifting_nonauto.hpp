#ifndef __femus_biharmonic_HM_oc_lifting_nonauto_hpp__
#define __femus_biharmonic_HM_oc_lifting_nonauto_hpp__

#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp"
#include "CurrentElem.hpp"
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

// =============================================================================
//  Regularization parameters (file scope, mirroring the distributed-control
//  nonauto convention).  Adjust here only.
// =============================================================================
static constexpr double alpha_0 = 0.00001;
static constexpr double alpha_1 = 0.00001;
static constexpr double alpha_2 = 0.00001;


namespace karthik {

  class biharmonic_HM_oc_lifting_nonauto {

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





// // // //========= BOUNDARY_IMPLEMENTATION_Sxy - BEGIN ==================
// // //
// // // static void natural_loop_1dS1(const MultiLevelProblem *    ml_prob,
// // //                      const Mesh *                    msh,
// // //                      const MultiLevelSolution *    ml_sol,
// // //                      const unsigned iel,
// // //                      CurrentElem < double > & geom_element,
// // //                      const unsigned xType,
// // //                      const std::string solname_sxy,
// // //                      const unsigned solFEType_sxy,
// // //                      std::vector< double > & Res
// // //                     ) {
// // //
// // //      double grad_sxy_dot_n = 0.;
// // //
// // //     for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {
// // //
// // //        geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);
// // //
// // //        geom_element.set_elem_center_bdry_3d();
// // //
// // //        std::vector <  double > xx_face_elem_center(3, 0.);
// // //           xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
// // //
// // //        const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);
// // //
// // //        if ( boundary_index < 0) { //I am on the boundary
// // //
// // //          unsigned int face = - (boundary_index + 1);
// // //
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxy.c_str(), grad_sxy_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet)  &&  (grad_sxy_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //                    unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxy);
// // //
// // //                   for (unsigned i = 0; i < n_dofs_face; i++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
// // //
// // //                  Res[i_vol] +=  grad_sxy_dot_n /* * phi[node] = 1. */;
// // //
// // //                          }
// // //
// // //                     }
// // //
// // //               }
// // //
// // //     }
// // //
// // // }
// // //
// // //
// // // template < class real_num, class real_num_mov >
// // // static void natural_loop_2d3dS1(const MultiLevelProblem *    ml_prob,
// // //                        const Mesh *                    msh,
// // //                        const MultiLevelSolution *    ml_sol,
// // //                        const unsigned iel,
// // //                        CurrentElem < double > & geom_element,
// // //                        const unsigned solType_coords,
// // //                        const std::string solname_sxy,
// // //                        const unsigned solFEType_sxy,
// // //                        std::vector< double > & Res,
// // //                        //-----------
// // //                        std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
// // //                        const unsigned dim,
// // //                        const unsigned space_dim,
// // //                        const unsigned max_size
// // //                     ) {
// // //
// // //
// // //     /// @todo - should put these outside the iel loop --
// // //     std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
// // //      std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
// // //     for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
// // //     for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }
// // //
// // //
// // //
// // //
// // //   double detJac_iqp_bdry;
// // //   double weight_iqp_bdry = 0.;
// // // // ---
// // //   //boundary state shape functions
// // //   std::vector <double> phi_sxy_bdry;
// // //   std::vector <double> phi_sxy_x_bdry;
// // //
// // //   phi_sxy_bdry.reserve(max_size);
// // //   phi_sxy_x_bdry.reserve(max_size * space_dim);
// // // // ---
// // //
// // // // ---
// // //   std::vector <double> phi_coords_bdry;
// // //   std::vector <double> phi_coords_x_bdry;
// // //
// // //   phi_coords_bdry.reserve(max_size);
// // //   phi_coords_x_bdry.reserve(max_size * space_dim);
// // // // ---
// // //
// // //
// // //
// // //      double grad_sxy_dot_n = 0.;
// // //
// // //     for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {
// // //
// // //        geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);
// // //
// // //        geom_element.set_elem_center_bdry_3d();
// // //
// // //        const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);
// // //
// // //
// // //        std::vector <  double > xx_face_elem_center(3, 0.);
// // //        xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
// // //
// // //        const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);
// // //
// // //        if ( boundary_index < 0) { //I am on the boundary
// // //
// // //          unsigned int face = - (boundary_index + 1);
// // //
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxy.c_str(), grad_sxy_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //     unsigned n_dofs_face_sxy = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxy);
// // //
// // // // dof-based - BEGIN
// // //      std::vector< double > grad_sxy_dot_n_at_dofs(n_dofs_face_sxy);
// // //
// // //
// // //     for (unsigned i_bdry = 0; i_bdry < grad_sxy_dot_n_at_dofs.size(); i_bdry++) {
// // //         std::vector<double> x_at_node(dim, 0.);
// // //         for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
// // //
// // //       double grad_sxy_dot_n_at_dofs_temp = 0.;
// // //       ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_sxy.c_str(), grad_sxy_dot_n_at_dofs_temp, face, 0.);
// // //      grad_sxy_dot_n_at_dofs[i_bdry] = grad_sxy_dot_n_at_dofs_temp;
// // //
// // //     }
// // //
// // // // dof-based - END
// // //
// // //
// // //                         const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();
// // //
// // //
// // // 		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {
// // //
// // //      elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
// // // //      elem_all[ielGeom_bdry][solType_coords]->compute_normal(Jac_iqp_bdry, normal);
// // //
// // //     weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];
// // //
// // //     elem_all[ielGeom_bdry][solFEType_sxy ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_sxy_bdry, phi_sxy_x_bdry,  boost::none, space_dim);
// // //
// // //
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //      elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);
// // //
// // //   std::vector<double> x_qp_bdry(dim, 0.);
// // //
// // //          for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
// // //            	for (unsigned d = 0; d < dim; d++) {
// // //  	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
// // //              }
// // //          }
// // //
// // //            double grad_sxy_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp
// // //
// // // // dof-based
// // //          for (unsigned i_bdry = 0; i_bdry < phi_sxy_bdry.size(); i_bdry ++) {
// // //            grad_sxy_dot_n_qp +=  grad_sxy_dot_n_at_dofs[i_bdry] * phi_sxy_bdry[i_bdry];
// // //          }
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //
// // //
// // //                   for (unsigned i_bdry = 0; i_bdry < n_dofs_face_sxy; i_bdry++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
// // //
// // //                  Res[i_vol] +=  weight_iqp_bdry * grad_sxy_dot_n_qp /*grad_u_dot_n*/  * phi_sxy_bdry[i_bdry];
// // //
// // //                            }
// // //
// // //
// // //                         }
// // //
// // //
// // //                     }
// // //
// // //               }
// // //     }
// // //
// // // }
// // //
// // //
// // // //========= BOUNDARY_IMPLEMENTATION_Sxy - END ==================






// // // //========= BOUNDARY_IMPLEMENTATION_S2 - BEGIN ==================
// // //
// // // static void natural_loop_1dS2(const MultiLevelProblem *    ml_prob,
// // //                      const Mesh *                    msh,
// // //                      const MultiLevelSolution *    ml_sol,
// // //                      const unsigned iel,
// // //                      CurrentElem < double > & geom_element,
// // //                      const unsigned xType,
// // //                      const std::string solname_syy,
// // //                      const unsigned solFEType_syy,
// // //                      std::vector< double > & Res
// // //                     ) {
// // //
// // //      double grad_syy_dot_n = 0.;
// // //
// // //     for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {
// // //
// // //        geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);
// // //
// // //        geom_element.set_elem_center_bdry_3d();
// // //
// // //        std::vector <  double > xx_face_elem_center(3, 0.);
// // //           xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
// // //
// // //        const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);
// // //
// // //        if ( boundary_index < 0) { //I am on the boundary
// // //
// // //          unsigned int face = - (boundary_index + 1);
// // //
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_syy.c_str(), grad_syy_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet)  &&  (grad_syy_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //
// // //
// // //                    unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_syy);
// // //
// // //                   for (unsigned i = 0; i < n_dofs_face; i++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
// // //
// // //                  Res[i_vol] +=  grad_syy_dot_n /* * phi[node] = 1. */;
// // //
// // //                          }
// // //
// // //                     }
// // //
// // //               }
// // //
// // //     }
// // //
// // // }
// // //
// // //
// // // template < class real_num, class real_num_mov >
// // // static void natural_loop_2d3dV2(const MultiLevelProblem *    ml_prob,
// // //                        const Mesh *                    msh,
// // //                        const MultiLevelSolution *    ml_sol,
// // //                        const unsigned iel,
// // //                        CurrentElem < double > & geom_element,
// // //                        const unsigned solType_coords,
// // //                        const std::string solname_syy,
// // //                        const unsigned solFEType_syy,
// // //                        std::vector< double > & Res,
// // //                        //-----------
// // //                        std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
// // //                        const unsigned dim,
// // //                        const unsigned space_dim,
// // //                        const unsigned max_size
// // //                     ) {
// // //
// // //
// // //     /// @todo - should put these outside the iel loop --
// // //     std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
// // //      std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
// // //     for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
// // //     for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }
// // //
// // //
// // //
// // //
// // //   double detJac_iqp_bdry;
// // //   double weight_iqp_bdry = 0.;
// // // // ---
// // //   //boundary state shape functions
// // //   std::vector <double> phi_syy_bdry;
// // //   std::vector <double> phi_syy_x_bdry;
// // //
// // //   phi_syy_bdry.reserve(max_size);
// // //   phi_syy_x_bdry.reserve(max_size * space_dim);
// // // // ---
// // //
// // // // ---
// // //   std::vector <double> phi_coords_bdry;
// // //   std::vector <double> phi_coords_x_bdry;
// // //
// // //   phi_coords_bdry.reserve(max_size);
// // //   phi_coords_x_bdry.reserve(max_size * space_dim);
// // // // ---
// // //
// // //
// // //
// // //      double grad_syy_dot_n = 0.;
// // //
// // //     for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {
// // //
// // //        geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);
// // //
// // //        geom_element.set_elem_center_bdry_3d();
// // //
// // //        const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);
// // //
// // //
// // //        std::vector <  double > xx_face_elem_center(3, 0.);
// // //        xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
// // //
// // //        const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);
// // //
// // //        if ( boundary_index < 0) { //I am on the boundary
// // //
// // //          unsigned int face = - (boundary_index + 1);
// // //
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_syy.c_str(), grad_syy_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //     unsigned n_dofs_face_syy = msh->GetElementFaceDofNumber(iel, jface, solFEType_syy);
// // //
// // // // dof-based - BEGIN
// // //      std::vector< double > grad_syy_dot_n_at_dofs(n_dofs_face_syy);
// // //
// // //
// // //     for (unsigned i_bdry = 0; i_bdry < grad_syy_dot_n_at_dofs.size(); i_bdry++) {
// // //         std::vector<double> x_at_node(dim, 0.);
// // //         for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
// // //
// // //       double grad_syy_dot_n_at_dofs_temp = 0.;
// // //       ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_syy.c_str(), grad_syy_dot_n_at_dofs_temp, face, 0.);
// // //      grad_syy_dot_n_at_dofs[i_bdry] = grad_syy_dot_n_at_dofs_temp;
// // //
// // //     }
// // //
// // // // dof-based - END
// // //
// // //
// // //                         const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();
// // //
// // //
// // // 		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {
// // //
// // //      elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
// // //
// // //     weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];
// // //
// // //     elem_all[ielGeom_bdry][solFEType_syy ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_syy_bdry, phi_syy_x_bdry,  boost::none, space_dim);
// // //
// // //
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //      elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);
// // //
// // //   std::vector<double> x_qp_bdry(dim, 0.);
// // //
// // //          for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
// // //            	for (unsigned d = 0; d < dim; d++) {
// // //  	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
// // //              }
// // //          }
// // //
// // //            double grad_syy_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp
// // //
// // // // dof-based
// // //          for (unsigned i_bdry = 0; i_bdry < phi_syy_bdry.size(); i_bdry ++) {
// // //            grad_syy_dot_n_qp +=  grad_syy_dot_n_at_dofs[i_bdry] * phi_syy_bdry[i_bdry];
// // //          }
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //
// // //
// // //                   for (unsigned i_bdry = 0; i_bdry < n_dofs_face_syy; i_bdry++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
// // //
// // //                  Res[i_vol] +=  weight_iqp_bdry * grad_syy_dot_n_qp /*grad_u_dot_n*/  * phi_syy_bdry[i_bdry];
// // //
// // //                            }
// // //
// // //
// // //                         }
// // //
// // //
// // //                     }
// // //
// // //               }
// // //     }
// // //
// // // }
// // //
// // //
// // // //========= BOUNDARY_IMPLEMENTATION_S2 - END ==================


template < class system_type, class real_num, class real_num_mov >
static void AssembleBilaplaceProblem_AD(
     const std::vector < std::vector < elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
     const std::vector < std::vector < elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
     const std::vector<Gauss> & quad_rules,
     system_type * mlPdeSys,
     MultiLevelMesh * ml_mesh_in,
     MultiLevelSolution * ml_sol_in,
     const std::vector< Unknown > &  unknowns,
     const std::vector< Math::Function< double > * > & source_functions) {

    // level is the level of the PDE system to be assembled
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh* msh = ml_mesh_in->GetLevel(level);
    MultiLevelSolution* ml_sol = ml_sol_in;
    Solution* sol = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    const unsigned iproc = msh->processor_id();

    RES->zero();
    if (assembleMatrix) KK->zero();


    constexpr unsigned int space_dim = 2; // max spatial dimension
    const unsigned int dim_offset_grad = dim;

    // Jacobian geometry containers
    std::vector< std::vector< real_num_mov > > JacI_qp(space_dim);
    std::vector< std::vector< real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) Jac_qp[d].resize(space_dim);
    for (unsigned d = 0; d < space_dim; d++) JacI_qp[d].resize(dim);

    real_num_mov detJac_qp = (real_num_mov)0.0;
    real_num_mov weight_qp = (real_num_mov)0.0;

    unsigned xType = CONTINUOUS_BIQUADRATIC; // geometry FE type

    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);

    // Unknowns setup - expect 12 unknowns
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 12) {
        std::cerr << "AssembleBilaplaceProblem: expected 12 unknowns but found " << n_unknowns << "\n";
        return;
    }

    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad));

    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    // element loop
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        // Geometry
        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // Unknown local DOFs
        for (unsigned u = 0; u < n_unknowns; u++) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // Prepare local mapping / storage
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs, (real_num)0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, (real_num)0.0);

        // Cache of per-unknown number of local dofs
        std::vector<unsigned> unk_num_elem_dofs(n_unknowns);
        for (unsigned u = 0; u < n_unknowns; u++) {
            unk_num_elem_dofs[u] = unknowns_local[u].num_elem_dofs();
        }

                // Local lengths for each unknown
        const unsigned nDofs_u = unk_num_elem_dofs[0];
        const unsigned nDofs_sxx = unk_num_elem_dofs[1];
        const unsigned nDofs_sxy = unk_num_elem_dofs[2];
        const unsigned nDofs_syy = unk_num_elem_dofs[3];
        const unsigned nDofs_ud = unk_num_elem_dofs[4];
        const unsigned nDofs_sxxd = unk_num_elem_dofs[5];
        const unsigned nDofs_sxyd = unk_num_elem_dofs[6];
        const unsigned nDofs_syyd = unk_num_elem_dofs[7];
        const unsigned nDofs_w = unk_num_elem_dofs[8];
        const unsigned nDofs_wsxxd = unk_num_elem_dofs[9];
        const unsigned nDofs_wsxyd = unk_num_elem_dofs[10];
        const unsigned nDofs_wsyyd = unk_num_elem_dofs[11];


        // Gauss loop
        const unsigned nGauss = quad_rules[ielGeom].GetGaussPointsNumber();
        for (unsigned ig = 0; ig < nGauss; ig++) {

            // Jacobian (element geometry)
            elem_all_for_domain[ielGeom][xType]->JacJacInv(geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            // Evaluate shape functions
            for (unsigned u = 0; u < n_unknowns; u++) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim
                );
            }

            // Geometry shape functions (for coordinate interpolation)
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim
            );



            // Source terms

            // NOTE: In the AD code, 'udr_term' was used in aRessyyd.
            // Also used assemble_function_for_rhs in 'aResu' as F_term,
            // but usually aResu is div(sigma)=0. Checking logic...
            // In AD code: aRessyyd += ... - udr_term.

            // Unpack Shape Functions for readability
            auto& phi_u = unknowns_phi_dof_qp[0].phi();
            auto& gradphi_u = unknowns_phi_dof_qp[0].phi_grad();

            auto& phi_sxx = unknowns_phi_dof_qp[1].phi();
            auto& gradphi_sxx = unknowns_phi_dof_qp[1].phi_grad();

            auto& phi_sxy = unknowns_phi_dof_qp[2].phi();
            auto& gradphi_sxy = unknowns_phi_dof_qp[2].phi_grad();

            auto& phi_syy = unknowns_phi_dof_qp[3].phi();
            auto& gradphi_syy = unknowns_phi_dof_qp[3].phi_grad();

            auto& phi_ud = unknowns_phi_dof_qp[4].phi();
            auto& gradphi_ud = unknowns_phi_dof_qp[4].phi_grad();

            auto& phi_sxxd = unknowns_phi_dof_qp[5].phi();
            auto& gradphi_sxxd = unknowns_phi_dof_qp[5].phi_grad();

            auto& phi_sxyd = unknowns_phi_dof_qp[6].phi();
            auto& gradphi_sxyd = unknowns_phi_dof_qp[6].phi_grad();

            auto& phi_syyd = unknowns_phi_dof_qp[7].phi();
            auto& gradphi_syyd = unknowns_phi_dof_qp[7].phi_grad();

            auto& phi_w = unknowns_phi_dof_qp[8].phi();
            auto& gradphi_w = unknowns_phi_dof_qp[8].phi_grad();

            auto& phi_wsxxd = unknowns_phi_dof_qp[9].phi();
            auto& gradphi_wsxxd = unknowns_phi_dof_qp[9].phi_grad();

            auto& phi_wsxyd = unknowns_phi_dof_qp[10].phi();
            auto& gradphi_wsxyd = unknowns_phi_dof_qp[10].phi_grad();

            auto& phi_wsyyd = unknowns_phi_dof_qp[11].phi();
            auto& gradphi_wsyyd = unknowns_phi_dof_qp[11].phi_grad();

            // Interpolate solution values & gradients at gauss point for each unknown
            real_num_mov u_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_u_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_u; ++a) {
                u_val_g += (real_num_mov) phi_u[a] * (real_num_mov) unknowns_local[0].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_u_g[d] += (real_num_mov) gradphi_u[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[0].elem_dofs()[a];
                }
            }

            real_num_mov sxx_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxx_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxx; ++a) {
                sxx_val_g += (real_num_mov) phi_sxx[a] * (real_num_mov) unknowns_local[1].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_sxx_g[d] += (real_num_mov) gradphi_sxx[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[1].elem_dofs()[a];
                }
            }

            real_num_mov sxy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxy; ++a) {
                sxy_val_g += (real_num_mov) phi_sxy[a] * (real_num_mov) unknowns_local[2].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_sxy_g[d] += (real_num_mov) gradphi_sxy[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[2].elem_dofs()[a];
                }
            }

            real_num_mov syy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_syy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_syy; ++a) {
                syy_val_g += (real_num_mov) phi_syy[a] * (real_num_mov) unknowns_local[3].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_syy_g[d] += (real_num_mov) gradphi_syy[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[3].elem_dofs()[a];
                }
            }

            real_num_mov ud_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_ud_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_ud; ++a) {
                ud_val_g += (real_num_mov) phi_ud[a] * (real_num_mov) unknowns_local[4].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_ud_g[d] += (real_num_mov) gradphi_ud[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[4].elem_dofs()[a];
                }
            }

            real_num_mov sxxd_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxxd_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxxd; ++a) {
                sxxd_val_g += (real_num_mov) phi_sxxd[a] * (real_num_mov) unknowns_local[5].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_sxxd_g[d] += (real_num_mov) gradphi_sxxd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[5].elem_dofs()[a];
                }
            }

            real_num_mov sxyd_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxyd_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxyd; ++a) {
                sxyd_val_g += (real_num_mov) phi_sxyd[a] * (real_num_mov) unknowns_local[6].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_sxyd_g[d] += (real_num_mov) gradphi_sxyd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[6].elem_dofs()[a];
                }
            }

            real_num_mov syyd_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_syyd_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_syyd; ++a) {
                syyd_val_g += (real_num_mov) phi_syyd[a] * (real_num_mov) unknowns_local[7].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_syyd_g[d] += (real_num_mov) gradphi_syyd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[7].elem_dofs()[a];
                }
            }

            real_num_mov w_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_w_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_w; ++a) {
                w_val_g += (real_num_mov) phi_w[a] * (real_num_mov) unknowns_local[8].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_w_g[d] += (real_num_mov) gradphi_w[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[8].elem_dofs()[a];
                }
            }


          real_num_mov wsxxd_val_g = (real_num_mov)0.0;
          std::vector< real_num_mov > grad_wsxxd_g(dim_offset_grad, (real_num_mov)0.0);
          for (unsigned a = 0; a < nDofs_wsxxd; ++a) {
              wsxxd_val_g += (real_num_mov) phi_wsxxd[a] * (real_num_mov) unknowns_local[9].elem_dofs()[a];
              for (unsigned d = 0; d < dim_offset_grad; ++d) {
                   grad_wsxxd_g[d] += (real_num_mov) gradphi_wsxxd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[9].elem_dofs()[a];
             }
         }

real_num_mov wsxyd_val_g = (real_num_mov)0.0;
std::vector< real_num_mov > grad_wsxyd_g(dim_offset_grad, (real_num_mov)0.0);
for (unsigned a = 0; a < nDofs_wsxyd; ++a) {
    wsxyd_val_g += (real_num_mov) phi_wsxyd[a] * (real_num_mov) unknowns_local[10].elem_dofs()[a];
    for (unsigned d = 0; d < dim_offset_grad; ++d) {
        grad_wsxyd_g[d] += (real_num_mov) gradphi_wsxyd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[10].elem_dofs()[a];
    }
}

real_num_mov wsyyd_val_g = (real_num_mov)0.0;
std::vector< real_num_mov > grad_wsyyd_g(dim_offset_grad, (real_num_mov)0.0);
for (unsigned a = 0; a < nDofs_wsyyd; ++a) {
    wsyyd_val_g += (real_num_mov) phi_wsyyd[a] * (real_num_mov) unknowns_local[11].elem_dofs()[a];
    for (unsigned d = 0; d < dim_offset_grad; ++d) {
        grad_wsyyd_g[d] += (real_num_mov) gradphi_wsyyd[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[11].elem_dofs()[a];
    }
}


            // Compute physical coordinates x_gss at gauss point
            std::vector< real_num_mov > x_gss(dim, (real_num_mov)0.0);
            auto & coords = geom_element.get_coords_at_dofs();
            const unsigned nGeomDofs = coords[0].size();
            for (unsigned a = 0; a < nGeomDofs; ++a) {
                const real_num_mov geom_phi = (real_num_mov) geom_element_phi_dof_qp.phi()[a];
                for (unsigned d = 0; d < dim; ++d) {
                    x_gss[d] += (real_num_mov) coords[d][a] * geom_phi;
                }
            }



            // Source/target u_D at the gauss point (returns the value from
            // source_function;  with Function_SmoothBump it is the bump 16(1/4-x^2)(1/4-y^2),
            // with Function_One it is 1, etc.).
            const real_num_mov f_val = (real_num_mov) source_functions[0]->value(x_gss);


            // ==================== RESIDUAL AND JACOBIAN CALCULATIONS ====================

// Test functions for R_u (first equation):  div(sigma + sigma_w) = 0   (no source: f = 0)
//   R_u = int  [ d_x phi_u_i * sxx_x  +  d_x phi_u_i * sxy_y  +  d_y phi_u_i * sxy_x
//              +  d_y phi_u_i * syy_y
//              +  d_x phi_u_i * wsxxd_x  +  d_x phi_u_i * wsxyd_y  +  d_y phi_u_i * wsxyd_x
//              +  d_y phi_u_i * wsyyd_y ] dOmega
for (unsigned i = 0; i < nDofs_u; ++i) {

    real_num_mov div_sigma = 0.0;
    if (dim == 2) {
        // ----- state sigma -----
        for (unsigned j = 0; j < nDofs_sxx; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxx[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[1].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_sxy; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[2].elem_dofs()[j];
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[2].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_syy; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syy[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[3].elem_dofs()[j];
        }
        // ----- control sigma_w (lifting contribution to div) -----
        for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[9].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[11].elem_dofs()[j];
        }
    }

    unk_element_jac_res.res()[i] += (real_num)(div_sigma * weight_qp);

    // Jacobian contributions for R_u
    // d/d sxx
    for (unsigned j = 0; j < nDofs_sxx; ++j) {
        real_num_mov jac_usxx = (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxx[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + j)] += (real_num)(jac_usxx * weight_qp);
    }
    // d/d sxy
    for (unsigned j = 0; j < nDofs_sxy; ++j) {
        real_num_mov jac_usxy = (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 1]
                              + (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + j)] += (real_num)(jac_usxy * weight_qp);
    }
    // d/d syy
    for (unsigned j = 0; j < nDofs_syy; ++j) {
        real_num_mov jac_usyy = (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syy[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + j)] += (real_num)(jac_usyy * weight_qp);
    }
    // d/d wsxxd
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov jac_uwsxxd = (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + j)] += (real_num)(jac_uwsxxd * weight_qp);
    }
    // d/d wsxyd
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov jac_uwsxyd = (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1]
                                + (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + j)] += (real_num)(jac_uwsxyd * weight_qp);
    }
    // d/d wsyyd
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov jac_uwsyyd = (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + j)] += (real_num)(jac_uwsyyd * weight_qp);
    }
}


// Test functions for R_sxx (second equation):  eps_xx(u) + s_xx = 0
//   R_sxx = int [ d_x phi_sxx_i * u_x  +  phi_sxx_i * sxx_h ] dOmega
for (unsigned i = 0; i < nDofs_sxx; ++i) {

    real_num_mov strain_xx = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_u; ++j) {
            strain_xx += (real_num_mov)gradphi_sxx[i * dim_offset_grad + 0] * (real_num_mov)gradphi_u[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
        }
    }

    real_num_mov mass_sxx = 0.0;
    for (unsigned j = 0; j < nDofs_sxx; ++j) {
        mass_sxx += (real_num_mov)phi_sxx[i] * (real_num_mov)phi_sxx[j] * (real_num_mov)unknowns_local[1].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + i] += (real_num)((strain_xx + mass_sxx) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_sxxu = (real_num_mov)gradphi_sxx[i * dim_offset_grad + 0] * (real_num_mov)gradphi_u[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + i) * total_local_dofs + j] += (real_num)(jac_sxxu * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_sxx; ++j) {
        real_num_mov jac_sxxsxx = (real_num_mov)phi_sxx[i] * (real_num_mov)phi_sxx[j];
        unk_element_jac_res.jac()[(nDofs_u + i) * total_local_dofs + (nDofs_u + j)] += (real_num)(jac_sxxsxx * weight_qp);
    }
}


// Test functions for R_sxy (third equation):  2 eps_xy(u) + s_xy = 0
//   R_sxy = int [ d_x phi_sxy_i * u_y  +  d_y phi_sxy_i * u_x  +  2 * phi_sxy_i * sxy_h ] dOmega
for (unsigned i = 0; i < nDofs_sxy; ++i) {

    real_num_mov strain_xy = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_u; ++j) {
            strain_xy += (real_num_mov)gradphi_sxy[i * dim_offset_grad + 0] * (real_num_mov)gradphi_u[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
            strain_xy += (real_num_mov)gradphi_sxy[i * dim_offset_grad + 1] * (real_num_mov)gradphi_u[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
        }
    }

    real_num_mov mass_sxy = 0.0;
    for (unsigned j = 0; j < nDofs_sxy; ++j) {
        mass_sxy += 2.0 * (real_num_mov)phi_sxy[i] * (real_num_mov)phi_sxy[j] * (real_num_mov)unknowns_local[2].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + i] += (real_num)((strain_xy + mass_sxy) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_sxyu = (real_num_mov)gradphi_sxy[i * dim_offset_grad + 0] * (real_num_mov)gradphi_u[j * dim_offset_grad + 1]
                              + (real_num_mov)gradphi_sxy[i * dim_offset_grad + 1] * (real_num_mov)gradphi_u[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + i) * total_local_dofs + j] += (real_num)(jac_sxyu * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_sxy; ++j) {
        real_num_mov jac_sxysxy = 2.0 * (real_num_mov)phi_sxy[i] * (real_num_mov)phi_sxy[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + i) * total_local_dofs + (nDofs_u + nDofs_sxx + j)] += (real_num)(jac_sxysxy * weight_qp);
    }
}


// Test functions for R_syy (fourth equation):  eps_yy(u) + s_yy = 0
//   R_syy = int [ d_y phi_syy_i * u_y  +  phi_syy_i * syy_h ] dOmega
for (unsigned i = 0; i < nDofs_syy; ++i) {

    real_num_mov strain_yy = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_u; ++j) {
            strain_yy += (real_num_mov)gradphi_syy[i * dim_offset_grad + 1] * (real_num_mov)gradphi_u[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
        }
    }

    real_num_mov mass_syy = 0.0;
    for (unsigned j = 0; j < nDofs_syy; ++j) {
        mass_syy += (real_num_mov)phi_syy[i] * (real_num_mov)phi_syy[j] * (real_num_mov)unknowns_local[3].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + i] += (real_num)((strain_yy + mass_syy) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_syyu = (real_num_mov)gradphi_syy[i * dim_offset_grad + 1] * (real_num_mov)gradphi_u[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + i) * total_local_dofs + j] += (real_num)(jac_syyu * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_syy; ++j) {
        real_num_mov jac_syysyy = (real_num_mov)phi_syy[i] * (real_num_mov)phi_syy[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + j)] += (real_num)(jac_syysyy * weight_qp);
    }
}


// Test functions for R_ud (fifth equation):  u + div(sigma_d) + w - u_D = 0
//   R_ud = int [ phi_ud_i * u_h
//              + d_x phi_ud_i * sxxd_x + d_x phi_ud_i * sxyd_y + d_y phi_ud_i * sxyd_x + d_y phi_ud_i * syyd_y
//              + phi_ud_i * w_h
//              - u_D * phi_ud_i ] dOmega
//   (Source u_D = f_val from source_functions[0], carried with MINUS sign in residual.)
for (unsigned i = 0; i < nDofs_ud; ++i) {

    real_num_mov mass_u = 0.0;
    for (unsigned j = 0; j < nDofs_u; ++j) {
        mass_u += (real_num_mov)phi_ud[i] * (real_num_mov)phi_u[j] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
    }

    real_num_mov div_sigmad = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_sxxd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[5].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_sxyd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_syyd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[7].elem_dofs()[j];
        }
    }

    real_num_mov mass_w = 0.0;
    for (unsigned j = 0; j < nDofs_w; ++j) {
        mass_w += (real_num_mov)phi_ud[i] * (real_num_mov)phi_w[j] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    real_num_mov source_term_ud = f_val * (real_num_mov)phi_ud[i];

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i] +=
        (real_num)((mass_u + div_sigmad + mass_w - source_term_ud) * weight_qp);

    // Jacobian contributions
    // d/d u
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_udu = (real_num_mov)phi_ud[i] * (real_num_mov)phi_u[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + j] += (real_num)(jac_udu * weight_qp);
    }
    // d/d sxxd
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov jac_udsxxd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + j)] += (real_num)(jac_udsxxd * weight_qp);
    }
    // d/d sxyd
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov jac_udsxyd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1]
                                + (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + j)] += (real_num)(jac_udsxyd * weight_qp);
    }
    // d/d syyd
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov jac_udsyyd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(jac_udsyyd * weight_qp);
    }
    // d/d w
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_udw = (real_num_mov)phi_ud[i] * (real_num_mov)phi_w[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_udw * weight_qp);
    }
}


// Test functions for R_sxxd (sixth equation):  eps_xx(ud) + s_xxd = 0
for (unsigned i = 0; i < nDofs_sxxd; ++i) {

    real_num_mov strain_xx_ud = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_ud; ++j) {
            strain_xx_ud += (real_num_mov)gradphi_sxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[4].elem_dofs()[j];
        }
    }

    real_num_mov mass_sxxd = 0.0;
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        mass_sxxd += (real_num_mov)phi_sxxd[i] * (real_num_mov)phi_sxxd[j] * (real_num_mov)unknowns_local[5].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + i] +=
        (real_num)((strain_xx_ud + mass_sxxd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_ud; ++j) {
        real_num_mov jac_sxxdud = (real_num_mov)gradphi_sxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + j)] += (real_num)(jac_sxxdud * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov jac_sxxdsxxd = (real_num_mov)phi_sxxd[i] * (real_num_mov)phi_sxxd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + j)] += (real_num)(jac_sxxdsxxd * weight_qp);
    }
}


// Test functions for R_sxyd (seventh equation):  2 eps_xy(ud) + s_xyd = 0
for (unsigned i = 0; i < nDofs_sxyd; ++i) {

    real_num_mov strain_xy_ud = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_ud; ++j) {
            strain_xy_ud += (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[4].elem_dofs()[j];
            strain_xy_ud += (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[4].elem_dofs()[j];
        }
    }

    real_num_mov mass_sxyd = 0.0;
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        mass_sxyd += 2.0 * (real_num_mov)phi_sxyd[i] * (real_num_mov)phi_sxyd[j] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + i] +=
        (real_num)((strain_xy_ud + mass_sxyd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_ud; ++j) {
        real_num_mov jac_sxydud = (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 1]
                                + (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + j)] += (real_num)(jac_sxydud * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov jac_sxydsxyd = 2.0 * (real_num_mov)phi_sxyd[i] * (real_num_mov)phi_sxyd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + j)] += (real_num)(jac_sxydsxyd * weight_qp);
    }
}


// Test functions for R_syyd (eighth equation):  eps_yy(ud) + s_yyd = 0
for (unsigned i = 0; i < nDofs_syyd; ++i) {

    real_num_mov strain_yy_ud = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_ud; ++j) {
            strain_yy_ud += (real_num_mov)gradphi_syyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[4].elem_dofs()[j];
        }
    }

    real_num_mov mass_syyd = 0.0;
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        mass_syyd += (real_num_mov)phi_syyd[i] * (real_num_mov)phi_syyd[j] * (real_num_mov)unknowns_local[7].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + i] +=
        (real_num)((strain_yy_ud + mass_syyd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_ud; ++j) {
        real_num_mov jac_syydud = (real_num_mov)gradphi_syyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + j)] += (real_num)(jac_syydud * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov jac_syydsyyd = (real_num_mov)phi_syyd[i] * (real_num_mov)phi_syyd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(jac_syydsyyd * weight_qp);
    }
}


// Test functions for R_w (ninth equation -- optimality):
//   u + w + alpha_0 w + alpha_1 (grad w, grad phi_w)
//   + div(sigma_d) - alpha_2 div(sigma_w) - u_D = 0
//
//  Discretized:
//   R_w = int [ phi_w_i * u_h
//             + (1 + alpha_0) * phi_w_i * w_h
//             + alpha_1 * (d_x phi_w_i * w_x + d_y phi_w_i * w_y)
//             + d_x phi_w_i * sxxd_x + d_x phi_w_i * sxyd_y + d_y phi_w_i * sxyd_x + d_y phi_w_i * syyd_y
//             - alpha_2 * ( d_x phi_w_i * wsxxd_x + d_x phi_w_i * wsxyd_y + d_y phi_w_i * wsxyd_x + d_y phi_w_i * wsyyd_y )
//             - u_D * phi_w_i ] dOmega
for (unsigned i = 0; i < nDofs_w; ++i) {

    // mass_u_w = (u, phi_w)
    real_num_mov mass_u_w = 0.0;
    for (unsigned j = 0; j < nDofs_u; ++j) {
        mass_u_w += (real_num_mov)phi_w[i] * (real_num_mov)phi_u[j] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
    }

    // div(sigma_d)
    real_num_mov div_sigmad_w = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_sxxd; ++j) {
            div_sigmad_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[5].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_sxyd; ++j) {
            div_sigmad_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
            div_sigmad_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_syyd; ++j) {
            div_sigmad_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[7].elem_dofs()[j];
        }
    }

    // (1 + alpha_0) (w, phi_w)
    real_num_mov mass_w_w = 0.0;
    for (unsigned j = 0; j < nDofs_w; ++j) {
        mass_w_w += (1.0 + alpha_0) * (real_num_mov)phi_w[i] * (real_num_mov)phi_w[j] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    // alpha_1 (grad w, grad phi_w)
    real_num_mov stiff_w_w = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_w; ++j) {
            stiff_w_w += alpha_1 * (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
            stiff_w_w += alpha_1 * (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
        }
    }

    // -alpha_2 * div(sigma_w)
    real_num_mov m_a2_div_sigmaw = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
            m_a2_div_sigmaw += - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[9].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
            m_a2_div_sigmaw += - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
            m_a2_div_sigmaw += - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
        }
        for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
            m_a2_div_sigmaw += - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[11].elem_dofs()[j];
        }
    }

    // Source u_D
    real_num_mov source_term_w = f_val * (real_num_mov)phi_w[i];

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i] +=
        (real_num)((mass_u_w + div_sigmad_w + mass_w_w + stiff_w_w + m_a2_div_sigmaw - source_term_w) * weight_qp);

    // Jacobian contributions
    // d/d u
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_wu = (real_num_mov)phi_w[i] * (real_num_mov)phi_u[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + j] += (real_num)(jac_wu * weight_qp);
    }
    // d/d sxxd
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov jac_wsxxd = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + j)] += (real_num)(jac_wsxxd * weight_qp);
    }
    // d/d sxyd
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov jac_wsxyd = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1]
                               + (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + j)] += (real_num)(jac_wsxyd * weight_qp);
    }
    // d/d syyd
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov jac_wsyyd = (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(jac_wsyyd * weight_qp);
    }
    // d/d w  :  (1 + alpha_0) * mass + alpha_1 * stiffness
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_ww = (1.0 + alpha_0) * (real_num_mov)phi_w[i] * (real_num_mov)phi_w[j]
                            + alpha_1 * ( (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0]
                                        + (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] );
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_ww * weight_qp);
    }
    // d/d wsxxd  :  -alpha_2 * d_x phi_w_i * d_x phi_wsxxd_j
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov jac_wwsxxd = - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + j)] += (real_num)(jac_wwsxxd * weight_qp);
    }
    // d/d wsxyd
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov jac_wwsxyd = - alpha_2 * ( (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1]
                                              + (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 0] );
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + j)] += (real_num)(jac_wwsxyd * weight_qp);
    }
    // d/d wsyyd
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov jac_wwsyyd = - alpha_2 * (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + j)] += (real_num)(jac_wwsyyd * weight_qp);
    }
}


// Test functions for R_wsxxd (tenth equation):  eps_xx(w) + s_w,xx = 0
for (unsigned i = 0; i < nDofs_wsxxd; ++i) {

    real_num_mov strain_xx_w = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_w; ++j) {
            strain_xx_w += (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
        }
    }

    real_num_mov mass_wsxxd = 0.0;
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        mass_wsxxd += (real_num_mov)phi_wsxxd[i] * (real_num_mov)phi_wsxxd[j] * (real_num_mov)unknowns_local[9].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + i] +=
        (real_num)((strain_xx_w + mass_wsxxd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_wsxxdw = (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_wsxxdw * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov jac_wsxxdwsxxd = (real_num_mov)phi_wsxxd[i] * (real_num_mov)phi_wsxxd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + j)] += (real_num)(jac_wsxxdwsxxd * weight_qp);
    }
}


// Test functions for R_wsxyd (eleventh equation):  2 eps_xy(w) + s_w,xy = 0
for (unsigned i = 0; i < nDofs_wsxyd; ++i) {

    real_num_mov strain_xy_w = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_w; ++j) {
            strain_xy_w += (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
            strain_xy_w += (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
        }
    }

    real_num_mov mass_wsxyd = 0.0;
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        mass_wsxyd += 2.0 * (real_num_mov)phi_wsxyd[i] * (real_num_mov)phi_wsxyd[j] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + i] +=
        (real_num)((strain_xy_w + mass_wsxyd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_wsxydw = (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1]
                                + (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_wsxydw * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov jac_wsxydwsxyd = 2.0 * (real_num_mov)phi_wsxyd[i] * (real_num_mov)phi_wsxyd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + j)] += (real_num)(jac_wsxydwsxyd * weight_qp);
    }
}


// Test functions for R_wsyyd (twelfth equation):  eps_yy(w) + s_w,yy = 0
for (unsigned i = 0; i < nDofs_wsyyd; ++i) {

    real_num_mov strain_yy_w = 0.0;
    if (dim == 2) {
        for (unsigned j = 0; j < nDofs_w; ++j) {
            strain_yy_w += (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
        }
    }

    real_num_mov mass_wsyyd = 0.0;
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        mass_wsyyd += (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_wsyyd[j] * (real_num_mov)unknowns_local[11].elem_dofs()[j];
    }

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + i] +=
        (real_num)((strain_yy_w + mass_wsyyd) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_wsyydw = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_wsyydw * weight_qp);
    }
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov jac_wsyydwsyyd = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_wsyyd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + j)] += (real_num)(jac_wsyydwsyyd * weight_qp);
    }
}
        } // end gauss loop

        // Finalize local residual (FEMUS convention negative)
        std::vector<double> Res_total(unk_element_jac_res.res().size());
        for (size_t kk = 0; kk < unk_element_jac_res.res().size(); ++kk) {
            Res_total[kk] = -(unk_element_jac_res.res()[kk]);
        }

        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());

        if (assembleMatrix) {
            KK->add_matrix_blocked(unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map());
        }

        // Optional printing
        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = unk_num_elem_dofs;
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }

    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
}

  };

}

#endif
