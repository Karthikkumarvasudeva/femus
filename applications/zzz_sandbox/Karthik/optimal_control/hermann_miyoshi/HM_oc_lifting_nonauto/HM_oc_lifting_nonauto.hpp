#ifndef __femus_biharmonic_HM_hpp__
#define __femus_biharmonic_HM_hpp__

#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp" // <-- ADD THIS LINE to fix ElementJacRes errors
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
  // // // const std::vector< Math::Function< double > * > & analytical_u;
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

    // Physics Constants
    const double beta = 0.0001;
    const double gamma = 0.01;
    // const double nu = 0.0; // Poisson ratio unused in provided AD code

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



            // ==================== RESIDUALS & JACOBIANS ====================



            const real_num_mov f_val = (real_num_mov) source_functions[0]->value(x_gss);


 // // // const real_num_mov f_u_val = (real_num_mov) source_functions[0]->laplacian(x_gss); // udr_term
            // --- 1. Equation for u (aResu) ---
            // AD: Bxx + Bxy + Byy + Bwxxud + Bwxyud + Bwyyud
            for (unsigned i = 0; i < nDofs_u; ++i) {
                // Residual
                real_num_mov div_sigma = 0.0;
                real_num_mov div_sigma_w = 0.0;

                // div(sigma): (grad_v, grad_sigma) pattern from AD code Bxx/Bxy/Byy
                for (unsigned j = 0; j < nDofs_sxx; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxx[j * dim_offset_grad] * (real_num_mov)unknowns_local[1].elem_dofs()[j];
        }

        // ∂σ_xy/∂y term
        for (unsigned j = 0; j < nDofs_sxy; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[2].elem_dofs()[j];
        }
        // ∂σ_xy/∂x term
        for (unsigned j = 0; j < nDofs_sxy; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[2].elem_dofs()[j];
        }
        // ∂σ_yy/∂y term
        for (unsigned j = 0; j < nDofs_syy; ++j) {
            div_sigma += (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syy[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[3].elem_dofs()[j];
        }
                // div(sigma_w) aka Bw...ud
                for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
            div_sigma_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j * dim_offset_grad] * (real_num_mov)unknowns_local[9].elem_dofs()[j];
        }

        // ∂σ_xy/∂y term
        for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
            div_sigma_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
        }
        // ∂σ_xy/∂x term
        for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
            div_sigma_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[10].elem_dofs()[j];
        }
        // ∂σ_yy/∂y term
        for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
            div_sigma_w += (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[11].elem_dofs()[j];
        }

                unk_element_jac_res.res()[i] += (div_sigma + div_sigma_w /*+ f_u_val*/) * weight_qp;

                // Jacobian blocks - Fixed indexing issues
    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_sxx; ++j) {
        real_num_mov jac_usxx = (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxx[j*dim_offset_grad];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + j)] += (real_num)(jac_usxx * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_sxy; ++j) {
        real_num_mov jac_usxy = (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov) gradphi_sxy [j * dim_offset_grad + 0] + (real_num_mov)gradphi_u[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxy[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + j)] += (real_num)(jac_usxy * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_syy; ++j) {
        real_num_mov jac_usyy = (real_num_mov)gradphi_u[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syy[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + j)] += (real_num)(jac_usyy * weight_qp);
    }


        for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov jac_wsxx = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxxd[j*dim_offset_grad];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + j)] += (real_num)(jac_wsxx * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov jac_wsxy = (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov) gradphi_wsxyd [j * dim_offset_grad + 0] + (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_wsxyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + j)] += (real_num)(jac_wsxy * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov jac_wsyy = (real_num_mov)gradphi_w[i * dim_offset_grad + 1] * (real_num_mov)gradphi_wsyyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd + j)] += (real_num)(jac_wsyy * weight_qp);
    }
            }

            // --- 2. Equation for sxx (aRessxx) ---
            // AD: Bxxu + M_sxx

            // Test functions for R_sxx (second equation): ε_xx(u) + s_xx = 0
for (unsigned i = 0; i < nDofs_sxx; ++i) {
    // R_sxx = ∫(ε_xx(u) + s_xx)·v_sxx dΩ
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

// Test functions for R_sxy (third equation): 2ε_xy(u) + s_xy = 0
for (unsigned i = 0; i < nDofs_sxy; ++i) {
    // R_sxy = ∫(2ε_xy(u) + s_xy)·v_sxy dΩ
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
        real_num_mov jac_sxyu = (real_num_mov)gradphi_sxy[i * dim_offset_grad + 0] * (real_num_mov)gradphi_u[j * dim_offset_grad + 1] +
                               (real_num_mov)gradphi_sxy[i * dim_offset_grad + 1] * (real_num_mov)gradphi_u[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + i) * total_local_dofs + j] += (real_num)(jac_sxyu * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_sxy; ++j) {
        real_num_mov jac_sxysxy = 2.0 * (real_num_mov)phi_sxy[i] * (real_num_mov)phi_sxy[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + i) * total_local_dofs + (nDofs_u + nDofs_sxx + j)] += (real_num)(jac_sxysxy * weight_qp);
    }
}

            // --- 4. Equation for syy (aRessyy) ---
// Test functions for R_syy (fourth equation): ε_yy(u) + s_yy = 0
for (unsigned i = 0; i < nDofs_syy; ++i) {
    // R_syy = ∫(ε_yy(u) + s_yy)·v_syy dΩ
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
//                         const real_num_mov u_rhs = (real_num_mov) source_function[0]->value(x_gss); // udr_term
// --- 5. Equation for ud ---
for (unsigned i = 0; i < nDofs_ud; ++i) {
    // R_ud = ∫(u + ∂σ_xxd/∂x + ∂σ_xyd/∂y + ∂σ_xyd/∂x + ∂σ_yyd/∂y + f)·v_ud dΩ
    real_num_mov mass_u = 0.0;
    for (unsigned j = 0; j < nDofs_u; ++j) {
        mass_u += (real_num_mov)phi_ud[i] * (real_num_mov)phi_u[j] * (real_num_mov)unknowns_local[0].elem_dofs()[j];
    }

        real_num_mov mass_w = 0.0;
    for (unsigned j = 0; j < nDofs_w; ++j) {
        mass_w += (real_num_mov)phi_ud[i] * (real_num_mov)phi_w[j] * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    real_num_mov div_sigmad = 0.0;
    if (dim == 2) {
        // ∂σ_xxd/∂x term
        for (unsigned j = 0; j < nDofs_sxxd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad] * (real_num_mov)unknowns_local[5].elem_dofs()[j];
        }
        // ∂σ_xyd/∂y term
        for (unsigned j = 0; j < nDofs_sxyd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
        }
        // ∂σ_xyd/∂x term
        for (unsigned j = 0; j < nDofs_sxyd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[6].elem_dofs()[j];
        }
        // ∂σ_yyd/∂y term
        for (unsigned j = 0; j < nDofs_syyd; ++j) {
            div_sigmad += (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1] * (real_num_mov)unknowns_local[7].elem_dofs()[j];
        }
    }

//     real_num_mov source_term = f_val * (real_num_mov)phi_ud[i];
                        const real_num_mov u_rhs = f_val * (real_num_mov)phi_ud[i];

    unk_element_jac_res.res()[nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + i] +=
        (real_num)((mass_u + div_sigmad + mass_w - u_rhs) * weight_qp);

    // Jacobian contributions
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_udu = (real_num_mov)phi_ud[i] * (real_num_mov)phi_u[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + j] += (real_num)(jac_udu * weight_qp);
    }

        for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_wdu = (real_num_mov)phi_ud[i] * (real_num_mov)phi_w[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + j)] += (real_num)(jac_wdu * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov jac_udsxxd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + j)] += (real_num)(jac_udsxxd * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov jac_udsxyd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0] + (real_num_mov)gradphi_ud[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + j)] += (real_num)(jac_udsxyd * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov jac_udsyyd = (real_num_mov)gradphi_ud[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(jac_udsyyd * weight_qp);
    }
}


// Sixth equation

for (unsigned i = 0; i < nDofs_sxxd; ++i) {
    // R_sxxd = ∫(ε_xx(ud) + s_xxd)·v_sxxd dΩ
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

// Test functions for R_sxyd (seventh equation): 2ε_xy(ud) + s_xyd = 0
for (unsigned i = 0; i < nDofs_sxyd; ++i) {
    // R_sxyd = ∫(2ε_xy(ud) + s_xyd)·v_sxyd dΩ
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
        real_num_mov jac_sxydud = (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 1] +
                                 (real_num_mov)gradphi_sxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_ud[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + j)] += (real_num)(jac_sxydud * weight_qp);
    }

    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov jac_sxydsxyd = 2.0 * (real_num_mov)phi_sxyd[i] * (real_num_mov)phi_sxyd[j];
        unk_element_jac_res.jac()[(nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + i) * total_local_dofs + (nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + j)] += (real_num)(jac_sxydsxyd * weight_qp);
    }
}

// Test functions for R_syyd (eighth equation): ε_yy(ud) + s_yyd = 0
for (unsigned i = 0; i < nDofs_syyd; ++i) {
    // R_syyd = ∫(ε_yy(ud) + s_yyd)·v_syyd dΩ
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


// ------------------ 9. Equation for w (aResW) ------------------
// Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7)
unsigned row_9_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd;

// Column Mapping
unsigned col_9_idx  = row_9_idx;            // Variable index 8: w
unsigned col_10_idx = row_9_idx + nDofs_w;  // Variable index 9: wsxxd

for (unsigned i = 0; i < nDofs_w; ++i) {
    // --- 1. RESIDUAL CALCULATION ---
    real_num_mov strain_w_w = 0.0;
    // Part A: Strain-like coupling with w (Var index 8)
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        strain_w_w += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    real_num_mov mass_w_wsxxd = 0.0;
    // Part B: Mass coupling with wsxxd (Var index 9)
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov val = (real_num_mov)phi_w[i] * (real_num_mov)phi_wsxxd[j];
        mass_w_wsxxd += val * (real_num_mov)unknowns_local[9].elem_dofs()[j];
    }

    // Final Residual Update for Row 9
    unk_element_jac_res.res()[row_9_idx + i] += (real_num)((strain_w_w + mass_w_wsxxd) * weight_qp);

    // --- 2. JACOBIAN CONTRIBUTIONS ---

    // Block [9, 9]: w coupling with w
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_ww = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(row_9_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(jac_ww * weight_qp);
    }

    // Block [9, 10]: w coupling with wsxxd
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov jac_wwsxxd = (real_num_mov)phi_w[i] * (real_num_mov)phi_wsxxd[j];
        unk_element_jac_res.jac()[(row_9_idx + i) * total_local_dofs + (col_10_idx + j)] += (real_num)(jac_wwsxxd * weight_qp);
    }
}



/*
// ------------------ 9. Equation for w (aResW) ------------------
// Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7)
unsigned row_9_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd;

// Column Mapping
unsigned col_9_idx  = row_9_idx;            // Variable index 8: w
unsigned col_10_idx = row_9_idx + nDofs_w;  // Variable index 9: wsxxd

for (unsigned i = 0; i < nDofs_w; ++i) {
    real_num_mov strain_w_w = 0.0;
    real_num_mov mass_w_wsxxd = 0.0; // Ensure this is initialized inside the i-loop

    // --- Part A: Strain-like coupling with w (9th Column / Var index 8) ---
    // Term: ∫ grad(phi_w_i) * grad(phi_w_j) * w_j
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_w[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        strain_w_w += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];

        // Jacobian [9, 9]
        unk_element_jac_res.jac()[(row_9_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(val * weight_qp);
    }

    // --- Part B: Mass coupling with wsxxd (10th Column / Var index 9) ---
    // Term: ∫ phi_w_i * phi_wsxxd_j * wsxxd_j
    for (unsigned j = 0; j < nDofs_wsxxd; ++j) {
        real_num_mov val = (real_num_mov)phi_w[i] * (real_num_mov)phi_wsxxd[j];
        mass_w_wsxxd += val * (real_num_mov)unknowns_local[9].elem_dofs()[j];

        // Jacobian [9, 10]
        unk_element_jac_res.jac()[(row_9_idx + i) * total_local_dofs + (col_10_idx + j)] += (real_num)(val * weight_qp);
    }

    // --- Final Residual Update for Row 9 ---
    unk_element_jac_res.res()[row_9_idx + i] += (real_num)((strain_w_w + mass_w_wsxxd) * weight_qp);
}
*/


// ------------------ 10. Equation for wsxxd (aReswsxxd) ------------------
// Target: 10th row. Offset follows variable 7 (syyd) and variable 8 (w).
unsigned row_10_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w;

// Target: 9th column (Variable index 8: w)
//unsigned col_9_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd;

// Target: 11th column (Variable index 10: wsxyd)
unsigned col_11_idx = row_10_idx + nDofs_wsxxd;

for (unsigned i = 0; i < nDofs_wsxxd; ++i) {
    // --- 1. RESIDUAL CALCULATION ---
    real_num_mov strain_wxy_ud = 0.0;
    // Part A: Strain block on the 9th Column (w)
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] +
                           (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        strain_wxy_ud += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    real_num_mov mass_wsxyd = 0.0;
    // Part B: Mass block on the 11th Column (wsxyd)
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov val = 2.0 * (real_num_mov)phi_wsxxd[i] * (real_num_mov)phi_wsxyd[j];
        mass_wsxyd += val * (real_num_mov)unknowns_local[10].elem_dofs()[j];
    }

    // Update Residual for Row 10
    unk_element_jac_res.res()[row_10_idx + i] += (real_num)((strain_wxy_ud + mass_wsxyd) * weight_qp);

    // --- 2. JACOBIAN CONTRIBUTIONS ---

    // Jacobian Block [10, 9]: Coupling with w
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov jac_wsxxd_w = (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] +
                                   (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(row_10_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(jac_wsxxd_w * weight_qp);
    }

    // Jacobian Block [10, 11]: Coupling with wsxyd
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov jac_wsxxd_wsxyd = 2.0 * (real_num_mov)phi_wsxxd[i] * (real_num_mov)phi_wsxyd[j];
        unk_element_jac_res.jac()[(row_10_idx + i) * total_local_dofs + (col_11_idx + j)] += (real_num)(jac_wsxxd_wsxyd * weight_qp);
    }
}


/*
// ------------------ 10. Equation for wsxxd (aReswsxxd) ------------------
// Target: 10th row. Offset follows variable 7 (syyd) and variable 8 (w).
unsigned row_10_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w;

// Target: 9th column (Variable index 8: w)
// unsigned col_9_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd;

// Target: 11th column (Variable index 10: wsxyd)
unsigned col_11_idx = row_10_idx + nDofs_wsxxd;

for (unsigned i = 0; i < nDofs_wsxxd; ++i) {
    real_num_mov strain_wxy_ud = 0.0;
    real_num_mov mass_wsxyd = 0.0;

    // --- Part A: Strain block on the 9th Column (w) ---
    // Mapping: ∫ 2ε_xy(w) · v_wsxxd
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1] +
                           (real_num_mov)gradphi_wsxxd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0];

        strain_wxy_ud += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];

        // Jacobian Block [10, 9]
        unk_element_jac_res.jac()[(row_10_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(val * weight_qp);
    }

    // --- Part B: Mass block on the 11th Column (wsxyd) ---
    // Mapping: ∫ 2.0 * s_wsxyd · v_wsxxd
    for (unsigned j = 0; j < nDofs_wsxyd; ++j) {
        real_num_mov val = 2.0 * (real_num_mov)phi_wsxxd[i] * (real_num_mov)phi_wsxyd[j];

        mass_wsxyd += val * (real_num_mov)unknowns_local[10].elem_dofs()[j];

        // Jacobian Block [10, 11]
        unk_element_jac_res.jac()[(row_10_idx + i) * total_local_dofs + (col_11_idx + j)] += (real_num)(val * weight_qp);
    }

    // Update Residual for Row 10
    unk_element_jac_res.res()[row_10_idx + i] += (real_num)((strain_wxy_ud + mass_wsxyd) * weight_qp);
}
*/


// ------------------ 11. Equation for wsxyd (aReswsxyd) ------------------
// Row Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7), w(8), wsxxd(9)
unsigned row_11_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd;

// Column Mapping
//unsigned col_9_idx  = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd; // Var 8: w
unsigned col_12_idx = row_11_idx + nDofs_wsxyd; // Var 11: wsyyd

for (unsigned i = 0; i < nDofs_wsxyd; ++i) {
    // --- 1. RESIDUAL CALCULATION ---
    real_num_mov strain_w_yy = 0.0;
    // Part A: Strain block on the 9th Column (Var 8: w)
    // Mapping: ∫ ε_yy(w) · v_wsxyd
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        strain_w_yy += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    real_num_mov mass_wsxyd_wsyyd = 0.0;
    // Part B: Mass block on the 12th Column (Var 11: wsyyd)
    // Mapping: ∫ phi_wsxyd * phi_wsyyd
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov val = (real_num_mov)phi_wsxyd[i] * (real_num_mov)phi_wsyyd[j];
        mass_wsxyd_wsyyd += val * (real_num_mov)unknowns_local[11].elem_dofs()[j];
    }

    // Update Residual for Row 11
    unk_element_jac_res.res()[row_11_idx + i] += (real_num)((strain_w_yy + mass_wsxyd_wsyyd) * weight_qp);

    // --- 2. JACOBIAN CONTRIBUTIONS ---

    // Jacobian Block [11, 9]: Coupling with w
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(row_11_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(val * weight_qp);
    }

    // Jacobian Block [11, 12]: Coupling with wsyyd
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov val = (real_num_mov)phi_wsxyd[i] * (real_num_mov)phi_wsyyd[j];
        unk_element_jac_res.jac()[(row_11_idx + i) * total_local_dofs + (col_12_idx + j)] += (real_num)(val * weight_qp);
    }
}


/*
// ------------------ 11. Equation for wsxyd (aResWsxyd) ------------------
// Row Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7), w(8), wsxxd(9)
unsigned row_11_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd;

// Column Mapping
//unsigned col_9_idx  = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd; // Var 8: w
unsigned col_12_idx = row_11_idx + nDofs_wsxyd; // Var 11: wsyyd

for (unsigned i = 0; i < nDofs_wsxyd; ++i) {
    real_num_mov strain_w_yy = 0.0;
    real_num_mov mass_wsxyd_wsyyd = 0.0;

    // --- Part A: Strain block on the 9th Column (Var 8: w) ---
    // Mapping: ∫ ε_yy(w) · v_wsxyd
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov val = (real_num_mov)gradphi_wsxyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];

        strain_w_yy += val * (real_num_mov)unknowns_local[8].elem_dofs()[j];

        // Jacobian Block [11, 9]
        unk_element_jac_res.jac()[(row_11_idx + i) * total_local_dofs + (col_9_idx + j)] += (real_num)(val * weight_qp);
    }

    // --- Part B: Mass block on the 12th Column (Var 11: wsyyd) ---
    // Mapping: ∫ phi_wsxyd * phi_wsyyd
    for (unsigned j = 0; j < nDofs_wsyyd; ++j) {
        real_num_mov val = (real_num_mov)phi_wsxyd[i] * (real_num_mov)phi_wsyyd[j];

        mass_wsxyd_wsyyd += val * (real_num_mov)unknowns_local[11].elem_dofs()[j];

        // Jacobian Block [11, 12]
        unk_element_jac_res.jac()[(row_11_idx + i) * total_local_dofs + (col_12_idx + j)] += (real_num)(val * weight_qp);
    }

    // Update Residual for Row 11
    unk_element_jac_res.res()[row_11_idx + i] += (real_num)((strain_w_yy + mass_wsxyd_wsyyd) * weight_qp);
}
*/


// ------------------ 12. Equation for wsyyd (aReswsyyd) ------------------
// Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7), w(8), wsxxd(9), wsxyd(10)
unsigned row_wsyyd_idx = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd;

// Column Offsets
unsigned col_u_idx     = 0;
unsigned shift_offset  = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud; // Points to Var 5 (sxxd)
unsigned col_w_idx     = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd; // Var 8

for (unsigned i = 0; i < nDofs_wsyyd; ++i) {
    // --- 1. RESIDUAL CALCULATION ---
    real_num_mov res_val = 0.0;

    // Part A: Coupling with Primal Displacement u (Var index 0)
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov a_utilde = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_u[j];
        res_val += a_utilde * (real_num_mov)unknowns_local[0].elem_dofs()[j];
    }

    // Part B: Adjoint Stress Couplings (sxxd, sxyd, syyd)
    // B_xx (Var 5)
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov bxx = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0];
        res_val += bxx * (real_num_mov)unknowns_local[5].elem_dofs()[j];
    }
    // B_xy (Var 6)
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov bxy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] +
                           (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0];
        res_val += bxy * (real_num_mov)unknowns_local[6].elem_dofs()[j];
    }
    // B_yy (Var 7)
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov byy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
        res_val += byy * (real_num_mov)unknowns_local[7].elem_dofs()[j];
    }

    // Part C: Coupling with Adjoint Displacement w (Var index 8)
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov stiff = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] +
                             (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        real_num_mov mass = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_w[j];
        real_num_mov total_w_kernel = (gamma * stiff + beta * mass);
        res_val += total_w_kernel * (real_num_mov)unknowns_local[8].elem_dofs()[j];
    }

    const real_num_mov w_rhs = f_val * (real_num_mov)phi_wsyyd[i];

    // Final Residual Update
    unk_element_jac_res.res()[row_wsyyd_idx + i] += (real_num)(res_val - w_rhs) * weight_qp;

    // --- 2. JACOBIAN CONTRIBUTIONS ---

    // Block [11, 0]: Coupling with u
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov jac_utilde = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_u[j];
        unk_element_jac_res.jac()[(row_wsyyd_idx + i) * total_local_dofs + (col_u_idx + j)] += (real_num)(jac_utilde * weight_qp);
    }

    // Block [11, 5]: Coupling with sxxd
    for (unsigned j = 0; j < nDofs_sxxd; ++j) {
        real_num_mov bxx = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(row_wsyyd_idx + i) * total_local_dofs + (shift_offset + j)] += (real_num)(bxx * weight_qp);
    }

    // Block [11, 6]: Coupling with sxyd
    for (unsigned j = 0; j < nDofs_sxyd; ++j) {
        real_num_mov bxy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] +
                           (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0];
        unk_element_jac_res.jac()[(row_wsyyd_idx + i) * total_local_dofs + (shift_offset + nDofs_sxxd + j)] += (real_num)(bxy * weight_qp);
    }

    // Block [11, 7]: Coupling with syyd
    for (unsigned j = 0; j < nDofs_syyd; ++j) {
        real_num_mov byy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
        unk_element_jac_res.jac()[(row_wsyyd_idx + i) * total_local_dofs + (shift_offset + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(byy * weight_qp);
    }

    // Block [11, 8]: Coupling with w
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov stiff = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] +
                             (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        real_num_mov mass = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_w[j];
        real_num_mov total_w_kernel = (gamma * stiff + beta * mass);
        unk_element_jac_res.jac()[(row_wsyyd_idx + i) * total_local_dofs + (col_w_idx + j)] += (real_num)(total_w_kernel * weight_qp);
    }
}



/*
// ------------------ 12. Equation for wsyyd (aResWsyyd) ------------------
// Offset: u(0), sxx(1), sxy(2), syy(3), ud(4), sxxd(5), sxyd(6), syyd(7), w(8), wsxxd(9), wsxyd(10), wsyyd(11)
unsigned row_wsyyd = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd + nDofs_w + nDofs_wsxxd + nDofs_wsxyd;

for (unsigned i = 0; i < nDofs_wsyyd; ++i) {
    real_num_mov res_val = 0.0;

    // --- 1. A_utilde Term (Coupling with Primal Displacement u) ---
    // Column 0
    for (unsigned j = 0; j < nDofs_u; ++j) {
        real_num_mov a_utilde = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_u[j]; // Replace with your specific A_u kernel
        res_val += a_utilde * (real_num_mov)unknowns_local[0].elem_dofs()[j];

        unk_element_jac_res.jac()[(row_wsyyd + i) * total_local_dofs + j] += (real_num)(a_utilde * weight_qp);
    }

// --- 2. B_xx, B_xy, B_yy Terms (Shifted 4 columns/blocks further) ---
// New Variable Indices: unknowns_local[5], [6], and [7]
// New Column Offsets: nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud

unsigned shift_offset = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud;

// B_xx (Shifted Col: shift_offset)
for (unsigned j = 0; j < nDofs_sxxd; ++j) {
    real_num_mov bxx = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxxd[j * dim_offset_grad + 0];
    res_val += bxx * (real_num_mov)unknowns_local[5].elem_dofs()[j];
    unk_element_jac_res.jac()[(row_wsyyd + i) * total_local_dofs + (shift_offset + j)] += (real_num)(bxx * weight_qp);
}

// B_xy (Shifted Col: shift_offset + nDofs_sxxd)
for (unsigned j = 0; j < nDofs_sxyd; ++j) {
    real_num_mov bxy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 1] +
                       (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_sxyd[j * dim_offset_grad + 0];
    res_val += bxy * (real_num_mov)unknowns_local[6].elem_dofs()[j];
    unk_element_jac_res.jac()[(row_wsyyd + i) * total_local_dofs + (shift_offset + nDofs_sxxd + j)] += (real_num)(bxy * weight_qp);
}

// B_yy (Shifted Col: shift_offset + nDofs_sxxd + nDofs_sxyd)
for (unsigned j = 0; j < nDofs_syyd; ++j) {
    real_num_mov byy = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_syyd[j * dim_offset_grad + 1];
    res_val += byy * (real_num_mov)unknowns_local[7].elem_dofs()[j];
    unk_element_jac_res.jac()[(row_wsyyd + i) * total_local_dofs + (shift_offset + nDofs_sxxd + nDofs_sxyd + j)] += (real_num)(byy * weight_qp);
}

    // --- 3. A_w + beta*M_w + gamma*K_w (Coupling with Adjoint Displacement w) ---
    // Column: nDofs_u + ... + nDofs_syyd (Variable index 8)
    unsigned col_w = nDofs_u + nDofs_sxx + nDofs_sxy + nDofs_syy + nDofs_ud + nDofs_sxxd + nDofs_sxyd + nDofs_syyd;
    for (unsigned j = 0; j < nDofs_w; ++j) {
        real_num_mov stiff = (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 0] * (real_num_mov)gradphi_w[j * dim_offset_grad + 0] +
                             (real_num_mov)gradphi_wsyyd[i * dim_offset_grad + 1] * (real_num_mov)gradphi_w[j * dim_offset_grad + 1];
        real_num_mov mass = (real_num_mov)phi_wsyyd[i] * (real_num_mov)phi_w[j];

        real_num_mov total_w_kernel = (gamma * stiff + beta * mass); // Add Aw component if different from mass/stiff
        res_val += total_w_kernel * (real_num_mov)unknowns_local[8].elem_dofs()[j];

        unk_element_jac_res.jac()[(row_wsyyd + i) * total_local_dofs + (col_w + j)] += (real_num)(total_w_kernel * weight_qp);
    }
   const real_num_mov w_rhs = f_val * (real_num_mov)phi_wsyyd[i];
    // Final Residual Update
    unk_element_jac_res.res()[row_wsyyd + i] += (real_num)(res_val - w_rhs ) * weight_qp;
}
*/



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
        constexpr bool print_algebra_local = true;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = unk_num_elem_dofs;
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 12, 12);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 12, 12);
        }

    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
}

  };

}

#endif
