#ifndef __femus_biharmonic_HM_hpp__
#define __femus_biharmonic_HM_hpp__
 
#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp" // Required for ElementJacRes
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

/*
//========= BOUNDARY_IMPLEMENTATION_V - BEGIN ==================

static void natural_loop_1dV(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_v,
                     const unsigned solFEType_v,
                     std::vector< double > & Res
                    ) {

     double grad_v_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_v.c_str(), grad_v_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_v_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_v);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_v_dot_n ;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dV(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_v,
                       const unsigned solFEType_v,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector <  elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
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
  std::vector <double> phi_v_bdry;
  std::vector <double> phi_v_x_bdry;

  phi_v_bdry.reserve(max_size);
  phi_v_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_v_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_v.c_str(), grad_v_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_v = msh->GetElementFaceDofNumber(iel, jface, solFEType_v);

// dof-based - BEGIN
     std::vector< double > grad_v_dot_n_at_dofs(n_dofs_face_v);


    for (unsigned i_bdry = 0; i_bdry < grad_v_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_v_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_v.c_str(), grad_v_dot_n_at_dofs_temp, face, 0.);
     grad_v_dot_n_at_dofs[i_bdry] = grad_v_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_v ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_v_bdry, phi_v_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_v_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_v_bdry.size(); i_bdry ++) {
           grad_v_dot_n_qp +=  grad_v_dot_n_at_dofs[i_bdry] * phi_v_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_v; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_v_dot_n_qp  * phi_v_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_V - END ==================

*/



// // // //========= BOUNDARY_IMPLEMENTATION_S1 - BEGIN ==================
// // //
// // // static void natural_loop_1dS1(const MultiLevelProblem *    ml_prob,
// // //                      const Mesh *                    msh,
// // //                      const MultiLevelSolution *    ml_sol,
// // //                      const unsigned iel,
// // //                      CurrentElem < double > & geom_element,
// // //                      const unsigned xType,
// // //                      const std::string solname_s1,
// // //                      const unsigned solFEType_s1,
// // //                      std::vector< double > & Res
// // //                     ) {
// // //
// // //      double grad_s1_dot_n = 0.;
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
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s1.c_str(), grad_s1_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet)  &&  (grad_s1_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //                    unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_s1);
// // //
// // //                   for (unsigned i = 0; i < n_dofs_face; i++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
// // //
// // //                  Res[i_vol] +=  grad_s1_dot_n /* * phi[node] = 1. */;
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
// // //                        const std::string solname_s1,
// // //                        const unsigned solFEType_s1,
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
// // //   std::vector <double> phi_s1_bdry;
// // //   std::vector <double> phi_s1_x_bdry;
// // //
// // //   phi_s1_bdry.reserve(max_size);
// // //   phi_s1_x_bdry.reserve(max_size * space_dim);
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
// // //      double grad_s1_dot_n = 0.;
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
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s1.c_str(), grad_s1_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //     unsigned n_dofs_face_s1 = msh->GetElementFaceDofNumber(iel, jface, solFEType_s1);
// // //
// // // // dof-based - BEGIN
// // //      std::vector< double > grad_s1_dot_n_at_dofs(n_dofs_face_s1);
// // //
// // //
// // //     for (unsigned i_bdry = 0; i_bdry < grad_s1_dot_n_at_dofs.size(); i_bdry++) {
// // //         std::vector<double> x_at_node(dim, 0.);
// // //         for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
// // //
// // //       double grad_s1_dot_n_at_dofs_temp = 0.;
// // //       ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_s1.c_str(), grad_s1_dot_n_at_dofs_temp, face, 0.);
// // //      grad_s1_dot_n_at_dofs[i_bdry] = grad_s1_dot_n_at_dofs_temp;
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
// // //     elem_all[ielGeom_bdry][solFEType_s1 ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_s1_bdry, phi_s1_x_bdry,  boost::none, space_dim);
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
// // //            double grad_s1_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp
// // //
// // // // dof-based
// // //          for (unsigned i_bdry = 0; i_bdry < phi_s1_bdry.size(); i_bdry ++) {
// // //            grad_s1_dot_n_qp +=  grad_s1_dot_n_at_dofs[i_bdry] * phi_s1_bdry[i_bdry];
// // //          }
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //
// // //
// // //                   for (unsigned i_bdry = 0; i_bdry < n_dofs_face_s1; i_bdry++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
// // //
// // //                  Res[i_vol] +=  weight_iqp_bdry * grad_s1_dot_n_qp /*grad_u_dot_n*/  * phi_s1_bdry[i_bdry];
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
// // // //========= BOUNDARY_IMPLEMENTATION_S1 - END ==================






// // // //========= BOUNDARY_IMPLEMENTATION_S2 - BEGIN ==================
// // //
// // // static void natural_loop_1dS2(const MultiLevelProblem *    ml_prob,
// // //                      const Mesh *                    msh,
// // //                      const MultiLevelSolution *    ml_sol,
// // //                      const unsigned iel,
// // //                      CurrentElem < double > & geom_element,
// // //                      const unsigned xType,
// // //                      const std::string solname_s2,
// // //                      const unsigned solFEType_s2,
// // //                      std::vector< double > & Res
// // //                     ) {
// // //
// // //      double grad_s2_dot_n = 0.;
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
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s2.c_str(), grad_s2_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet)  &&  (grad_s2_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //
// // //
// // //                    unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_s2);
// // //
// // //                   for (unsigned i = 0; i < n_dofs_face; i++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
// // //
// // //                  Res[i_vol] +=  grad_s2_dot_n /* * phi[node] = 1. */;
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
// // //                        const std::string solname_s2,
// // //                        const unsigned solFEType_s2,
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
// // //   std::vector <double> phi_s2_bdry;
// // //   std::vector <double> phi_s2_x_bdry;
// // //
// // //   phi_s2_bdry.reserve(max_size);
// // //   phi_s2_x_bdry.reserve(max_size * space_dim);
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
// // //      double grad_s2_dot_n = 0.;
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
// // //          bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s2.c_str(), grad_s2_dot_n, face, 0.);
// // //          //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
// // //          //while here we pass the FACE ELEMENT CENTER coordinates.
// // //          // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!
// // //
// // //              if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann
// // //
// // //     unsigned n_dofs_face_s2 = msh->GetElementFaceDofNumber(iel, jface, solFEType_s2);
// // //
// // // // dof-based - BEGIN
// // //      std::vector< double > grad_s2_dot_n_at_dofs(n_dofs_face_s2);
// // //
// // //
// // //     for (unsigned i_bdry = 0; i_bdry < grad_s2_dot_n_at_dofs.size(); i_bdry++) {
// // //         std::vector<double> x_at_node(dim, 0.);
// // //         for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
// // //
// // //       double grad_s2_dot_n_at_dofs_temp = 0.;
// // //       ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_s2.c_str(), grad_s2_dot_n_at_dofs_temp, face, 0.);
// // //      grad_s2_dot_n_at_dofs[i_bdry] = grad_s2_dot_n_at_dofs_temp;
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
// // //     elem_all[ielGeom_bdry][solFEType_s2 ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_s2_bdry, phi_s2_x_bdry,  boost::none, space_dim);
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
// // //            double grad_s2_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp
// // //
// // // // dof-based
// // //          for (unsigned i_bdry = 0; i_bdry < phi_s2_bdry.size(); i_bdry ++) {
// // //            grad_s2_dot_n_qp +=  grad_s2_dot_n_at_dofs[i_bdry] * phi_s2_bdry[i_bdry];
// // //          }
// // //
// // // //---------------------------------------------------------------------------------------------------------
// // //
// // //
// // //
// // //                   for (unsigned i_bdry = 0; i_bdry < n_dofs_face_s2; i_bdry++) {
// // //
// // //                  unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
// // //
// // //                  Res[i_vol] +=  weight_iqp_bdry * grad_s2_dot_n_qp /*grad_u_dot_n*/  * phi_s2_bdry[i_bdry];
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
     const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
                            const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
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

    // Keep same conventions as your code
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

    // Unknowns setup - expect 5 unknowns for Ciarlet-Raviart formulation
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 5) {
        std::cerr << "AssembleBilaplaceProblem: expected 5 unknowns (u, v, s1, s2, p) but found " << n_unknowns << "\n";
        return;
    }

    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad));

    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    // element loop: each process loops only on the elements it owns
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
        const unsigned nDofs_v = unk_num_elem_dofs[1];
        const unsigned nDofs_s1 = unk_num_elem_dofs[2];
        const unsigned nDofs_s2 = unk_num_elem_dofs[3];
        const unsigned nDofs_p = unk_num_elem_dofs[4];

        // Gauss loop
        const unsigned nGauss = quad_rules[ielGeom].GetGaussPointsNumber();
        for (unsigned ig = 0; ig < nGauss; ig++) {

            // Jacobian (element geometry)
            elem_all_for_domain[ielGeom][xType]->JacJacInv(geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            // Evaluate shape functions for each unknown at this gauss point
            for (unsigned u = 0; u < n_unknowns; u++) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim
                );
            }

            // Geometry shape functions
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim
            );

            // Local references for shape functions
            auto & phi_u = unknowns_phi_dof_qp[0].phi();
            auto & gradphi_u = unknowns_phi_dof_qp[0].phi_grad();
            auto & phi_v = unknowns_phi_dof_qp[1].phi();
            auto & gradphi_v = unknowns_phi_dof_qp[1].phi_grad();
            auto & phi_s1 = unknowns_phi_dof_qp[2].phi();
            auto & gradphi_s1 = unknowns_phi_dof_qp[2].phi_grad();
            auto & phi_s2 = unknowns_phi_dof_qp[3].phi();
            auto & gradphi_s2 = unknowns_phi_dof_qp[3].phi_grad();
            auto & phi_p = unknowns_phi_dof_qp[4].phi();
            auto & gradphi_p = unknowns_phi_dof_qp[4].phi_grad();

            // Interpolate solution values & gradients at gauss point for each unknown
            real_num_mov u_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_u_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_u; ++a) {
                u_val_g += (real_num_mov) phi_u[a] * (real_num_mov) unknowns_local[0].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_u_g[d] += (real_num_mov) gradphi_u[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[0].elem_dofs()[a];
                }
            }

            real_num_mov v_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_v_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_v; ++a) {
                v_val_g += (real_num_mov) phi_v[a] * (real_num_mov) unknowns_local[1].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_v_g[d] += (real_num_mov) gradphi_v[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[1].elem_dofs()[a];
                }
            }

            real_num_mov s1_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_s1_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_s1; ++a) {
                s1_val_g += (real_num_mov) phi_s1[a] * (real_num_mov) unknowns_local[2].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_s1_g[d] += (real_num_mov) gradphi_s1[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[2].elem_dofs()[a];
                }
            }

            real_num_mov s2_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_s2_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_s2; ++a) {
                s2_val_g += (real_num_mov) phi_s2[a] * (real_num_mov) unknowns_local[3].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_s2_g[d] += (real_num_mov) gradphi_s2[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[3].elem_dofs()[a];
                }
            }

            real_num_mov p_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_p_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_p; ++a) {
                p_val_g += (real_num_mov) phi_p[a] * (real_num_mov) unknowns_local[4].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    grad_p_g[d] += (real_num_mov) gradphi_p[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[4].elem_dofs()[a];
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

            // Source f(x) at gauss point
            const real_num_mov f_val = (real_num_mov) source_functions[0]->value(x_gss);
            const double alpha = 0.00000001; // Regularization parameter

            // ==================== RESIDUAL AND JACOBIAN CALCULATIONS ====================

            // Test functions for R_u (first equation): -Δu + v = 0
            for (unsigned i = 0; i < nDofs_u; ++i) {
                // R_u = ∫(∇u·∇v_u + v·v_u) dΩ
                real_num_mov laplace_u = 0.0;
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    laplace_u += grad_u_g[d] * (real_num_mov) gradphi_u[i * dim_offset_grad + d];
                }

                real_num_mov mass_v = v_val_g * (real_num_mov)phi_u[i];

                unk_element_jac_res.res()[i] += (real_num)((laplace_u + mass_v) * weight_qp);

                // Jacobian: ∂R_u/∂u_j and ∂R_u/∂v_j
                for (unsigned j = 0; j < nDofs_u; ++j) {
                    real_num_mov jac_uu = 0.0;
                    for (unsigned d = 0; d < dim_offset_grad; ++d) {
                        jac_uu += (real_num_mov)gradphi_u[i * dim_offset_grad + d] * (real_num_mov)gradphi_u[j * dim_offset_grad + d];
                    }
                    unk_element_jac_res.jac()[i * total_local_dofs + j] += (real_num)(jac_uu * weight_qp);
                }

                for (unsigned j = 0; j < nDofs_v; ++j) {
                    real_num_mov jac_uv = (real_num_mov)phi_u[i] * (real_num_mov)phi_v[j];
                    unk_element_jac_res.jac()[i * total_local_dofs + (nDofs_u + j)] += (real_num)(jac_uv * weight_qp);
                }
            }

            // Test functions for R_v (second equation): -Δv + p = 0
            for (unsigned i = 0; i < nDofs_v; ++i) {
                // R_v = ∫(∇v·∇v_v + p·v_v) dΩ
                real_num_mov laplace_v = 0.0;
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    laplace_v += grad_v_g[d] * (real_num_mov) gradphi_v[i * dim_offset_grad + d];
                }

                real_num_mov mass_p = p_val_g * (real_num_mov)phi_v[i];

                unk_element_jac_res.res()[nDofs_u + i] += (real_num)((laplace_v + mass_p) * weight_qp);

                // Jacobian: ∂R_v/∂v_j and ∂R_v/∂p_j
                for (unsigned j = 0; j < nDofs_v; ++j) {
                    real_num_mov jac_vv = 0.0;
                    for (unsigned d = 0; d < dim_offset_grad; ++d) {
                        jac_vv += (real_num_mov)gradphi_v[i * dim_offset_grad + d] * (real_num_mov)gradphi_v[j * dim_offset_grad + d];
                    }
                    unk_element_jac_res.jac()[(nDofs_u + i) * total_local_dofs + (nDofs_u + j)] += (real_num)(jac_vv * weight_qp);
                }

                for (unsigned j = 0; j < nDofs_p; ++j) {
                    real_num_mov jac_vp = (real_num_mov)phi_v[i] * (real_num_mov)phi_p[j];
                    unk_element_jac_res.jac()[(nDofs_u + i) * total_local_dofs + (nDofs_u + nDofs_v + nDofs_s1 + nDofs_s2 + j)] += (real_num)(jac_vp * weight_qp);
                }
            }

            // Test functions for R_s1 (third equation): -Δs1 + s2 = 0
            for (unsigned i = 0; i < nDofs_s1; ++i) {
                // R_s1 = ∫(∇s1·∇v_s1 + s2·v_s1) dΩ
                real_num_mov laplace_s1 = 0.0;
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    laplace_s1 += grad_s1_g[d] * (real_num_mov) gradphi_s1[i * dim_offset_grad + d];
                }

                real_num_mov mass_s2 = s2_val_g * (real_num_mov)phi_s1[i];

                unk_element_jac_res.res()[nDofs_u + nDofs_v + i] += (real_num)((laplace_s1 + mass_s2) * weight_qp);

                // Jacobian: ∂R_s1/∂s1_j and ∂R_s1/∂s2_j
                for (unsigned j = 0; j < nDofs_s1; ++j) {
                    real_num_mov jac_s1s1 = 0.0;
                    for (unsigned d = 0; d < dim_offset_grad; ++d) {
                        jac_s1s1 += (real_num_mov)gradphi_s1[i * dim_offset_grad + d] * (real_num_mov)gradphi_s1[j * dim_offset_grad + d];
                    }
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + i) * total_local_dofs + (nDofs_u + nDofs_v + j)] += (real_num)(jac_s1s1 * weight_qp);
                }

                for (unsigned j = 0; j < nDofs_s2; ++j) {
                    real_num_mov jac_s1s2 = (real_num_mov)phi_s1[i] * (real_num_mov)phi_s2[j];
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + i) * total_local_dofs + (nDofs_u + nDofs_v + nDofs_s1 + j)] += (real_num)(jac_s1s2 * weight_qp);
                }
            }

            // Test functions for R_s2 (fourth equation): u - Δs2 - f = 0
            for (unsigned i = 0; i < nDofs_s2; ++i) {
                // R_s2 = ∫(u·v_s2 + ∇s2·∇v_s2 - f·v_s2) dΩ
                real_num_mov mass_u = u_val_g * (real_num_mov)phi_s2[i];

                real_num_mov laplace_s2 = 0.0;
                for (unsigned d = 0; d < dim_offset_grad; ++d) {
                    laplace_s2 += grad_s2_g[d] * (real_num_mov) gradphi_s2[i * dim_offset_grad + d];
                }

                real_num_mov source_term = f_val * (real_num_mov)phi_s2[i];

                unk_element_jac_res.res()[nDofs_u + nDofs_v + nDofs_s1 + i] +=
                    (real_num)((mass_u + laplace_s2 - source_term) * weight_qp);

                // Jacobian: ∂R_s2/∂u_j, ∂R_s2/∂s2_j
                for (unsigned j = 0; j < nDofs_u; ++j) {
                    real_num_mov jac_s2u = (real_num_mov)phi_s2[i] * (real_num_mov)phi_u[j];
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + nDofs_s1 + i) * total_local_dofs + j] += (real_num)(jac_s2u * weight_qp);
                }

                for (unsigned j = 0; j < nDofs_s2; ++j) {
                    real_num_mov jac_s2s2 = 0.0;
                    for (unsigned d = 0; d < dim_offset_grad; ++d) {
                        jac_s2s2 += (real_num_mov)gradphi_s2[i * dim_offset_grad + d] * (real_num_mov)gradphi_s2[j * dim_offset_grad + d];
                    }
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + nDofs_s1 + i) * total_local_dofs + (nDofs_u + nDofs_v + nDofs_s1 + j)] += (real_num)(jac_s2s2 * weight_qp);
                }
            }

            // Test functions for R_p (fifth equation): s1 + α·p = 0
            for (unsigned i = 0; i < nDofs_p; ++i) {
                // R_p = ∫(s1·v_p + α·p·v_p) dΩ
                real_num_mov mass_s1 = s1_val_g * (real_num_mov)phi_p[i];
                real_num_mov mass_p = alpha * p_val_g * (real_num_mov)phi_p[i];

                unk_element_jac_res.res()[nDofs_u + nDofs_v + nDofs_s1 + nDofs_s2 + i] +=
                    (real_num)((mass_s1 + mass_p) * weight_qp);

                // Jacobian: ∂R_p/∂s1_j and ∂R_p/∂p_j
                for (unsigned j = 0; j < nDofs_s1; ++j) {
                    real_num_mov jac_ps1 = (real_num_mov)phi_p[i] * (real_num_mov)phi_s1[j];
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + nDofs_s1 + nDofs_s2 + i) * total_local_dofs + (nDofs_u + nDofs_v + j)] += (real_num)(jac_ps1 * weight_qp);
                }

                for (unsigned j = 0; j < nDofs_p; ++j) {
                    real_num_mov jac_pp = alpha * (real_num_mov)phi_p[i] * (real_num_mov)phi_p[j];
                    unk_element_jac_res.jac()[(nDofs_u + nDofs_v + nDofs_s1 + nDofs_s2 + i) * total_local_dofs + (nDofs_u + nDofs_v + nDofs_s1 + nDofs_s2 + j)] += (real_num)(jac_pp * weight_qp);
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
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = {nDofs_u, nDofs_v, nDofs_s1, nDofs_s2, nDofs_p};
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
