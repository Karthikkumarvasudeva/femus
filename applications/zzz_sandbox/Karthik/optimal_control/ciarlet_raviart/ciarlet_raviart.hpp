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


static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use

  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> (ml_prob.get_app_specs_pointer()->_system_name);

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*          el         = msh->GetMeshElements();  // pointer to the elem object in msh (level)

  MultiLevelSolution*  ml_sol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

 const std::string solname_u = ml_sol->GetSolName_string_vec()[0];

  //solution variable
  unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());    // get the position of "u" in the ml_sol object
  unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);    // get the finite element type for "u"
  unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());    // get the position of "u" in the pdeSys object

  SparseMatrix*             JAC = pdeSys->_KK;

  std::vector < adept::adouble >  solu; // local solution


  const std::string solname_v = ml_sol->GetSolName_string_vec()[1];
  unsigned solvIndex = ml_sol->GetIndex(solname_v.c_str());    // get the position of "v" in the ml_sol object
  unsigned solFEType_v = ml_sol->GetSolutionType(solvIndex);    // get the finite element type for "v"
  unsigned solvPdeIndex = mlPdeSys->GetSolPdeIndex(solname_v.c_str());    // get the position of "v" in the pdeSys object
  std::vector < adept::adouble >  solv; // local solution

  const std::string solname_s1 = ml_sol->GetSolName_string_vec()[2];
  unsigned sols1Index = ml_sol->GetIndex(solname_s1.c_str());    // get the position of "v" in the ml_sol object
  unsigned solFEType_s1 = ml_sol->GetSolutionType(sols1Index);    // get the finite element type for "v"
  unsigned sols1PdeIndex = mlPdeSys->GetSolPdeIndex(solname_s1.c_str());    // get the position of "v" in the pdeSys object
  std::vector < adept::adouble >  sols1; // local solution

  const std::string solname_s2 = ml_sol->GetSolName_string_vec()[3];
  unsigned sols2Index = ml_sol->GetIndex(solname_s2.c_str());    // get the position of "v" in the ml_sol object
  unsigned solFEType_s2 = ml_sol->GetSolutionType(sols2Index);    // get the finite element type for "v"
  unsigned sols2PdeIndex = mlPdeSys->GetSolPdeIndex(solname_s2.c_str());    // get the position of "v" in the pdeSys object
  std::vector < adept::adouble >  sols2; // local solution

  const std::string solname_p = ml_sol->GetSolName_string_vec()[4];
  unsigned solpIndex = ml_sol->GetIndex(solname_p.c_str());    // get the position of "v" in the ml_sol object
  unsigned solFEType_p = ml_sol->GetSolutionType(solpIndex);    // get the finite element type for "v"
  unsigned solpPdeIndex = mlPdeSys->GetSolPdeIndex(solname_p.c_str());    // get the position of "v" in the pdeSys object
  std::vector < adept::adouble >  solp; // local solution


  std::vector < std::vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  std::vector < int > sysDof; // local to global pdeSys dofs
  std::vector <double> phi;  // local test function
  std::vector <double> phi_x; // local test function first order partial derivatives
  std::vector <double> phi_xx; // local test function second order partial derivatives
  double weight; // gauss point weight

  std::vector < double > Res; // local redidual vector
  std::vector < adept::adouble > aResu; // local redidual vector
  std::vector < adept::adouble > aResv; // local redidual vector

  std::vector < adept::adouble > aRess1; // local redidual vector
  std::vector < adept::adouble > aRess2; // local redidual vector
  std::vector < adept::adouble > aResp; // local redidual vector

  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27
  solu.reserve(maxSize);
  solv.reserve(maxSize);

  sols1.reserve(maxSize);
  sols2.reserve(maxSize);

  solp.reserve(maxSize);


  for (unsigned i = 0; i < dim; i++)
    x[i].reserve(maxSize);

  sysDof.reserve(5 * maxSize);

  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);
// // //   unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
    unsigned dim2 = (6 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)

  phi_xx.reserve(maxSize * dim2);

  Res.reserve(2 * maxSize);
  aResu.reserve(maxSize);
  aResv.reserve(maxSize);

  aRess1.reserve(maxSize);
  aRess2.reserve(maxSize);

  aResp.reserve(maxSize);


  std::vector < double > Jac; // local Jacobian matrix (ordered by column, adept)
  Jac.reserve(5 * maxSize * maxSize);

  KK->zero(); // Set to zero all the entries of the Global Matrix


double alpha = .001 ;



  for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel); 

// // //     unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_u);    // number of solution element dofs
    unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_p);    // number of solution element dofs



    unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs

    std::vector<unsigned> Sol_n_el_dofs_Mat_vol(5, nDofs);

    // resize local arrays
    sysDof.resize(5 * nDofs);
    solu.resize(nDofs);
    solv.resize(nDofs);
    sols1.resize(nDofs);
    sols2.resize(nDofs);
    solp.resize(nDofs);


    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofs2);
    }

    aResu.assign(nDofs, 0.);    //resize
    aResv.assign(nDofs, 0.);    //resize
    aRess1.assign(nDofs, 0.0);
    aRess2.assign(nDofs, 0.0);
    aResp.assign(nDofs, 0.0);


    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {

// // //       unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);    // global to global mapping between solution node and solution dof
      unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_v);    // global to global mapping between solution node and solution dof



      solu[i]          = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
      solv[i]          = (*sol->_Sol[solvIndex])(solDof);      // global extraction and local storage for the solution
      sols1[i]         = (*sol->_Sol[sols1Index])(solDof);      // s1  -> secondary row2, col2
      sols2[i]         = (*sol->_Sol[sols2Index])(solDof);      // s2  -> secondary row1, col2
      solp[i]         = (*sol->_Sol[solpIndex])(solDof);      // s2  -> secondary row1, col2



      sysDof[i]             = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[nDofs + i]     = pdeSys->GetSystemDof(solvIndex, solvPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[2 * nDofs + i] = pdeSys->GetSystemDof(sols1Index, sols1PdeIndex, i, iel); // s1
      sysDof[3 * nDofs + i] = pdeSys->GetSystemDof(sols2Index, sols2PdeIndex, i, iel); // s2
      sysDof[4 * nDofs + i] = pdeSys->GetSystemDof(solpIndex, solpPdeIndex, i, iel); // s2


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

    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_v]->GetGaussPointNumber(); ig++) {
// *** get gauss point weight, test function and test function partial derivatives ***

      msh->_finiteElement[ielGeom][solFEType_v]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

      // evaluate the solution, the solution derivatives and the coordinates in the gauss point
      adept::adouble soluGauss = 0;
      std::vector < adept::adouble > soluGauss_x(dim, 0.);

      adept::adouble solvGauss = 0;
      std::vector < adept::adouble > solvGauss_x(dim, 0.);

      adept::adouble sols1Gauss = 0;
      std::vector < adept::adouble > sols1Gauss_x(dim, 0.);

      adept::adouble sols2Gauss = 0;
      std::vector < adept::adouble > sols2Gauss_x(dim, 0.);

      adept::adouble solpGauss = 0;
      std::vector < adept::adouble > solpGauss_x(dim, 0.);


      std::vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        soluGauss += phi[i] * solu[i];
        solvGauss += phi[i] * solv[i];

        sols1Gauss += phi[i] * sols1[i];
        sols2Gauss += phi[i] * sols2[i];

        solpGauss += phi[i] * solp[i];


        for (unsigned jdim = 0; jdim < dim; jdim++) {
          soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
          solvGauss_x[jdim] += phi_x[i * dim + jdim] * solv[i];

          sols1Gauss_x[jdim] += phi_x[i * dim + jdim] * sols1[i];
          sols2Gauss_x[jdim] += phi_x[i * dim + jdim] * sols2[i];

          solpGauss_x[jdim] += phi_x[i * dim + jdim] * solp[i];


          xGauss[jdim] += x[jdim][i] * phi[i];
        }
      }
      // *** phi_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {

        adept::adouble Laplace_u = 0.;
        adept::adouble Laplace_v = 0.;

        adept::adouble Laplace_s1 = 0.;
        adept::adouble Laplace_s2 = 0.;
        adept::adouble Laplace_p = 0.;


        adept::adouble M_u = phi[i] * soluGauss;
        adept::adouble M_v = phi[i] * solvGauss;
        adept::adouble M_s1 = phi[i] * sols1Gauss;
        adept::adouble M_s2 = phi[i] * sols2Gauss;
        adept::adouble M_p = phi[i] * solpGauss;


        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * soluGauss_x[jdim];
          Laplace_v   +=  - phi_x[i * dim + jdim] * solvGauss_x[jdim];

          Laplace_s1   +=  - phi_x[i * dim + jdim] * sols1Gauss_x[jdim];
          Laplace_s2   +=  - phi_x[i * dim + jdim] * sols2Gauss_x[jdim];

          Laplace_p   +=  - phi_x[i * dim + jdim] * solpGauss_x[jdim];

        }

        double pi = acos(-1.);


// // //         adept::adouble F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

     adept::adouble F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->value(xGauss) * phi[i];

     adept::adouble F_term_d = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->value(xGauss) * phi[i];



        // System residuals - signs adjusted to match matrix form
     aResu[i] += ( Laplace_u + M_v) * weight;  // M*W + B^T*U = 0
     aResv[i] += (Laplace_v + M_p ) * weight;  // B*W + ν1*C1*S1 + ν1*C2*S2 = -ν2*F
     aRess1[i] += ( Laplace_s1 + M_s2) * weight;  // C1^T*W + M*S1 = 0
     aRess2[i] += (M_u + Laplace_s2 - F_term_d ) * weight;  // C2^T*W + M*S2 = 0
     aResp[i] += (  M_s1 + alpha  * M_p ) * weight;  // C2^T*W + M*S2 = 0

      } // end phi_i loop

    } // end gauss point loop

    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store

   Res.resize(5 * nDofs,0.0);

    for (int i = 0; i < nDofs; i++) {
      Res[i]         = -aResu[i].value();
      Res[nDofs + i] = -aResv[i].value();

      Res[2 * nDofs + i  ] = -aRess1[i].value(); // s1
      Res[3 * nDofs + i  ] = -aRess2[i].value(); // s2
      Res[4 * nDofs + i  ] = -aResp[i].value(); // s2

    }

    RES->add_vector_blocked(Res, sysDof);

    Jac.resize(25 * nDofs * nDofs);

    // define the independent variables
    s.independent(&solu[0], nDofs);
    s.independent(&solv[0], nDofs);

    s.independent(&sols1[0], nDofs);
    s.independent(&sols2[0], nDofs);

    s.independent(&solp[0], nDofs);

        // define the dependent variables
    s.dependent(&aResu[0], nDofs);
    s.dependent(&aResv[0], nDofs);
    s.dependent(&aRess1[0], nDofs);
    s.dependent(&aRess2[0], nDofs);
    s.dependent(&aResp[0], nDofs);

    // get the jacobian matrix (ordered by column)
    s.jacobian(&Jac[0], true);

    KK->add_matrix_blocked(Jac, sysDof, sysDof);

         constexpr bool print_algebra_local = true;
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
