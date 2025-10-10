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
 * The system is assembled according to the matrix formulation:
 * [ M   B^T    0     0  ] [W]   [   0   ]
 * [ B    0    ν1C1  ν1C2] [U] = [-ν2F   ]
 * [ 0   C1^T   M     0  ] [S1]  [   0   ]
 * [ 0   C2^T   0     M  ] [S2]  [   0   ]
 *
 * using automatic differentiation
 **/

using namespace femus;


namespace karthik {
  
  class biharmonic_HM_with_decomposition {
    
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


//========= BOUNDARY_IMPLEMENTATION_s1 - BEGIN ==================

static void natural_loop_1ds1(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_s1,
                     const unsigned solFEType_s1,
                     std::vector< double > & Res
                    ) {

     double grad_s1_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s1.c_str(), grad_s1_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_s1_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_s1);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_s1_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3ds1(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_s1,
                       const unsigned solFEType_s1,
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
  std::vector <double> phi_s1_bdry;
  std::vector <double> phi_s1_x_bdry;

  phi_s1_bdry.reserve(max_size);
  phi_s1_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_s1_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s1.c_str(), grad_s1_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_s1 = msh->GetElementFaceDofNumber(iel, jface, solFEType_s1);

// dof-based - BEGIN
     std::vector< double > grad_s1_dot_n_at_dofs(n_dofs_face_s1);


    for (unsigned i_bdry = 0; i_bdry < grad_s1_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_s1_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_s1.c_str(), grad_s1_dot_n_at_dofs_temp, face, 0.);
     grad_s1_dot_n_at_dofs[i_bdry] = grad_s1_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_s1 ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_s1_bdry, phi_s1_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_s1_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_s1_bdry.size(); i_bdry ++) {
           grad_s1_dot_n_qp +=  grad_s1_dot_n_at_dofs[i_bdry] * phi_s1_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_s1; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_s1_dot_n_qp /*grad_u_dot_n*/  * phi_s1_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_s1 - END ==================



//========= BOUNDARY_IMPLEMENTATION_s2 - BEGIN ==================

static void natural_loop_1ds2(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_s2,
                     const unsigned solFEType_s2,
                     std::vector< double > & Res
                    ) {

     double grad_s2_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s2.c_str(), grad_s2_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_s2_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_s2);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_s2_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3ds2(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_s2,
                       const unsigned solFEType_s2,
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
  std::vector <double> phi_s2_bdry;
  std::vector <double> phi_s2_x_bdry;

  phi_s2_bdry.reserve(max_size);
  phi_s2_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_s2_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_s2.c_str(), grad_s2_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_s2 = msh->GetElementFaceDofNumber(iel, jface, solFEType_s2);

// dof-based - BEGIN
     std::vector< double > grad_s2_dot_n_at_dofs(n_dofs_face_s2);


    for (unsigned i_bdry = 0; i_bdry < grad_s2_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_s2_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_s2.c_str(), grad_s2_dot_n_at_dofs_temp, face, 0.);
     grad_s2_dot_n_at_dofs[i_bdry] = grad_s2_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_s2 ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_s2_bdry, phi_s2_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_s2_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_s2_bdry.size(); i_bdry ++) {
           grad_s2_dot_n_qp +=  grad_s2_dot_n_at_dofs[i_bdry] * phi_s2_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_s2; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_s2_dot_n_qp /*grad_u_dot_n*/  * phi_s2_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_s2 - END ==================



//========= BOUNDARY_IMPLEMENTATION_w - BEGIN ==================

static void natural_loop_1dw(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_w,
                     const unsigned solFEType_w,
                     std::vector< double > & Res
                    ) {

     double grad_w_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_w.c_str(), grad_w_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_w_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_w);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_w_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dw(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_w,
                       const unsigned solFEType_w,
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
  std::vector <double> phi_w_bdry;
  std::vector <double> phi_w_x_bdry;

  phi_w_bdry.reserve(max_size);
  phi_w_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_w_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_w.c_str(), grad_w_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_w = msh->GetElementFaceDofNumber(iel, jface, solFEType_w);

// dof-based - BEGIN
     std::vector< double > grad_w_dot_n_at_dofs(n_dofs_face_w);


    for (unsigned i_bdry = 0; i_bdry < grad_w_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_w_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_w.c_str(), grad_w_dot_n_at_dofs_temp, face, 0.);
     grad_w_dot_n_at_dofs[i_bdry] = grad_w_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_w ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_w_bdry, phi_w_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_w_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_w_bdry.size(); i_bdry ++) {
           grad_w_dot_n_qp +=  grad_w_dot_n_at_dofs[i_bdry] * phi_w_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_w; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_w_dot_n_qp /*grad_u_dot_n*/  * phi_w_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_w - END ==================






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


  // Solution, Global, s1 - BEGIN
  const std::string solname_s1 = ml_sol->GetSolName_string_vec()[1];
  const unsigned sols1Index = ml_sol->GetIndex(solname_s1.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_s1 = ml_sol->GetSolutionType(sols1Index);    // get the finite element type for "v"
  const unsigned sols1PdeIndex = mlPdeSys->GetSolPdeIndex(solname_s1.c_str());    // get the position of "v" in the pdeSys object

  // Solution, Global, s1 - END

  // Solution, Global, s2 - BEGIN
  const std::string solname_s2 = ml_sol->GetSolName_string_vec()[2];
  const unsigned sols2Index = ml_sol->GetIndex(solname_s2.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_s2 = ml_sol->GetSolutionType(sols2Index);    // get the finite element type for "v"
  const unsigned sols2PdeIndex = mlPdeSys->GetSolPdeIndex(solname_s2.c_str());    // get the position of "v" in the pdeSys object

  // Solution, Global, s2 - END

  // Solution, Global, w - BEGIN
  const std::string solname_w = ml_sol->GetSolName_string_vec()[3];
  const unsigned solwIndex = ml_sol->GetIndex(solname_w.c_str());    // get the position of "v" in the ml_sol object
  const unsigned solFEType_w = ml_sol->GetSolutionType(solwIndex);    // get the finite element type for "v"
  const unsigned solwPdeIndex = mlPdeSys->GetSolPdeIndex(solname_w.c_str());    // get the position of "v" in the pdeSys object
  // Solution, Global, w - END



  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27


  // Geometry, Local - BEGIN
  std::vector < std::vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  for (unsigned i = 0; i < dim; i++)    x[i].reserve(maxSize);
  // Geometry, Local - END


  // Solution, Local - BEGIN
  std::vector < adept::adouble >  solu; // local solution
  std::vector < adept::adouble >  sols1; // local solution
  std::vector < adept::adouble >  sols2; // local solution
  std::vector < adept::adouble >  solw; // local solution
  // reserve memory for the local standar vectors
  solu.reserve(maxSize);
  sols1.reserve(maxSize);

  sols2.reserve(maxSize);
  solw.reserve(maxSize);
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
  std::vector < adept::adouble > aRess1;
  std::vector < adept::adouble > aRess2;
  std::vector < adept::adouble > aResw;

  Res.reserve(4 * maxSize);

  aResu.reserve(maxSize);
  aRess1.reserve(maxSize);
  aRess2.reserve(maxSize);
  aResw.reserve(maxSize);

  std::vector < double > Jac; // local Jacobian matrix (ordered by column, adept)
  Jac.reserve(4 * maxSize * 4 * maxSize);
  // Equation, Local - END

  double weight; // gauss point weight


double nu =  0.7 /* Poisson ratio value */;
double nu1 = (4.0 * (1.0 - nu)) / (1.0 + nu);
double nu2 = 2.0 / (1.0 + nu);
// // // double nu2 = 1. - nu;



  for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);

// // //     unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_u);    // number of solution element dofs
    unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_s1);    // number of solution element dofs

    unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs

    std::vector<unsigned> Sol_n_el_dofs_Mat_vol(4, nDofs);

    // resize local arrays
    sysDof.resize(4 * nDofs);

    solu.resize(nDofs);
    sols1.resize(nDofs);
    sols2.resize(nDofs);
    solw.resize(nDofs);

    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofs2);
    }

    aResu.assign(nDofs, 0.);
    aRess1.assign(nDofs, 0.);
    aRess2.assign(nDofs, 0.);
    aResw.assign(nDofs, 0.);

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {

// // //       unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);    // global to global mapping between solution node and solution dof
      unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_s1);    // global to global mapping between solution node and solution dof



      solu[i]          = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
      sols1[i]        = (*sol->_Sol[sols1Index])(solDof);      // global extraction and local storage for the solution
      sols2[i]        = (*sol->_Sol[sols2Index])(solDof);      // s2  -> secondary row2, col2
      solw[i]        = (*sol->_Sol[solwIndex])(solDof);      // w  -> secondary row1, col2

      sysDof[i]             = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[nDofs + i]     = pdeSys->GetSystemDof(sols1Index, sols1PdeIndex, i, iel);
      sysDof[2 * nDofs + i] = pdeSys->GetSystemDof(sols2Index, sols2PdeIndex, i, iel); // s2
      sysDof[3 * nDofs + i] = pdeSys->GetSystemDof(solwIndex, solwPdeIndex, i, iel); // w

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

      msh->_finiteElement[ielGeom][solFEType_s1]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

      std::vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        for (unsigned jdim = 0; jdim < dim; jdim++) {
          xGauss[jdim] += x[jdim][i] * phi[i];
        }
      }

      // evaluate the solution, the solution derivatives in the gauss point
      adept::adouble soluGauss = 0.;
      std::vector < adept::adouble > soluGauss_x(dim, 0.);

      adept::adouble sols1Gauss = 0.;
      std::vector < adept::adouble > sols1Gauss_x(dim, 0.);

      adept::adouble sols2Gauss = 0.;
      std::vector < adept::adouble > sols2Gauss_x(dim, 0.);

      adept::adouble solwGauss = 0.;
      std::vector < adept::adouble > solwGauss_x(dim, 0.);


      for (unsigned i = 0; i < nDofs; i++) {
        soluGauss += phi[i] * solu[i];
        sols1Gauss += phi[i] * sols1[i];

        sols2Gauss += phi[i] * sols2[i];
        solwGauss += phi[i] * solw[i];

        for (unsigned jdim = 0; jdim < dim; jdim++) {
          soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
          sols1Gauss_x[jdim] += phi_x[i * dim + jdim] * sols1[i];

          sols2Gauss_x[jdim] += phi_x[i * dim + jdim] * sols2[i];
          solwGauss_x[jdim] += phi_x[i * dim + jdim] * solw[i];

        }
      }
      // *** phi_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {

        adept::adouble Laplace_u = 0.;
        adept::adouble Laplace_s1 = 0.;

        adept::adouble Laplace_s2 = 0.;
        adept::adouble Laplace_w = 0.;


        adept::adouble M_s1 = phi[i] * sols1Gauss;
        adept::adouble M_s2 = phi[i] * sols2Gauss;
        adept::adouble M_w = phi[i] * solwGauss;


        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * soluGauss_x[jdim];
          Laplace_s1   +=  - phi_x[i * dim + jdim] * sols1Gauss_x[jdim];

          Laplace_s2   +=  - phi_x[i * dim + jdim] * sols2Gauss_x[jdim];
          Laplace_w   +=  - phi_x[i * dim + jdim] * solwGauss_x[jdim];

        }

        double pi = acos(-1.);

    adept::adouble Bc1u = 0.;
    adept::adouble Bc2u = 0.;
    adept::adouble Bc1s1 = 0.;
    adept::adouble Bc2s2 = 0.;

// // //     if (dim == 2) {

        Bc1u += 0.5 * ( phi_x[i * dim + 1] * soluGauss_x[1] - phi_x[i * dim + 0] * soluGauss_x[0] ) ;
        Bc2u +=  0.5 * ( phi_x[i * dim + 0] * soluGauss_x[1] + phi_x[i * dim + 1] * soluGauss_x[0] ) ;
// //         Bu +=  phi_x[i * dim + 1] * soluGauss_x[1];

        Bc1s1 += nu1 * 0.5 * ( phi_x[i * dim] * sols1Gauss_x[0] - phi_x[i * dim + 1] * sols1Gauss_x[1] );
        Bc2s2 += nu1 * 0.5 * ( phi_x[i * dim + 1] * sols2Gauss_x[0] + phi_x[i * dim ] * sols2Gauss_x[1] );
// //         Byyw +=  phi_x[i * dim + 1] * solwGauss_x[1];


// // //     }
        adept::adouble F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

        // System residuals - signs adjusted to match matrix form
     aResu[i] += ( Bc1s1 + Bc2s2 + Laplace_w + nu2 * F_term ) * weight;
     aRess1[i] += (Bc1u + M_s1 ) * weight;
     aRess2[i] += ( Bc2u + M_s2 ) * weight;
     aResw[i] += (Laplace_u + M_w) * weight;

      } // end phi_i loop

    } // end gauss point loop

    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store

   Res.resize(4 * nDofs, 0.);

    for (int i = 0; i < nDofs; i++) {
      Res[i]               = - aResu[i].value();
      Res[nDofs + i]       = - aRess1[i].value();

      Res[2 * nDofs + i  ] = - aRess2[i].value(); // s2
      Res[3 * nDofs + i  ] = - aResw[i].value(); // w

    }

    RES->add_vector_blocked(Res, sysDof);

    Jac.resize(16 * nDofs * nDofs);

    // define the independent variables
    s.independent(&solu[0], nDofs);
    s.independent(&sols1[0], nDofs);

    s.independent(&sols2[0], nDofs);
    s.independent(&solw[0], nDofs);

        // define the dependent variables
    s.dependent(&aResu[0], nDofs);
    s.dependent(&aRess1[0], nDofs);
    s.dependent(&aRess2[0], nDofs);
    s.dependent(&aResw[0], nDofs);

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
