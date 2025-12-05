#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fields.h"
using namespace std;


template<int T_M, int T_N, int T_K>
__device__ void __forceinline__  matmul_t(const double* a, const double* b, double*C){
    for(int ni = 0; ni < T_N; ++ni){
      #pragma unroll
      for(int mi = 0; mi <T_M; ++mi){
      C[(ni*T_M + mi)]=0.0;
      }
    } 
    for (int mi = 0; mi < T_M; ++mi) {
        for (int ki = 0; ki < T_K; ++ki) {
            #pragma unroll
            for (int ni = 0; ni < T_N; ++ni) {
                C[ni* T_M + mi] +=  a[ki * T_M + mi] * b[T_K * ni + ki];
            }
        }
    }
}

 __device__ __forceinline__ const GpuElementType*gpu_ele_shape_vector(const GpuVectorField &field, const int ele_number)
{
 return  &field.mesh.shape; 
};
template<int NN>
__device__ void gpu_ele_nodes_mesh(const GpuMeshType &mesh, const int ele_number,int *ele_nodes){
  #pragma unroll 
  for(int index = 0; index < NN; ++index){
    ele_nodes[index]=mesh.ndglno[mesh.shape.loc*(ele_number-1) + index];
  }
}


template<int NN>
__device__ __forceinline__ void gpu_ele_nodes_scalar(const GpuScalarField &field, const int ele_number,int *ele_nodes){
  gpu_ele_nodes_mesh<NN>(field.mesh,ele_number,ele_nodes);
}


__device__ void print_val(const double *a,  int length){
for(int ll = 0; ll < length; ++ll){
printf("(%.5e)",*(a + ll));
}
printf("\n");
}
template<int NN>
__device__ __forceinline__ void gpu_ele_val_scalar(const GpuScalarField&field, 
    const int ele_number,double *ele_val){
    int nodes[NN];
    double val=0; 
    switch(field.field_type){
      case FIELD_TYPE_NORMAL:
      gpu_ele_nodes_scalar<NN>(field,ele_number,nodes);
      #pragma unroll
      for(int ii=0; ii < NN; ++ii){
          ele_val[ii]=field.val[(nodes[ii] - 1)];
      }
      break;
      case FIELD_TYPE_CONSTANT:
      val=field.val[0]; 
      #pragma unroll
      for(int ii=0; ii < NN; ++ii){
          ele_val[ii]=val;
      }
      break;
      default:
      break;
    } 
}

__device__ __forceinline__ const GpuElementType*gpu_ele_shape_scalar(const GpuScalarField &field, int ele_number)
{
  return  &field.mesh.shape; 
};

__device__ __forceinline__ void gpu_ele_val_at_quad_scalar(GpuScalarField &scalar_field,const int ele_number, double *quad_val)
{
   const GpuElementType *shape = gpu_ele_shape_scalar(scalar_field,ele_number);
   double ele_val[4];
   gpu_ele_val_scalar<4>(scalar_field,ele_number,&ele_val[0]);  
   matmul_t<1,11,4>(&ele_val[0], shape->n,quad_val);
}


__device__ __forceinline__ void gpu_ele_nodes_vector(const GpuVectorField&field, const int ele_number,
  int *ele_nodes){
  gpu_ele_nodes_mesh<4>(field.mesh,ele_number,ele_nodes);
}

__device__ __forceinline__ void gpu_ele_val_vector(const GpuVectorField &field, const int ele_number, double *__restrict__ ele_val){
    int nodes[4];
    switch(field.field_type){
      case FIELD_TYPE_NORMAL:
      gpu_ele_nodes_vector(field,ele_number,nodes);
      for(int ii=0; ii < field.mesh.shape.loc; ++ii){
          ele_val[ii * field.dim + 0]=field.val[(nodes[ii] -1)*field.dim + 0];
          ele_val[ii * field.dim + 1]=field.val[(nodes[ii] -1)*field.dim + 1];
          ele_val[ii * field.dim + 2]=field.val[(nodes[ii] -1)*field.dim + 2];
      }
      break;
      case FIELD_TYPE_CONSTANT:
      //for(int i = 0; i < field.dim;++i){
      //   for(int j=0; j < field.mesh.shape.loc; ++j){
      //    ele_val[j][i]=field.val[0][i]; 
      //}
      break;

      default:
      break;
    } 
}

__device__ __forceinline__  void gpu_ele_val_at_quad_vector(const GpuVectorField &field, const int ele_number,double *quad_val){
    //! Return the values of field at the quadrature points of ele_number.
    const GpuElementType *shape = gpu_ele_shape_vector(field,ele_number);
    double ele_val[3][4];
    gpu_ele_val_vector(field,ele_number,&ele_val[0][0]);  
    matmul_t<3,11,4>(&ele_val[0][0], shape->n,quad_val);
}

__device__ void gpu_ele_nodes_tensor(const GpuTensorField&field, const int ele_number,int *ele_nodes)
{
  gpu_ele_nodes_mesh<4>(field.mesh,ele_number,ele_nodes);
}


__device__ void gpu_ele_val_tensor(const GpuTensorField &field, const int ele_number, double *ele_val)
{ 
    //! Return the values of field at the nodes of ele_number.
    //real, dimension(field%dim(1), field%dim(2), field%mesh%shape%loc) :: ele_val
    int nodes[4];
    int ijdim=field.dim[0]*field.dim[1]; 

    switch(field.field_type){
    case FIELD_TYPE_NORMAL:
    gpu_ele_nodes_tensor(field,ele_number,nodes);
    //printf("node[0]=%d,nodes[1]=%d,node[2]=%d,node[3]=%d\n",nodes[0],nodes[1],nodes[2],nodes[3]);fflush(stdout);
    for(int k = 0; k < field.mesh.shape.loc;++k){ 
    for(int i = 0; i < field.dim[0];++i){
    for(int j = 0; j < field.dim[1]; ++j){
    ele_val[k * ijdim + i * field.dim[1] + j] =field.val[(nodes[k] - 1) * ijdim  + i*field.dim[1] + j];
    }
    }
    }
    //print_val(ele_val,3*3*4); 
    break;
    case FIELD_TYPE_CONSTANT:
    //to do 
    break;
    default:
    break; 

    }
}
__device__ const GpuElementType*gpu_ele_shape_tensor(const GpuTensorField &field, const int ele_number)
{
 return  &field.mesh.shape; 
};

__device__ void matmul_ld(const double* __restrict__ a, const double* __restrict__  b, double * __restrict__ C, const int m,const int n,const int k,
    const int lda,const int ldb,const int ldc){
    for(int ni = 0; ni < n; ++ni){
      for(int mi = 0; mi <m; ++mi){
      C[(ni*m + mi)*ldc]=0.0;
      }
    } 
    for (int mi = 0; mi < m; mi ++) {
        for (int ki = 0; ki < k; ki ++) {
            for (int ni = 0; ni < n; ni ++) {
                C[(ni* m + mi)*ldc] +=  a[(ki * m + mi)*lda] * b[(k * ni + ki)*ldb];
            }
        }
    }
}
template<int T_W,int T_M,int T_N, int T_K>
__device__ void gpu_tensormul_3_2(const double *a, const double *b, double *c){
for(int wi = 0; wi < T_W; ++wi){
 matmul_ld(a+wi,b,c+wi,T_M,T_N,T_K,T_W,1,T_W);
}
}
template<int T_dim, int T_loc, int T_ngi>
__device__ void gpu_ele_val_at_quad_tensor(const GpuTensorField &tensor_field,const int ele_number,double *quad_val){
    //! Return the values of field at the quadrature points of ele_number.
    const GpuElementType *shape = gpu_ele_shape_tensor(tensor_field,ele_number);
    double ele_val[T_dim][T_dim][T_loc];
    gpu_ele_val_tensor(tensor_field,ele_number,&ele_val[0][0][0]); 
   //to do gpu_tensormatmul
    gpu_tensormul_3_2<T_dim,T_dim,T_ngi,T_loc>(&ele_val[0][0][0],shape->n,quad_val);
}


__device__ void matrix_scale(double *a, const double scale,const int m, const int n){
  for(int mi = 0; mi < m; ++mi){
   for(int ni = 0; ni < n; ++ni){
    a[mi*n + ni]=scale * a[mi *n + ni];
   } 
  }
}
__device__ void matrix_transpose(const double *a, double *transpose, const int m, const int n){
  for(int mi = 0; mi < m; ++mi){
   for(int ni = 0; ni < n; ++ni){
     transpose[ni * m + mi]=a[mi*n + ni];
   } 
  }
  
}
__device__ void matrix_add(const double *a, const double *b,double *c, const int m, const int n){
  for(int mi = 0; mi < m; ++mi){
   for(int ni = 0; ni < n; ++ni){
     c[mi*n +ni]=a[mi *n +ni] + b[mi * n + ni];
   } 
  }
}
__device__ double matrix_sum(double *a,const int m,const int n){
  double m_sum=0.0;
   for(int mi = 0; mi < m; ++mi){
   for(int ni = 0; ni < n; ++ni){
     m_sum+= a[mi*n + ni];
    }
   } 
return m_sum;
}
__device__ double matrix_two(double *a,const int m,const int n){
  double m_sum=0.0;
   for(int mi = 0; mi < m; ++mi){
   for(int ni = 0; ni < n; ++ni){
     a[mi*n + ni] *=a[mi*n + ni];
    }
   } 
return m_sum;
}

__device__ void gpu_les_viscosity_strength(const double*du_t, const double *relu, const int nloc, const int ngi, const int dim,double *les_vis_strength)
{
  double s[3][3];
  double st[3][3];
  for(int gi = 0; gi < ngi; ++gi){
  for(int di = 0; di < dim; ++di){
  matmul_ld(relu, du_t + di*ngi*nloc + gi * nloc, &s[di][0], 3,1,4,1,1,1); 
  } 
  matrix_scale(&s[0][0],0.5,3,3);
  matrix_transpose(&s[0][0],&st[0][0],3,3);
  matrix_add(&s[0][0],&st[0][0],&s[0][0],3,3);
  matrix_two(&s[0][0],3,3);
  double m_sum = matrix_sum(&s[0][0],3,3);
  double vis = std::sqrt(2*m_sum);
  les_vis_strength[gi]=vis;
  }
}

void outer_product(double *a, double *b, double *c, 
  int m, int n, int lda, int ldb, int ldc){
  for(int mi = 0; mi < m; ++mi){
    for(int ni = 0; ni < n; ++ni){
      c[mi*n + ni]=a[mi*lda]*b[ni*ldb];
    }
  }
}
__device__ double matrix_sum_diag(const double *a, const int m, const int n){
  double r = 0.0;
  for(int mi = 0; mi < m; ++mi){
    r +=a[mi * n + mi];
  }
  return r;
}

__device__ void gpu_length_scale_tensor(const double *du_t,
    const double *shape_n, const int ngi, const  int nloc, double *result_t){
//Computes a length scale tensor to be used in LES (units are in length^2)
//derivative of velocity shape function (nloc x ngi x dim)
//real, dimension(:,:,:), intent(in):: du_t
//!! the resulting tensor (dim x dim x ngi)
//real, dimension(size(du_t,3),size(du_t,3),size(du_t,2)) :: t
//!! for a simplex if degree==1 the tensor is the same for all gaussian points
  double M[3][3];
  double r=0.0;
  int compute_ngi=0;
  memset(result_t,0,sizeof(double)*3*3*ngi);
  // if (.not.(shape%degree==1 .and. shape%numbering%family==FAMILY_SIMPLEX))
  // to do
  // for linear
  compute_ngi=1;
  for(int gi = 0; gi < compute_ngi; ++gi){
    for(int loc = 0; loc < nloc; ++loc){
      //outer_product
     for(int ni = 0; ni < 3; ++ni){
      //for(int mi = 0; mi < 3; ++mi){
     //   for(int ni = 0; ni < 3; ++ni){
          M[ni][0] = du_t[gi * nloc + loc + nloc * ngi *0] * du_t[gi * nloc + loc + nloc * ngi *ni];
          M[ni][1] = du_t[gi * nloc + loc + nloc * ngi *1] * du_t[gi * nloc + loc + nloc * ngi *ni];
          M[ni][2] = du_t[gi * nloc + loc + nloc * ngi *2] * du_t[gi * nloc + loc + nloc * ngi *ni];
       //   printf("(%.5e,%.5e)",du_t[gi * nloc + loc + nloc * ngi *mi],du_t[gi * nloc + loc + nloc * ngi *ni]);
       // }
      }
      //printf("*********M*********\n");fflush(stdout);
      //print_val(&M[0][0],3*3);
      //printf("*********end M*********\n");fflush(stdout);
     // outer_product( du_t + gi * nloc + loc, du_t + gi * nloc + loc,
      //               &M[0][0],3,3, nloc*ngi, nloc*ngi,1);
      r = matrix_sum_diag(&M[0][0],3,3);
     double inverserr=(1.0/(r*r));
     // printf("r=%.5e\n",r);fflush(stdout);
      if( r != 0.0){
        for(int mi = 0; mi < 3; ++mi){
          //for(int ni = 0; ni < 3; ++ni){
            result_t[gi*3*3 + mi * 3 + 0]=result_t[gi*3*3 + mi * 3 + 0] + inverserr * M[mi][0];
            result_t[gi*3*3 + mi * 3 + 1]=result_t[gi*3*3 + mi * 3 + 1] + inverserr * M[mi][1];
            result_t[gi*3*3 + mi * 3 + 2]=result_t[gi*3*3 + mi * 3 + 2] + inverserr * M[mi][2];
          //}
        }
      }
    }
  }
  //copy the rest
  for(int gi = compute_ngi; gi < ngi; ++gi){
    for(int mi = 0; mi < 3; ++mi){
      //for(int ni = 0; ni < 3; ++ni){
        result_t[gi*3*3 + mi * 3 + 0]=result_t[mi * 3 + 0];
        result_t[gi*3*3 + mi * 3 + 1]=result_t[mi * 3 + 1];
        result_t[gi*3*3 + mi * 3 + 2]=result_t[mi * 3 + 2];
      //}
    }
  }
}



__device__ void gpu_les_add_density(double *les_tensor_gi,const double *density_gi,
        const double *les_coef_gi, const double smagorinsky_coefficient,
        double *viscosity_gi,const int dim0, const int dim1, const int ngi)
{
  for(int gi = 0; gi < ngi; ++gi){
    for(int di1 = 0; di1 < dim1; ++di1){
      for(int di0 = 0; di0 < dim0; ++di0){
        les_tensor_gi[gi*dim0*dim1 + di1 * dim0 + di0]=4.0*les_tensor_gi[gi*dim0*dim1 + di1 * dim0 + di0]
            *density_gi[gi]*les_coef_gi[gi]*(smagorinsky_coefficient*smagorinsky_coefficient);
      }
    }
  }
///viscosity_gi=viscosity_gi+les_tensor_gi
  for(int gi = 0; gi < ngi; ++gi){
    for(int di1 = 0; di1 < dim1; ++di1){
      for(int di0 = 0; di0 < dim0; ++di0){
      viscosity_gi[gi*dim0*dim1 + di1 * dim0 + di0]
      +=les_tensor_gi[gi*dim0*dim1 + di1 * dim0 + di0];
      }
     }
   }
}
__device__ double dot_product(const double* __restrict__ a, const double* __restrict__  b, const int m){
  double r=0.0;
  for(int i=0;i<m;i++){
    r += a[i]*b[i];
  }
  return r;
}
template<int M>
__device__ double dot_product_t(const double* __restrict__ a, const double* __restrict__ b){
  double r=0.0;
  for(int i=0;i<M;i++){
    r += a[i]*b[i];
  }
  return r;
}
__device__ void dshape_matmul(const double* a, const double* b, double*C,const  int m,const int n,const int k){
    for(int ni = 0; ni < n; ++ni){
      for(int mi = 0; mi <m; ++mi){
      C[(ni*m + mi)]=0.0;
      }
    } 

  for (int mi = 0; mi < m; mi ++) {
    for (int ki = 0; ki < k; ki ++) {
      for (int ni = 0; ni < n; ni ++) {
        C[ni* m + mi] +=  a[ki * m + mi] * b[k * ni + ki];
      }
    }
  }
}
template<int T_dim, int T_loc1, int T_loc2, int T_ngi>
__device__ void gpu_dshape_tensor_dshape_t(const double *dshape1, const double *tensor,const  double *dshape2,
   const double *detwei, double *r){
   double tmp[T_dim]={0.0},mat_mul[T_dim]={0.0},dotproduct=0.0;
   for(int gi=0; gi< T_ngi;++gi){
     double gidetwei=detwei[gi];
     for(int iloc=0;iloc<T_loc1;iloc++){
       #pragma unroll
       for(int n=0;n<T_dim;++n){
         tmp[n]=dshape1[n*T_ngi*T_loc1+gi*T_loc1+iloc];
       }
       matmul_t<1,3,3>(tmp,&tensor[gi*T_dim*T_dim],mat_mul); 
       for(int jloc=0;jloc<T_loc2;jloc++){
         #pragma unroll
         for(int n=0;n<T_dim;++n){
           tmp[n]=dshape2[n*T_ngi*T_loc2+gi*T_loc2+jloc];
         }
         dotproduct = dot_product_t<T_dim>(mat_mul,tmp);
         r[jloc*T_loc1+iloc]=r[jloc*T_loc1+iloc]+dotproduct*gidetwei;
       }
     }
   }
 }

__device__ void gpu_viscosity_mat(const double *du_t, const double *viscosity_gi,const  double *detwei,const int dim, const int loc,const  int ngi, const double dt,const  double theta,
     double *viscosity_mat,double *big_m_tensor_addto){
//real, dimension(u%dim, u%dim, ele_loc(u, ele), ele_loc(u, ele))   :: viscosity_mat
//real, dimension(ele_loc(u, ele), ele_ngi(u, ele), u%dim), intent(in)  :: du_t
    for(int di = 0; di < dim; ++di){
      double result[16]={0.0};
      gpu_dshape_tensor_dshape_t<3,4,4,11>(du_t,viscosity_gi,du_t,detwei,&result[0]);
      for(int loc2=0; loc2<loc; ++loc2){
        for(int loc1=0; loc1<loc; ++loc1){
        viscosity_mat[loc2 * loc * dim * dim + loc1 * dim * dim + di*dim + di]
        =result[loc2 * loc + loc1];
      }
    }
  }
  //big_m_tensor_addto = big_m_tensor_addto + dt*theta*viscosity_mat
  double dtxtheta=dt*theta;
  for(int loc2=0; loc2<loc;++loc2){
    for(int loc1=0; loc1<loc;++loc1){
      for(int di2=0; di2<dim; ++di2){
        for(int di1=0; di1<dim;++di1){
          big_m_tensor_addto[loc2 * loc * dim * dim + loc1 * dim * dim + di2 * dim + di1]
            += dtxtheta * viscosity_mat[loc2 * loc * dim * dim + loc1 * dim * dim + di2 * dim + di1]; 
        }
      }
    }
  }
}                  

__device__ void gpu_vis_rhs_addto(const double *viscosity_mat, double *rhs_addto,const double *oldu_val,
    const  int dim, const int loc){
//rhs_addto(dim, :) = rhs_addto(dim, :) - matmul(viscosity_mat(dim,dim,:,:), oldu_val(dim,:))
  for(int di = 0; di < dim; ++di){
   double result[4]={0.0};
   matmul_ld(viscosity_mat + di * dim + di, oldu_val + di, &result[0],4,1,4,dim*dim,dim,1);
   for(int nloc = 0; nloc < loc; ++nloc){
     rhs_addto[nloc *dim + di]=rhs_addto[nloc *dim + di] - result[nloc];
   }
  }     
}

__device__ void gpu_shape_shape(double *shape1n,double *shape2n,double *detwei,
double *shape_shape){
double local[11],local2[11];
for(int iloc=0; iloc < 4; ++iloc){
  for(int gi = 0; gi < 11; ++gi){ 
   local[gi]=shape1n[gi*4 + iloc];
  }
  for(int jloc=0; jloc < 4;++jloc){
  for(int gi=0; gi < 11;++gi){
    local2[gi]=local[gi]*shape2n[gi*4+jloc];
  } 
  shape_shape[jloc*4+iloc]=dot_product(local2,detwei,11); 
  } 
 }
}

__device__ void gpu_ele_val_vector_dim(const GpuVectorField &field,int val_dim, const int ele_number, double *ele_val){
    int nodes[4];
    switch(field.field_type){
      case FIELD_TYPE_NORMAL:
      gpu_ele_nodes_vector(field,ele_number,nodes);
      for(int ii=0; ii < field.mesh.shape.loc; ++ii){
      for(int jj = (val_dim); jj < (val_dim+1); ++jj){
          ele_val[ii]=field.val[(nodes[ii] -1)*field.dim + jj];
        } 
      }
      break;
      case FIELD_TYPE_CONSTANT:
      break;

      default:
      break;
    } 
}

__device__ void gpu_ele_div_at_quad_vector(GpuVectorField &field, int ele_number,double *du_t,double *result){
for(int di = 0; di < 3; ++di){
double ele_val_dim[4];
gpu_ele_val_vector_dim(field,di,ele_number,&ele_val_dim[0]);
double one_quad_div[11]={0.0};
matmul_t<1,11,4>(ele_val_dim,&du_t[di*4*11],&one_quad_div[0]);

for(int gi=0; gi < 11; ++gi){
  result[gi]+=one_quad_div[gi];
}
}
}

__device__ void matrix_sum_row(double *a,int m, int n, double *result){

for(int mi = 0; mi < m; ++mi){
  double sum=0.0;
  for(int ni = 0; ni < n; ++ni){
   sum+=a[ni*m + mi]; 
  }
  result[mi]=sum;
}

} 
__device__ void gpu_shape_vector_dot_dshape(double *shape, double *vec, double *dshape,
  double *detwei, double *result){
double vecT[3][11];
double sum_row[11];
for(int iloc = 0; iloc < 4; ++iloc){
  for(int jloc= 0; jloc < 4; ++jloc){
   matrix_transpose(vec,&vecT[0][0],11,3);   
   for(int ii = 0; ii < 3; ++ii){
    for(int jj = 0; jj < 11; ++jj){
     vecT[ii][jj]*=dshape[jloc + jj*4 + ii *4*11]; 
    } 
   }
   
   matrix_sum_row(&vecT[0][0],11,3,sum_row);

   for(int gi=0; gi < 11; ++gi){
   sum_row[gi]*=shape[4 * gi + iloc];
   } 
   result[jloc*4 + iloc]=dot_product(sum_row,detwei,11);
   }
}
}


__device__ void gpu_vector_field_dim_addto(GpuVectorField &field,int idim,int *nodes,
  double *val){

  for(int jj = 0; jj < 4; ++jj){
  // field.val[idim + (nodes[jj] -1) * field.dim] += val[jj];
   atomicAdd(&field.val[idim + (nodes[jj] -1) * field.dim],val[jj]);
  }

}

__device__ void gpu_shape_dshape(double *shapen,double *dshape,double *detwei,
double *shape_dshape){
//dshape(4,11,3) shape(4,11)
//shape_dshape(3,4,4)
const int loc = 4;
const int dim=3;
double tmp[11][4];

for(int iloc = 0; iloc < loc; ++iloc){
  for(int jj = 0; jj < 11; ++jj){
     tmp[jj][iloc] = detwei[jj]*shapen[ jj *4 + iloc];
  }
}
double local[11];
for(int jloc = 0; jloc < 4; ++jloc){
  for(int idim = 0; idim < dim; ++idim){
    for(int gi=0; gi < 11; ++gi){
      local[gi]=dshape[jloc + gi * 4 + idim * 11 * 4];
    }

    for(int iloc  = 0; iloc < 4; ++iloc){
    double val=0.0;
      for(int kloc=0; kloc < 11; ++kloc){
       val += local[kloc] *tmp[kloc][iloc]; 
      } 
     shape_dshape[idim + iloc *3 + jloc *3 *4]=val;
    }  
 }
}

}



__device__ void gpu_shape_rhs(double *shape_n, double *detwei,double*rhs){
matmul_t<4,1,11>(shape_n,detwei,rhs);
}


__device__ void gpu_adv_matrix_addto(double*addto, double*newmat){
for(int ii = 0; ii < 4; ++ii){
#pragma unroll
for(int jj = 0; jj < 4; ++jj){
addto[ii*4 + jj] += newmat[ii*4 + jj];
}
}
}
__device__ void gpu_shape_shape_vector(double *shape1,double *shape2,
                double *detwei, double *vectorval,double *shape_shape_vector){
  const int loc = 4;
  const int dim=3;
  const int ngi=11;
  double shape_shape[11];
  for(int iloc = 0; iloc < loc; ++iloc){
    for(int jloc = 0; jloc < loc; ++jloc){
     for(int gi = 0; gi < ngi; ++gi){
      shape_shape[gi]=shape1[gi * 4 + iloc] * shape2[gi * 4 + jloc];
     } 
     for(int ndim = 0; ndim < dim; ++ndim){
       double result=0.0;
       for(int gi = 0; gi < ngi; ++gi){
         result +=shape_shape[gi]*vectorval[gi*3 + ndim]*detwei[gi]; 
       }
       shape_shape_vector[jloc * 4 * 3 + iloc * 3 + ndim]=result;
     }  
    }
  } 
}

__device__ void gpu_ele_grad_at_quad_scalar(GpuScalarField &field,int ele_number,double *dn, double *result){
  double ele_val[4];
  gpu_ele_val_scalar<4>(field,ele_number,ele_val);
  for(int ndim = 0; ndim < 3; ++ndim){
    for(int gi = 0; gi < 11; ++gi){
      double sum=0.0; 
      for(int iloc=0; iloc < 4; ++iloc){
        sum += ele_val[iloc] * dn[ndim*11*4 + gi*4 + iloc]; 
      }
      result[gi*3 + ndim]=sum;
    } 
  }
}
 

#endif 


