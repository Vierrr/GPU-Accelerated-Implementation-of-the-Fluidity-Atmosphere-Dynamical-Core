
#ifndef CUDA_GLOBAL_H
#define CUDA_GLOBAL_H
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fields.h"
#include "cuda_kernels.h"
using namespace std;

__global__ void gpu_add_viscosity_element(GpuTensorField viscosity_field, GpuVectorField nu_field,
GpuVectorField oldu_field, GpuVectorField u_field,GpuScalarField density_field, 
int ele_num, double dt,double theta, double smagorinsky_coefficient,
double *du_t,  double *detwei, double *big_m_tensor_addto,double *rhs_addto){
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
// if(thread_idx !=0)return;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;
  
  double viscosity_gi[11][3][3];
  gpu_ele_val_at_quad_tensor<3,4,11>(viscosity_field,thread_idx,&viscosity_gi[0][0][0]);
  
  double nu_ele[4][3];
  gpu_ele_val_vector(nu_field,thread_idx,&nu_ele[0][0]);

  double les_coef_gi[11];
  gpu_les_viscosity_strength(&du_t[(thread_idx-1) * 4*11*3 ], &nu_ele[0][0],4,11,3,&les_coef_gi[0]);

  double density_gi[11];
  gpu_ele_val_at_quad_scalar(density_field,thread_idx,&density_gi[0]);

  double les_tensor_gi[11][3][3];
  gpu_length_scale_tensor(&du_t[(thread_idx-1) * 4*11*3],u_field.mesh.shape.n,
  u_field.mesh.shape.ngi,u_field.mesh.shape.loc,&les_tensor_gi[0][0][0]);
  
  gpu_les_add_density(&les_tensor_gi[0][0][0],&density_gi[0],
  &les_coef_gi[0],smagorinsky_coefficient,&viscosity_gi[0][0][0],
  viscosity_field.dim[0], viscosity_field.dim[1], viscosity_field.mesh.shape.ngi);
  
  double viscosity_mat[4][4][3][3];
  gpu_viscosity_mat(&du_t[(thread_idx-1) * 4*11*3], &viscosity_gi[0][0][0], 
  &detwei[(thread_idx-1)*11], u_field.dim, u_field.mesh.shape.loc,u_field.mesh.shape.ngi,dt, theta,
  &viscosity_mat[0][0][0][0],&big_m_tensor_addto[(thread_idx-1)*4*4*3*3]);
  
  double oldu_val[4][3];
  gpu_ele_val_vector(oldu_field,thread_idx,&oldu_val[0][0]);
  gpu_vis_rhs_addto(&viscosity_mat[0][0][0][0],&rhs_addto[(thread_idx-1)*4*3],&oldu_val[0][0],3,4);

}

__global__ void gpu_add_sources(GpuScalarField density_field,GpuVectorField source_field,GpuVectorField u_field,int ele_num, double *detwei, double *rhs_addto){

 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

  double density_gi[11];
  gpu_ele_val_at_quad_scalar(density_field,thread_idx,&density_gi[0]);
  for(int gi=0; gi < 11;++gi){
   density_gi[gi]=detwei[(thread_idx - 1)*11+gi]*density_gi[gi]; 
  }
 double source_mat[4][4];
 double *shapen=gpu_ele_shape_vector(source_field,thread_idx)->n;
 double *test_function=gpu_ele_shape_vector(u_field,thread_idx)->n;

 gpu_shape_shape(test_function,shapen,density_gi,&source_mat[0][0]);
 
 for(int di = 0; di < 3; ++di){
  double result[4];
  double source_ele_val_dim[4];

  gpu_ele_val_vector_dim(source_field,di,thread_idx,source_ele_val_dim);
   
  matmul_t<4,1,4>(&source_mat[0][0],&source_ele_val_dim[0], &result[0]);
   for(int nloc = 0; nloc < 4; ++nloc){
     rhs_addto[(thread_idx-1)*3*4 + nloc *3 + di]=rhs_addto[(thread_idx - 1) *3*4 + nloc *3 + di] + result[nloc];
   }
  }  

} 

__global__ void gpu_add_advection(GpuVectorField nu_field,GpuScalarField density_field, GpuVectorField u_field, GpuVectorField oldu_field, 
int ele_num, double beta,double dt, double theta,double *du_t, double *detwei,double *big_m_tensor_addto, double *rhs_addto)
{
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
// if(thread_idx != 0)return;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

  double relu_gi[11][3];
  gpu_ele_val_at_quad_vector(nu_field,thread_idx,&relu_gi[0][0]);
  
  double density_gi[11];
  gpu_ele_val_at_quad_scalar(density_field,thread_idx,&density_gi[0]);
  
  double div_relu_gi[11]={0.0};
  gpu_ele_div_at_quad_vector(nu_field,thread_idx,&du_t[(thread_idx - 1)*4*11*3],&div_relu_gi[0]); 
 
  double coefficient_detwei[11]; 
  for(int gi = 0; gi < 11; ++gi){
    coefficient_detwei[gi]=density_gi[gi] * detwei[(thread_idx - 1)*11 + gi];
  } 
 
 double *u_shapen=gpu_ele_shape_vector(u_field,thread_idx)->n;
 double *test_function=gpu_ele_shape_vector(u_field,thread_idx)->n;
 double advection_mat_1[4][4];
 
 gpu_shape_vector_dot_dshape(test_function,&relu_gi[0][0],&du_t[(thread_idx-1)*4*11*3],
  coefficient_detwei,&advection_mat_1[0][0]); 
for(int gi=0; gi < 11; ++gi){
  coefficient_detwei[gi]=div_relu_gi[gi]*detwei[(thread_idx -1)*11 + gi]*density_gi[gi];
 }
 double advection_mat_2[4][4];
 gpu_shape_shape(test_function,u_shapen,coefficient_detwei,&advection_mat_2[0][0]);

 matrix_scale(&advection_mat_2[0][0],beta,4,4);
 matrix_add(&advection_mat_1[0][0],&advection_mat_2[0][0],&advection_mat_1[0][0],4,4);
 
 for(int di = 0; di < 3; ++di){
 for(int wi = 0; wi < 4; ++wi){
#pragma unroll
 for(int ki = 0; ki < 4; ++ki){
 big_m_tensor_addto[(thread_idx-1)*3*3*4*4 + wi * 4*3*3 + ki *3*3 + di *3 + di ] +=dt*theta*advection_mat_1[wi][ki]; 
 }
 } 
 double val_result[4];
 gpu_ele_val_vector_dim(oldu_field,di,thread_idx,val_result); 
 double result[4];
 matmul_t<4,1,4>(&advection_mat_1[0][0],val_result,result);
 for(int ii=0; ii < 4;++ii){
   rhs_addto[(thread_idx-1)*4*3 + ii * 3 + di] -=result[ii];
 }
 }
}

__global__ void gpu_add_mass_element(GpuVectorField u_field,GpuScalarField density_field, GpuVectorField inverse_masslump_field, int ele_num, double *detwei, double *big_m_diag_addto)
{
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
// if(thread_idx != 0)return;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

  double density_gi[11];
  gpu_ele_val_at_quad_scalar(density_field,thread_idx,&density_gi[0]);
  
  double coefficient_detwei[11]; 
  for(int gi = 0; gi < 11; ++gi){
    coefficient_detwei[gi]=density_gi[gi] * detwei[(thread_idx - 1)*11 + gi];
  } 
 double *u_shapen=gpu_ele_shape_vector(u_field,thread_idx)->n;
 double *test_function=gpu_ele_shape_vector(u_field,thread_idx)->n;

 double mass_mat[4][4];
 gpu_shape_shape(test_function,u_shapen,coefficient_detwei,&mass_mat[0][0]);
 double mass_lump[4];
 matrix_sum_row(&mass_mat[0][0],4,4,mass_lump);
 
 for(int di = 0; di < 3; ++di){
  #pragma unroll
  for(int ll=0; ll < 4; ++ll){
   big_m_diag_addto[(thread_idx-1)*3*4 + ll*3 + di] += mass_lump[ll]; 
  } 
 }

 int nodes[4];
 gpu_ele_nodes_vector(u_field,thread_idx,nodes); 
 for(int idim = 0; idim < 3; ++idim){
   gpu_vector_field_dim_addto(inverse_masslump_field,idim,nodes,mass_lump); 
 }  
}

template<int T_dim, int T_loc,int T_ngi>
 __global__ void transform_to_physical_full_detwei(double*field_val,double* shape_dn,double* lx_shape_dn,
double* quadrature_weight, int *ndglno, int elecount,
double* dshape,double* detwei){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=elecount)return;

    bool x_spherical = false;
    bool x_nonlinear = false;
    bool m_nonlinear = false;
    
    constexpr int dshape_size = T_dim*T_ngi*T_loc;
//先用寄存器吧，也可以看看能否复用
    double J_local_T[T_dim*T_dim]={0.0};
    double invJ_local[T_dim*T_dim]={0.0};
    double detJ_local=0.0;

//--------实现x_val = ele_val_vec(X，ele)---------------
//ele_val(i, :) = field%val(i,nodes) i这维是3，第二维是4
//field val中是将每个node的三个坐标连着存
    
    int index[T_loc]={0};//循环展开，128bits可以有什么优化读取嘛
    #pragma unroll
    for(int iloc=0; iloc < T_loc; ++iloc){
    index[iloc]= ndglno[idx*T_loc + iloc]-1;
    }

    double x_val[T_dim*T_loc]={0.0};//fortran dim(3,4) 一列的3个元素连续存
    for(int iloc=0;iloc<T_loc;++iloc){
    #pragma unroll
    for(int d=0; d < T_dim; ++d){
        x_val[iloc*T_dim+d] = field_val[index[iloc]*T_dim + d];
    }
    }
    
    if(x_spherical){
    }

    double dn[T_dim*T_loc];//dn[dim][loc] :loc行，dim列，所有ele统一
    double dshape_local[T_dim*T_loc]; 
    //若是x linear或m linear需要在gi维度多次计算dshape
    //每次计算记得初始化
    
    int cyc3[5]={0,1,2,0,1};//{1,2,3,1,2};
    for(int gi = 0; gi<T_ngi; ++gi){
        if ((x_nonlinear)||gi==0){
            for(int i=0;i<T_dim;++i){
                #pragma unroll
                for(int j=0;j<T_loc;++j){
                    dn[i*T_loc+j] = lx_shape_dn[i*T_loc*T_ngi+gi*T_loc+j];
                }
            }//如果一会是m_nonlinear ngi维度每个都要取一次

            matmul_t<T_dim,T_dim,T_loc>(x_val,dn,J_local_T);//算J_local_T
            if (x_spherical){
            }
            if(T_dim==0){
            }
            if(T_dim==2){
            }
            if(T_dim==3){
                for(int j=0;j<T_dim;++j){
                 #pragma unroll
                    for(int i=0;i<T_dim;++i){
                        invJ_local[j*T_dim+i]=J_local_T[cyc3[j+1]*T_dim+cyc3[i+1]]*J_local_T[cyc3[j+2]*T_dim+cyc3[i+2]]
                                        -J_local_T[cyc3[j+1]*T_dim+cyc3[i+2]]*J_local_T[cyc3[j+2]*T_dim+cyc3[i+1]];
                    }
                }
            }
            //detJ_local=dot_product(J_local_T(:,1),invJ_local(:,1))只算前3个数
            detJ_local=dot_product_t<T_dim>(J_local_T,invJ_local);
            double invlocal = 1.0/detJ_local;
            for(int i=0;i<T_dim;++i){
              #pragma unroll
              for(int j=0;j<T_dim;++j){
                 invJ_local[j*T_dim+i] = invJ_local[j*T_dim+i]*invlocal;
              }
           }
        }
        if((x_nonlinear)||(m_nonlinear)||gi==0){
            for(int i=0;i<T_dim;++i){
                for(int j=0;j<T_loc;++j){
                    //之前的dn用来装lx_shape_dn，计算完invJ就可以腾出
                    //在下面的计算里 dn要使用不连续存的第二维xyz，所以在这里把xyz的dim维先存
                    dn[j*T_dim+i] = shape_dn[i*T_loc*T_ngi+gi*T_loc+j];
                    dshape_local[i*T_loc+j]=0.0;  
                    //若是x linear或m linear需要在gi维度多次计算dshape
                    //每次计算记得初始化
                }
            }//如果一会是m_nonlinear ngi维度每个都要取一次
            for(int j=0;j<T_loc;++j){
                //比较麻烦的是dshape_local是先存的dim维，所以之后写回
                matmul_t<T_dim,1,T_dim>(invJ_local,&dn[j*T_dim],&dshape_local[j*T_dim]);
                //3*3 3*1--->3*1
            }

        }else{
            //就直接用前面dshape_local的结果 不操作可以直接删除
        }
        for(int i=0;i<T_dim;++i){
            for(int j=0;j<T_loc;++j){
                dshape[idx*dshape_size+i*T_loc*T_ngi+gi*T_loc+j]=dshape_local[j*T_dim+i];
            }
        }
    //计算detwei   
    //quadrature_weight就是个一维数组 ngi个元素
    //detwei[k] = quadrature_weight[k];
    detwei[idx*T_ngi+gi] = fabs(detJ_local)*quadrature_weight[gi];
    }
}



__global__ void gpu_grad_mat(GpuScalarField ct_rhs_field,
int ele_num, double *grad_p_u_mat,double *du_t, double *detwei){

 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
// if(thread_idx != 0)return;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;
 
double *p_shapen=gpu_ele_shape_scalar(ct_rhs_field,thread_idx)->n;
 gpu_shape_dshape(p_shapen,&du_t[(thread_idx - 1)*4*11*3],
     &detwei[(thread_idx-1)*11],&grad_p_u_mat[(thread_idx-1)*3*4*4]);

return;
}

__global__ void gpu_add_diagonal_to_tensor(int ele_num,
double *big_m_diag_addto, double *big_m_tensor_addto){
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
// if(thread_idx != 0)return;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;
const int dim = 3, nloc = 4;
double *ele_diag = big_m_diag_addto + (thread_idx - 1)*dim*nloc;
double *ele_tensor = big_m_tensor_addto + (thread_idx - 1)*dim*dim*nloc*nloc;

for(int i = 0; i < nloc; ++i){
for(int j = 0; j < dim; ++j){
   *(ele_tensor +i * (dim * dim * nloc) + i*(dim*dim) + j * dim + j ) += *(ele_diag + i * dim + j);
  }
}

}

__global__ void gpu_rhs_addto_kernel(GpuVectorField field, const int ele_num, double *gpu_ele_val){
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
 if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

 int nodes[4];
 gpu_ele_nodes_vector(field,thread_idx,nodes); 
double *ptr = gpu_ele_val + (thread_idx-1) * 4*3;
  for(int ii = 0; ii < 4;++ii){
   atomicAdd(&(field).val[(nodes[ii] -1)*3 + 0],*(ptr+ii*3 + 0));
   atomicAdd(&(field).val[(nodes[ii] -1)*3 + 1],*(ptr+ii*3 + 1));
   atomicAdd(&(field).val[(nodes[ii] -1)*3 + 2],*(ptr+ii*3 + 2));
  }
}

//    rhs_addto = rhs_addto + shape_rhs(test_function, detwei * ele_val_at_quad(source, ele))
__global__ void gpu_adv_add_source(GpuScalarField t_field,GpuScalarField source_field,int ele_num,double *detwei,double*rhs_addto){
int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
if(thread_idx>=ele_num)return;
thread_idx+=1;
double rhs[4]={0};

double *test_function=gpu_ele_shape_scalar(t_field,thread_idx)->n;
double source_gi[11];
gpu_ele_val_at_quad_scalar(source_field,thread_idx,&source_gi[0]);
for(int i = 0; i < 11; ++i){
  source_gi[i]=source_gi[i]*detwei[(thread_idx - 1)*11 + i];
}
gpu_shape_rhs(test_function,source_gi,rhs);
for(int jj = 0; jj < 4; ++jj){
//atomicAdd(&rhs_addto[(thread_idx-1)*4 + jj],rhs[jj]);
rhs_addto[(thread_idx-1)*4 + jj]+=rhs[jj];
}
}
///mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei)


__global__ void gpu_adv_add_mass(GpuScalarField t_field,int ele_num, double *detwei,double *matrix_addto){
int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
if(thread_idx>=ele_num)return;
thread_idx+=1;
double massmatrix[4][4]={0};
double *test_function=gpu_ele_shape_scalar(t_field,thread_idx)->n;
gpu_shape_shape(test_function,test_function,detwei+(thread_idx-1) * 11, &massmatrix[0][0]);
gpu_adv_matrix_addto(matrix_addto + 16*(thread_idx - 1),&massmatrix[0][0]);
}

//dshape_tensor_dshape
template<int T_dim, int T_loc1, int T_loc2, int T_ngi>
__global__ void gpu_adv_add_diffusivity(GpuTensorField diffu_field, GpuScalarField t_field,int ele_num,double dt_theta, 
           double*dt_t,double *detwei,double *matrix_addto,double *rhs_addto) {
int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
if(thread_idx>=ele_num)return;
thread_idx+=1;
double diffusivity_gi[T_ngi][T_dim][T_dim];
gpu_ele_val_at_quad_tensor<T_dim,T_loc1,T_ngi>(diffu_field,thread_idx,&diffusivity_gi[0][0][0]);
double diffumat[T_loc1][T_loc2]={0};
gpu_dshape_tensor_dshape_t<T_dim,T_loc1,T_loc2,T_ngi>(dt_t + (thread_idx - 1)*T_loc1*T_ngi*T_dim, &diffusivity_gi[0][0][0],dt_t + (thread_idx - 1)*T_loc1*T_ngi*T_dim,
        detwei + (thread_idx - 1)*T_ngi,  &diffumat[0][0]);

double ele_val[T_loc1];
gpu_ele_val_scalar<T_loc1>(t_field,thread_idx,&ele_val[0]);
double matval[T_loc1];
matmul_t<T_loc1,1,T_loc2>(&diffumat[0][0],&ele_val[0],&matval[0]);
for(int jj = 0; jj < T_loc1; ++jj){
rhs_addto[(thread_idx-1)*T_loc1+jj]-=matval[jj];
}
if(dt_theta >std::numeric_limits<double>::epsilon()){
for(int ii = 0; ii < T_loc1; ++ii){
#pragma unroll
for(int jj = 0; jj < T_loc2; ++jj){
diffumat[ii][jj]=dt_theta*diffumat[ii][jj];
}
}
}
double *addto_ptr = (matrix_addto + T_loc1*T_loc2*(thread_idx - 1));
gpu_adv_matrix_addto(addto_ptr,&diffumat[0][0]);

}

// advection_mat = shape_vector_dot_dshape(test_function, velocity_at_quad, dt_t, detwei)
//!        if(abs(beta) > epsilon(0.0)) then
//          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
//          advection_mat = advection_mat &
//                    + beta*shape_shape(test_function, t_shape, velocity_div_at_quad*detwei)
//       end if

__global__ void gpu_adv_add_advection(GpuVectorField vector_field,GpuScalarField t_field, int ele_num,double*du_t,double *detwei,
double beta,double dt_theta, double *matrix_addto,double *rhs_addto) {
int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
if(thread_idx>=ele_num)return;
thread_idx+=1;
double velocity_at_quad[11][3];
gpu_ele_val_at_quad_vector(vector_field,thread_idx,&velocity_at_quad[0][0]);

double *test_function=gpu_ele_shape_vector(vector_field,thread_idx)->n;

double advection_mat[4][4]={0};
gpu_shape_vector_dot_dshape(test_function,&velocity_at_quad[0][0],
du_t + (thread_idx - 1)*4*11*3,detwei + (thread_idx-1)*11,&advection_mat[0][0]);
double ele_val[4];
gpu_ele_val_scalar<4>(t_field,thread_idx,&ele_val[0]);
double matval[4];
matmul_t<4,1,4>(&advection_mat[0][0],&ele_val[0],&matval[0]);
for(int jj = 0; jj < 4; ++jj){
rhs_addto[(thread_idx-1)*4+jj]-= matval[jj];
}

if(dt_theta >std::numeric_limits<double>::epsilon()){
for(int ii = 0; ii < 4; ++ii){
for(int jj = 0; jj < 4; ++jj){
advection_mat[ii][jj]=dt_theta*advection_mat[ii][jj];
}
}
}
gpu_adv_matrix_addto(matrix_addto + 16*(thread_idx - 1),&advection_mat[0][0]);

}
template<int T_loc>
__global__ void adv_rhs_addto_kernel(GpuScalarField field, const int ele_num, double *gpu_ele_val){
 int  element_index = blockDim.x * blockIdx.x + threadIdx.x;
 if(element_index >=ele_num) return;
  element_index = element_index + 1;
 int nodes[T_loc];
 gpu_ele_nodes_scalar<4>(field, element_index,nodes);
 double *ptr = gpu_ele_val + (element_index -1)*T_loc;
  #pragma unroll
  for(int i = 0; i < T_loc;++i){
   atomicAdd(&(field).val[(nodes[i] -1)],*(ptr+i));
  }

}

 
__global__ void gpu_div_ele_mat(GpuScalarField pressure_field, GpuScalarField den_field,
                GpuScalarField olddensity_field,int ele_num,double theta, 
                double *dn,double *div_dfield, double *div_detwei,double *div_mat){
 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

  double density_grad_at_quad[11][3];
  double olddensity_grad_at_quad[11][3];
  double *dfield = div_dfield + (thread_idx - 1)*3*11*4; 
  gpu_ele_grad_at_quad_scalar(den_field,thread_idx,dfield,&density_grad_at_quad[0][0]);
  gpu_ele_grad_at_quad_scalar(olddensity_field,thread_idx,dfield,&olddensity_grad_at_quad[0][0]);

  for(int gi = 0; gi < 11; ++gi){
    for(int ndim = 0; ndim < 3; ++ndim){
      density_grad_at_quad[gi][ndim]=theta*density_grad_at_quad[gi][ndim]
      +(1-theta)*olddensity_grad_at_quad[gi][ndim]; 
    } 
  }
   double density_gi[11];
   gpu_ele_val_at_quad_scalar(den_field,thread_idx,&density_gi[0]);
   double olddensity_gi[11];
   gpu_ele_val_at_quad_scalar(olddensity_field,thread_idx,&olddensity_gi[0]);
   double *detwei = div_detwei + (thread_idx - 1)*11; 
   
   for(int gi = 0; gi < 11; ++gi){
   density_gi[gi]=detwei[gi]*(theta * density_gi[gi] + (1-theta)*olddensity_gi[gi]); 
   }
   double *test_shape=gpu_ele_shape_scalar(pressure_field,thread_idx)->n;  
   double *field_shape=test_shape; ///velocity shape
   
   double *ele_mat = div_mat + (thread_idx - 1)*3*4*4;
   double local_ele_mat[4][4][3];
   gpu_shape_dshape(test_shape,dfield,density_gi,&local_ele_mat[0][0][0]); 
   gpu_shape_shape_vector(test_shape,field_shape,detwei,&density_grad_at_quad[0][0],ele_mat);

   for(int i = 0; i < 4; ++i){
     for(int j = 0; j < 4; ++j){
       #pragma unroll
       for(int k=0; k <3; ++k){
         ele_mat[i*4*3 + j * 3 + k] += local_ele_mat[i][j][k]; 
       } 
     } 
   }
}
__global__ void gpu_prj_ele_mat(GpuScalarField den_field,GpuScalarField oldden_field,GpuScalarField pressure_field,
      GpuScalarField drhodp_field,GpuScalarField eospressure_field,
      int ele_num,double atmos_pressure, double invdt, double factor,
      double *dev_detwei,double *dev_ele_mat, double *dev_ele_rhs){

 int  thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(thread_idx >=ele_num)return;
  thread_idx = thread_idx + 1;

  double den_at_quad[11];
  double oldden_at_quad[11];
  gpu_ele_val_at_quad_scalar(den_field,thread_idx,den_at_quad);
  gpu_ele_val_at_quad_scalar(oldden_field,thread_idx,oldden_at_quad);
  for(int gi = 0; gi < 11;++gi){
    den_at_quad[gi] = oldden_at_quad[gi] - den_at_quad[gi]; 
  }
  double p_at_quad[11];
  double eosp_at_quad[11];
  gpu_ele_val_at_quad_scalar(pressure_field,thread_idx,p_at_quad);
  
    #pragma unroll
  for(int gi = 0; gi < 11; ++gi){
    p_at_quad[gi]=p_at_quad[gi] + atmos_pressure;
  }
  gpu_ele_val_at_quad_scalar(eospressure_field,thread_idx,eosp_at_quad);
    #pragma unroll
  for(int gi = 0; gi < 11; ++gi){
    p_at_quad[gi]=eosp_at_quad[gi] - p_at_quad[gi];
  }

  double drhodp_at_quad[11];
  gpu_ele_val_at_quad_scalar(drhodp_field,thread_idx,drhodp_at_quad);
  double *detwei = (dev_detwei + (thread_idx - 1)*11);
    #pragma unroll
  for(int gi = 0; gi < 11; ++gi){
   den_at_quad[gi] = detwei[gi] * (drhodp_at_quad[gi]*p_at_quad[gi] + den_at_quad[gi]); 
  }
  double *test_shape=gpu_ele_shape_scalar(pressure_field,thread_idx)->n;  
  double *ele_rhs = dev_ele_rhs + (thread_idx - 1)*4;
  gpu_shape_rhs(test_shape,den_at_quad,ele_rhs);
  
    #pragma unroll
  for(int iloc = 0; iloc < 4; ++iloc){
   ele_rhs[iloc] = invdt * ele_rhs[iloc];  
  }  
  
    #pragma unroll
  for(int gi = 0; gi < 11; ++gi){
    drhodp_at_quad[gi] = detwei[gi] * drhodp_at_quad[gi];  
  }

  double *ele_mat = dev_ele_mat + (thread_idx - 1)*4*4; 
  gpu_shape_shape(test_shape,test_shape,drhodp_at_quad,ele_mat); 
  for(int i = 0; i < 4; ++i){
    #pragma unroll
    for(int j = 0; j < 4; ++j){
      ele_mat[i*4 + j]=factor * ele_mat[i*4+j];
    } 
  }
}



 __global__ void assemble_stiff_matrix(double* dshape1, double* tensor, double *dshape2, double *detwei,
    double* value, int* findrm, int* colm, int *ndglno, int ele_num, int dim, int loc, int ngi){
    //dshape1[dim][ngi][loc1]3/11/4     tensor[ngi][dim][dim]11/3/3 从c看的
    //这里维度写死 真正应该将维度作为参数传入
    //目前一个线程大约要用29个32bit寄存器 512线程
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=ele_num)return;
    int loc1=loc;
    int loc2=loc;
    // int ngi=11;
    // int dim=3;

    int stride1=dim*ngi*loc;
    int stride2=ngi;
    int stride3=ngi*dim*dim;

    double* ds1 = dshape1+blockIdx.x*blockDim.x*stride1 + threadIdx.x*stride1;
    double* ds2 = dshape2+blockIdx.x*blockDim.x*stride1 + threadIdx.x*stride1;
    double* detw = detwei+blockIdx.x*blockDim.x*stride2 + threadIdx.x*stride2;
    double* tens = tensor+blockIdx.x*blockDim.x*stride3 + threadIdx.x*stride3;
    int iloc=0,jloc=0,n=0;
    double r[16]={0.0};

    gpu_dshape_tensor_dshape_t<3,4,4,11>(ds1,tens,ds2,detw,&r[0]);

    //addto

    int mpos,base,upper_j,upper_pos,lower_j,lower_pos,this_pos,this_j;
    //To do: 若想寄存器最小化 尝试重新使用gi,n
    int i[4];
    int j;
    //To do: Vectorize
    int *row;
    i[0]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1];
    i[1]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+1];
    i[2]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+2];
    i[3]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+3];
    //int j[4]; 不用两个来记录loc的no了 反正一样的。。。
    //printf("%d %d %d %d\n",i[0],i[1],i[2],i[3]);
    for(iloc=0;iloc<loc1;iloc++){
        base = findrm[i[iloc]-1]-1;
        n = findrm[i[iloc]] -1 - base;
        for(jloc=0;jloc<loc2;jloc++){
            if(r[jloc*loc1+iloc]==0)continue;
            //To do：如果是for i{for j}这俩和i相关的值就可以放到最里层循环的外面

            //由于gpu不能动态分配数组 先直接用global中的value进行bisearch
            //To do:找方法简化
            row = colm+base;//指向colmbase
            upper_pos=n-1;
            upper_j=row[n-1]-1;    //因为row不是单独记录的数据，没经过减一
            lower_pos=0;
            lower_j=row[0]-1;
            mpos=-2;
            j = i[jloc]-1; //待search的j
            if (upper_j<j){
                mpos=-1;
            }else if (upper_j==j){
                mpos=upper_pos+base;
            }else if (lower_j>j){
                mpos=-1;
            }else if(lower_j==j){
                mpos=lower_pos+base;
            }

            while(((upper_pos-lower_pos)>1)&&(mpos==-2)){
                this_pos=(upper_pos+lower_pos)/2;
                this_j=row[this_pos]-1;
                if(this_j == (i[jloc]-1)){
                    mpos=this_pos+base;

                }     
                else if(this_j > (i[jloc]-1)){
                    // this_j>j
                    upper_j=this_j;
                    upper_pos=this_pos;
                }
                else{
                    // this_j<j
                    lower_j=this_j;
                    lower_pos=this_pos;
                }
            }            
            //printf("mpos:%d from thd%d\n",mpos,threadIdx.x);
            if(mpos<0){
                //printf("%d impossible! from block%d thd%d\n",mpos,blockIdx.x,threadIdx.x);
            }else{//val_associated
                atomicAdd(value+mpos,0.5*r[jloc*loc1+iloc]);    //mpos里包含了base
            }
        }
    }
    //考虑下 32个线程分别算完自己的16个indices再利用shuffle去改global中的value？？
    //因为相同i行的列（4个） 在colm和value里都姑且算接近

 }

 __global__ void assemble_csr_matrix(double *matrix_addto,
    double* value, int* findrm, int* colm, int *ndglno, int ele_num, int dim, int loc, int ngi,int bi=0){
    //dshape1[dim][ngi][loc1]3/11/4     tensor[ngi][dim][dim]11/3/3 从c看的
    //这里维度写死 真正应该将维度作为参数传入
    //目前一个线程大约要用29个32bit寄存器 512线程
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=ele_num)return;
    //addto

    int loc1=loc;
    int loc2=loc;

    int mpos,base,upper_j,upper_pos,lower_j,lower_pos,this_pos,this_j;
    //To do: 若想寄存器最小化 尝试重新使用gi,n
    int i[4];
    int j;
    double *r=matrix_addto + idx * 4 * 4*dim;
    //To do: Vectorize
    int *row=nullptr;
    int iloc=0, jloc=0,n=0;
    i[0]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1];
    i[1]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+1];
    i[2]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+2];
    i[3]= ndglno[(blockIdx.x*blockDim.x+threadIdx.x)*loc1+3];
    //int j[4]; 不用两个来记录loc的no了 反正一样的。。。
    //printf("%d %d %d %d\n",i[0],i[1],i[2],i[3]);
    for(iloc=0;iloc<loc1;iloc++){
        base = findrm[i[iloc]-1]-1;
        //printf("base:%d\n",base);
        n = findrm[i[iloc]] -1 - base;
        for(jloc=0;jloc<loc2;jloc++){
            if(r[jloc*loc1+iloc]==0)continue;
    //         base = findrm[i[iloc]-1]-1;
    //         printf("base:%d\n",base);
    //         n = findrm[i[iloc]] - base -1;       
            //To do：如果是for i{for j}这俩和i相关的值就可以放到最里层循环的外面

            //由于gpu不能动态分配数组 先直接用global中的value进行bisearch
            //To do:找方法简化
            row = colm+base;//指向colmbase
            upper_pos=n-1;
            upper_j=row[n-1]-1;    //因为row不是单独记录的数据，没经过减一
            lower_pos=0;
            lower_j=row[0]-1;
            mpos=-2;
            j = i[jloc]-1; //待search的j
            if (upper_j<j){
                mpos=-1;
            }else if (upper_j==j){
                mpos=upper_pos+base;
            }else if (lower_j>j){
                mpos=-1;
            }else if(lower_j==j){
                mpos=lower_pos+base;
            }

            while(((upper_pos-lower_pos)>1)&&(mpos==-2)){
                this_pos=(upper_pos+lower_pos)/2;
                this_j=row[this_pos]-1;
                if(this_j == (i[jloc]-1)){
                    mpos=this_pos+base;

                }     
                else if(this_j > (i[jloc]-1)){
                    // this_j>j
                    upper_j=this_j;
                    upper_pos=this_pos;
                }
                else{
                    // this_j<j
                    lower_j=this_j;
                    lower_pos=this_pos;
                }
            }            
            //printf("mpos:%d from thd%d\n",mpos,threadIdx.x);
            if(mpos<0){
                //printf("%d impossible! from block%d thd%d\n",mpos,blockIdx.x,threadIdx.x);
            }else{//val_associated
                //if(mpos==0)printf("value[0]+%f\n",0.5*r[jloc*loc1+iloc]);
                atomicAdd(value+mpos,r[(jloc*loc1+iloc)*dim+bi]);    //mpos里包含了base
                //printf("we're gonna add r[%d][%d]=%f to csr matrix\n",jloc,iloc,r[jloc*loc1+iloc]);
                //printf("we set value[%d] = %f\n",base+mpos,value[base+mpos]);
                //matrix.value[mpos] += value[jloc*loc1+iloc]; //gpu里改原子加
            }
        }
    }
    //考虑下 32个线程分别算完自己的16个indices再利用shuffle去改global中的value？？
    //因为相同i行的列（4个） 在colm和value里都姑且算接近

 }


#endif
