#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sm_20_atomic_functions.h"
#include "fields.h"
#include "cuda_global.h"
using namespace std;


GpuScalarField density_field;

GpuVectorField nu_field;
GpuVectorField rhs_field;
GpuVectorField source_field;
GpuVectorField oldu_field;
GpuVectorField u_field;
GpuVectorField x_field;
GpuVectorField inverse_masslump_field;
GpuTensorField viscosity_field;
GpuScalarField ct_rhs_field;

GpuVectorField global_host_rhs_field;
//copy back
GpuVectorField host_inverse_masslump_field;

cudaStream_t transfer_stream;
cudaStream_t second_stream;
cudaStream_t kernel_stream;
extern "C" void create_streams_(){
   cudaStreamCreateWithFlags(&transfer_stream,cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&second_stream,cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&kernel_stream,cudaStreamNonBlocking);
}
extern "C" void destroy_streams(){
 cudaStreamDestroy(transfer_stream);
 cudaStreamDestroy(second_stream);
 cudaStreamDestroy(kernel_stream);
}
int8_t *global_dev_ptr = nullptr;
int8_t *global_dev_shape_n = nullptr;
int8_t *global_gpu_momentum = nullptr;
int8_t *global_gpu_big_m = nullptr;
int8_t *global_field_obj=nullptr;
extern "C" void dealloc_gpu_momentum_(){
 cudaFreeAsync(global_dev_ptr,transfer_stream);
 cudaFreeAsync(global_dev_shape_n,transfer_stream);
 cudaFreeAsync(global_gpu_momentum,transfer_stream);
 cudaFreeAsync(global_gpu_big_m,transfer_stream);
 cudaFreeAsync(global_field_obj,transfer_stream);
 destroy_streams();
}

double *pinned_big_m=nullptr;
double *pinned_u_shape_dn=nullptr;
double *pinned_lx_shape_dn=nullptr;
double *pinned_quadra_weight=nullptr;
double *pinned_ct_rhs_val=nullptr;
double *pinned_density_val=nullptr;
double *pinned_shape_n=nullptr;
double *pinned_grad_p_u_mat=nullptr;

int8_t *pinned_mem = nullptr;

extern "C" void alloc_momentum_pinned_mem_(int *p_ele_num,int*p_node_num){
  int ele_num = *p_ele_num;
  const int dim=3, ngi = 11, nloc = 4;
  int dshape_size = dim*ngi*nloc;
  size_t pinned_len = sizeof(double)*dim*dim*nloc*nloc*ele_num //big_m_tensor_addto
  + sizeof(double)*dshape_size               //u_shape_dn
  + sizeof(double)*dshape_size               //lx_shape_dn   
  + sizeof(double)*ngi                       //quadr 
  + sizeof(double) + sizeof(double)          //ct_rhs_val,density_val
  + sizeof(double)*nloc*ngi                   //shape_n
  + sizeof(double)*dim * nloc * nloc * ele_num;// grad_p_u_mat
//  int8_t *pinned_mem = nullptr;
  cudaMallocHost((void**)&pinned_mem,pinned_len);
  size_t used_len=0;
  pinned_big_m = (double*) (pinned_mem + used_len);
  
  used_len += sizeof(double)*dim*dim*nloc*nloc*ele_num;
  pinned_u_shape_dn=(double*)(pinned_mem + used_len);    
  
  used_len +=sizeof(double)*dshape_size;  
  pinned_lx_shape_dn=(double*)(pinned_mem + used_len);    
  
  used_len +=sizeof(double)*dshape_size;  
  pinned_quadra_weight=(double*)(pinned_mem + used_len);    
  
  used_len +=sizeof(double)*ngi;  
  pinned_ct_rhs_val=(double*)(pinned_mem + used_len); 
  
  used_len +=sizeof(double);  
  pinned_density_val=(double*)(pinned_mem + used_len); 

  used_len +=sizeof(double);  
  pinned_shape_n=(double*)(pinned_mem + used_len);
  
  used_len +=sizeof(double) * nloc * ngi;
  pinned_grad_p_u_mat = (double*)(pinned_mem + used_len);
 
}
extern "C" void dealloc_momentum_pinned_mem_(){
cudaFreeHost(pinned_mem);
pinned_mem=nullptr;
}

extern "C" void alloc_gpu_momentum_(int *p_ele_num, int *p_node_num){
   const int nloc = 4, ngi=11,dim=3,dim0=3,dim1=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   
   int64_t total_gpu_mem = sizeof(int)*nloc*ele_num   //one ndglno
   +  sizeof(double)*dim0*dim1*node_num  + sizeof(double)*dim*node_num //viscosity, x
   + sizeof(double)*dim*node_num + sizeof(double)*dim*node_num  //invserse_masslump,u
   + sizeof(double)*dim*node_num +sizeof(double)*dim*node_num //oldu,nu
   + sizeof(double)*dim*node_num  //momentum, rhs 
   + sizeof(double)*dim*node_num + sizeof(double)  //source,ct_rhs
   + sizeof(double);  //density
   
   //cudaStreamCreateWithFlags(&transfer_stream,cudaStreamNonBlocking);

   cudaMallocAsync((void**)&global_dev_ptr,total_gpu_mem, transfer_stream);
   int64_t shape_len=sizeof(double)*nloc *ngi;  
   cudaMallocAsync((void**)&global_dev_shape_n,shape_len,transfer_stream); 
  
    int64_t dshape_size = dim * ngi * nloc; 
    int64_t detwei_size = ngi * ele_num;
    int64_t total_momentum_mem = sizeof(double)*dshape_size //d_u_shape_dn
    + sizeof(double)*dshape_size + sizeof(double)*ngi //d_lx_shape_dn,quadrature_weight
    + sizeof(double)*dshape_size *ele_num + sizeof(double)*detwei_size //d_du_t,d_detwei
    + sizeof(double) * dim * nloc * nloc * ele_num; //gpu_grad_p_u_mat 
    int64_t total_big_m_mem =
    sizeof(double) * dim0 * dim1 * nloc *nloc * ele_num//gpu_big_m_tensor_addto
    + sizeof(double) * dim * nloc * ele_num //rhs_addto
    + sizeof(double) * dim * nloc * ele_num; //gpu_big_m_diag_addto  

   cudaMallocAsync((void**)&global_gpu_momentum,total_momentum_mem,transfer_stream); 
   cudaMallocAsync((void**)&global_gpu_big_m,total_big_m_mem,transfer_stream); 
   
}

extern "C" void shape_dn_to_device_(double *u_shape_dn,double *lx_shape_dn,
double *lx_shape_weight){
    const int nloc = 4, ngi=11,dim=3;
    double *d_u_shape_dn=nullptr;
    int dshape_size = dim*ngi*nloc;
     
     d_u_shape_dn =(double*) global_gpu_momentum;
     size_t total_shape_dn_len=sizeof(double)*dshape_size
     + sizeof(double)*dshape_size
     + sizeof(double)*ngi;

     memcpy(pinned_u_shape_dn,u_shape_dn,sizeof(double)*dshape_size); 
     memcpy(pinned_lx_shape_dn,lx_shape_dn,sizeof(double)*dshape_size); 
     memcpy(pinned_quadra_weight,lx_shape_weight,sizeof(double)*ngi); 
     cudaMemcpyAsync(d_u_shape_dn, pinned_u_shape_dn, total_shape_dn_len,cudaMemcpyHostToDevice,transfer_stream);
}

double *global_du_t = nullptr;
double *global_detwei=nullptr;


double *global_gpu_grad_p_u_mat=nullptr;
extern "C" void gpu_transform_to_physical_face_(int *p_ele_num, double *u_shape_dn, double *lx_shape_dn,
double *lx_shape_weight,double *grad_p_u_mat){
    const int nloc = 4,ngi=11,dim=3;
    int dshape_size = dim*ngi*nloc;
    int detwei_size = (ngi)*(*p_ele_num);
    int ele_count=*p_ele_num;
    int64_t used_len = 0;
     
    //double  *d_u_shape_dn =(double*) global_gpu_momentum;
     used_len += sizeof(double)*dshape_size; 
   //double * d_lx_shape_dn =(double*)(global_gpu_momentum + used_len);   
     used_len += sizeof(double)*dshape_size;
   //double * d_quadrature_weight = (double*)(global_gpu_momentum + used_len);
     used_len += sizeof(double)*ngi;

    double *d_du_t=nullptr;
    double *d_detwei=nullptr;
 
    d_du_t = (double*)(global_gpu_momentum + used_len);
    used_len += sizeof(double)*dshape_size*(ele_count);
    d_detwei = (double*)(global_gpu_momentum + used_len);
    used_len += sizeof(double) * detwei_size;
    
    global_du_t = d_du_t;
    global_detwei=d_detwei;


    int block_size=256;
    dim3 grid = ((ele_count)+block_size-1)/block_size;
    dim3 block = block_size;
 
   double *gpu_grad_p_u_mat=nullptr; 
    gpu_grad_p_u_mat = (double*)(global_gpu_momentum + used_len); 
    used_len += sizeof(double)*dim *nloc * nloc *ele_count;
   global_gpu_grad_p_u_mat = gpu_grad_p_u_mat;
   gpu_grad_mat<<<grid,block,0,kernel_stream>>>(ct_rhs_field,ele_count, gpu_grad_p_u_mat,d_du_t, d_detwei);
 
}


extern "C" void gpu_viscosity_face_(int *p_ele_num,double *beta,double *dt,
double *theta, double *smagorinsky_coefficient, double *big_m_tensor_addto,int* assemble_ct){
   int ele_num = *p_ele_num;

   const int nloc = 4, dim=3,dim0=3,dim1=3;
   
  double *gpu_du_t = nullptr;
   double *gpu_detwei = nullptr;
   double *gpu_big_m_tensor_addto = nullptr;
   double *gpu_rhs_addto = nullptr;
   double *gpu_big_m_diag_addto=nullptr;
   gpu_du_t=global_du_t;
   gpu_detwei =global_detwei;

     int64_t used_len = 0;  
     gpu_big_m_tensor_addto = (double*)(global_gpu_big_m + used_len);
     used_len += sizeof(double)*dim0 * dim1 * nloc *nloc * ele_num; 
     
     gpu_rhs_addto =(double*)(global_gpu_big_m + used_len);
     used_len += sizeof(double) * dim * nloc *ele_num;
     
    gpu_big_m_diag_addto = (double*)(global_gpu_big_m + used_len); 
 
//    cudaStreamSynchronize(second_stream);
    
   dim3 threadsPerBlock(128,1,1);
   int num_of_block=(ele_num + 128 -1)/128;
   dim3 blocksPerGrid(num_of_block,1,1); 

  gpu_add_mass_element<<<blocksPerGrid,threadsPerBlock,0,kernel_stream>>>(u_field,density_field,inverse_masslump_field, ele_num,gpu_detwei,gpu_big_m_diag_addto); 

   cudaStreamSynchronize(kernel_stream);
  
  gpu_add_advection<<<blocksPerGrid,threadsPerBlock,0,kernel_stream>>>(nu_field,density_field,u_field,oldu_field,ele_num,*beta,*dt,*theta,gpu_du_t,gpu_detwei,gpu_big_m_tensor_addto,gpu_rhs_addto);
 
  cudaMemcpyAsync(host_inverse_masslump_field.mesh.ndglno,host_inverse_masslump_field.val,
   sizeof(double)*dim*host_inverse_masslump_field.mesh.nodes,cudaMemcpyDeviceToHost,second_stream);
  gpu_add_sources<<<blocksPerGrid,threadsPerBlock,0,kernel_stream>>>(density_field,source_field,u_field,ele_num,gpu_detwei,gpu_rhs_addto);
  gpu_add_viscosity_element<<<blocksPerGrid,threadsPerBlock,0,kernel_stream>>>(viscosity_field, nu_field,oldu_field,u_field,
                      density_field,ele_num,*dt,*theta,*smagorinsky_coefficient,gpu_du_t,gpu_detwei,gpu_big_m_tensor_addto,gpu_rhs_addto);
//   if(*assemble_ct==0){
//   }else{
   cudaMemcpyAsync(pinned_grad_p_u_mat,global_gpu_grad_p_u_mat,
   sizeof(double)*dim*nloc*nloc*ele_num,cudaMemcpyDeviceToHost,second_stream);
//   } 
  gpu_add_diagonal_to_tensor<<<blocksPerGrid,threadsPerBlock,0,kernel_stream>>>(ele_num,gpu_big_m_diag_addto,gpu_big_m_tensor_addto);
 
  gpu_rhs_addto_kernel<<<blocksPerGrid, threadsPerBlock,0,kernel_stream>>>(rhs_field,ele_num,gpu_rhs_addto); 
  
  
  cudaStreamSynchronize(kernel_stream);

  cudaMemcpyAsync(global_host_rhs_field.mesh.ndglno,global_host_rhs_field.val,
  sizeof(double)*3*global_host_rhs_field.mesh.nodes,cudaMemcpyDeviceToHost,second_stream); 
  
 cudaMemcpyAsync(pinned_big_m,gpu_big_m_tensor_addto,sizeof(double)*3*3*4*4*ele_num,
              cudaMemcpyDeviceToHost,second_stream);
 cudaStreamSynchronize(second_stream);
   
}



extern "C" void pinned_copy_back_(int *p_ele_num, int *p_node_num,
double *big_m_tensor_addto, double *masslump, double *grad_p_u_mat){
const int dim=3, nloc = 4;
int ele_num = *p_ele_num;
memcpy(grad_p_u_mat,pinned_grad_p_u_mat,sizeof(double)*dim*nloc*nloc*ele_num);
memcpy(big_m_tensor_addto,pinned_big_m,sizeof(double)*dim*dim*nloc*nloc*ele_num);
}


extern "C" void set_gpu_data_(int *p_ele_num, int *p_node_num,
int *field_mesh_ndglno, double *field_mesh_shape_n,
double *field_viscosity_val, double *field_x_val,double*field_inverse_val,
double *field_u_val, double *field_oldu_val, double *field_nu_val,
double *field_rhs_val, double *field_source_val, double*field_ct_rhs_val, double*field_density_val){

   const int nloc = 4, ngi=11,dim=3,dim0=3,dim1=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   
   cudaStreamSynchronize(transfer_stream);

   int8_t * dev_ptr = global_dev_ptr;
   int8_t *dev_shape_n = global_dev_shape_n;
   int *dev_ndglno = (int*)dev_ptr;

   int64_t used_len = sizeof(int)*nloc*ele_num; 
   double *dev_viscosity_val = (double*)(dev_ptr + used_len); 
      
   viscosity_field.mesh.shape.n = (double*)dev_shape_n;
   viscosity_field.mesh.ndglno = dev_ndglno;
   viscosity_field.mesh.shape.loc=nloc;
   viscosity_field.mesh.shape.ngi=ngi;
   viscosity_field.dim[0]=dim0;
   viscosity_field.dim[1]=dim1;
   viscosity_field.val=dev_viscosity_val;
   viscosity_field.field_type=FIELD_TYPE_NORMAL;
   viscosity_field.mesh.elements=*p_ele_num;
   viscosity_field.mesh.nodes=*p_node_num;
   
   used_len += sizeof(double)*dim0*dim1*node_num;
   double *dev_x_val = (double*)(dev_ptr + used_len);    
       x_field.mesh.shape.n = (double*)dev_shape_n;
       x_field.mesh.ndglno= dev_ndglno;
       x_field.mesh.shape.loc=nloc;
       x_field.mesh.shape.ngi=ngi;
       x_field.dim=dim;
       x_field.val=dev_x_val;
       x_field.field_type=FIELD_TYPE_NORMAL;
       x_field.mesh.elements=*p_ele_num;
       x_field.mesh.nodes=*p_node_num;
   used_len += sizeof(double)*dim*node_num;
   double *dev_inverse_val = (double*)(dev_ptr + used_len);    
       inverse_masslump_field.mesh.shape.n = (double*)dev_shape_n;
       inverse_masslump_field.mesh.ndglno= dev_ndglno;
       inverse_masslump_field.mesh.shape.loc=nloc;
       inverse_masslump_field.mesh.shape.ngi=ngi;
       inverse_masslump_field.dim=dim;
       inverse_masslump_field.val=dev_inverse_val;
       inverse_masslump_field.field_type=FIELD_TYPE_NORMAL;
       inverse_masslump_field.mesh.elements=*p_ele_num;
       inverse_masslump_field.mesh.nodes=*p_node_num;
///hack hack
   host_inverse_masslump_field.mesh.ndglno=(int*)field_inverse_val; 
   host_inverse_masslump_field.mesh.nodes=*p_node_num; 
   host_inverse_masslump_field.val=dev_inverse_val;
    
   used_len += sizeof(double)*dim*node_num;
   double *dev_u_val = (double*)(dev_ptr + used_len);    

       u_field.mesh.shape.n = (double*)dev_shape_n;
       u_field.mesh.ndglno= dev_ndglno;
       u_field.mesh.shape.loc=nloc;
       u_field.mesh.shape.ngi=ngi;
       u_field.dim=dim;
       u_field.val=dev_u_val;
       u_field.field_type=FIELD_TYPE_NORMAL;
       u_field.mesh.elements=*p_ele_num;
       u_field.mesh.nodes=*p_node_num;
       
   used_len += sizeof(double)*dim*node_num;
   double *dev_oldu_val = (double*)(dev_ptr + used_len);    
       oldu_field.mesh.shape.n = (double*)dev_shape_n;
       oldu_field.mesh.ndglno= dev_ndglno;
       oldu_field.mesh.shape.loc=nloc;
       oldu_field.mesh.shape.ngi=ngi;
       oldu_field.dim=dim;
       oldu_field.val=dev_oldu_val;
       oldu_field.field_type=FIELD_TYPE_NORMAL;
       oldu_field.mesh.elements=*p_ele_num;
       oldu_field.mesh.nodes=*p_node_num;

   used_len += sizeof(double)*dim*node_num;
   double *dev_nu_val = (double*)(dev_ptr + used_len);    
       nu_field.mesh.shape.n = (double*)dev_shape_n;
       nu_field.mesh.ndglno= dev_ndglno;
       nu_field.mesh.shape.loc=nloc;
       nu_field.mesh.shape.ngi=ngi;
       nu_field.dim=dim;
       nu_field.val=dev_nu_val;
       nu_field.field_type=FIELD_TYPE_NORMAL;
       nu_field.mesh.elements=*p_ele_num;
       nu_field.mesh.nodes=*p_node_num;
  
   used_len += sizeof(double)*dim*node_num;
   double *dev_rhs_val = (double*)(dev_ptr + used_len);    
       rhs_field.mesh.shape.n = (double*)dev_shape_n;
       rhs_field.mesh.ndglno= dev_ndglno;
       rhs_field.mesh.shape.loc=nloc;
       rhs_field.mesh.shape.ngi=ngi;
       rhs_field.dim=dim;
       rhs_field.val=dev_rhs_val;
       rhs_field.field_type=FIELD_TYPE_NORMAL;
       rhs_field.mesh.elements=*p_ele_num;
       rhs_field.mesh.nodes=*p_node_num;
       
       global_host_rhs_field.mesh.ndglno=(int*)field_rhs_val;
       global_host_rhs_field.val = dev_rhs_val;
       global_host_rhs_field.mesh.nodes=*p_node_num;

   used_len += sizeof(double)*dim*node_num;
   double *dev_source_val = (double*)(dev_ptr + used_len);    
       source_field.mesh.shape.n = (double*)dev_shape_n;
       source_field.mesh.ndglno= dev_ndglno;
       source_field.mesh.shape.loc=nloc;
       source_field.mesh.shape.ngi=ngi;
       source_field.dim=dim;
       source_field.val=dev_source_val;
       source_field.field_type=FIELD_TYPE_NORMAL;
       source_field.mesh.elements=*p_ele_num;
       source_field.mesh.nodes=*p_node_num;
       //source_field.serialize("source_field");
   used_len += sizeof(double)*dim*node_num;
   double *dev_ct_rhs_val = (double*)(dev_ptr + used_len);    
  ct_rhs_field.mesh.shape.n = (double*)dev_shape_n;
  ct_rhs_field.mesh.ndglno= dev_ndglno;
  ct_rhs_field.mesh.shape.loc=nloc;
  ct_rhs_field.mesh.shape.ngi=ngi;
  ct_rhs_field.val=dev_ct_rhs_val;
  ct_rhs_field.field_type=FIELD_TYPE_CONSTANT;
  ct_rhs_field.mesh.elements=*p_ele_num;
  ct_rhs_field.mesh.nodes=*p_node_num;
   used_len += sizeof(double);
   double *dev_density_val = (double*)(dev_ptr + used_len);    

  density_field.mesh.shape.n = (double*)dev_shape_n;
  density_field.mesh.ndglno= dev_ndglno;
  density_field.mesh.shape.loc=nloc;
  density_field.mesh.shape.ngi=ngi;
  density_field.val=dev_density_val;
  density_field.field_type=FIELD_TYPE_CONSTANT;
  density_field.mesh.elements=*p_ele_num;
  density_field.mesh.nodes=*p_node_num;

  cudaMemcpyAsync(dev_ndglno,field_mesh_ndglno,sizeof(int)*nloc*ele_num,cudaMemcpyHostToDevice,transfer_stream);
  cudaMemcpyAsync(dev_x_val,field_x_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,transfer_stream);
 
  memcpy(pinned_ct_rhs_val,field_ct_rhs_val,sizeof(double)); 
  memcpy(pinned_density_val,field_density_val,sizeof(double)); 
  memcpy(pinned_shape_n,field_mesh_shape_n,sizeof(double)*nloc*ngi); 
  
  cudaMemcpyAsync(dev_ct_rhs_val,pinned_ct_rhs_val,sizeof(double),cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_density_val,pinned_density_val,sizeof(double),cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_shape_n,pinned_shape_n,sizeof(double)*nloc*ngi,cudaMemcpyHostToDevice,second_stream);

   double *d_u_shape_dn=nullptr;
   double *d_lx_shape_dn=nullptr;
    double *d_quadrature_weight=nullptr; 
    int dshape_size = dim*ngi*nloc;
    int detwei_size = (ngi)*(*p_ele_num);
    int ele_count=*p_ele_num;
    used_len = 0;
     
     d_u_shape_dn =(double*) global_gpu_momentum;
     used_len += sizeof(double)*dshape_size; 
     d_lx_shape_dn =(double*)(global_gpu_momentum + used_len); 
     used_len += sizeof(double)*dshape_size;
     d_quadrature_weight = (double*)(global_gpu_momentum + used_len);
     used_len += sizeof(double)*ngi;

    double *d_du_t=nullptr;
    double *d_detwei=nullptr;
 
    d_du_t = (double*)(global_gpu_momentum + used_len);
    used_len += sizeof(double)*dshape_size*(ele_count);
    d_detwei = (double*)(global_gpu_momentum + used_len);
    used_len += sizeof(double) * detwei_size;
    
    global_du_t = d_du_t;
    global_detwei=d_detwei;

    cudaStreamSynchronize(transfer_stream);
    int block_size=256;
    dim3 grid = ((ele_count)+block_size-1)/block_size;
    dim3 block = block_size;

   transform_to_physical_full_detwei<3,4,11><<<grid,block,0,kernel_stream>>>(x_field.val,d_u_shape_dn,d_lx_shape_dn,
    d_quadrature_weight,x_field.mesh.ndglno, ele_count,d_du_t, d_detwei);
   
  cudaMemcpyAsync(dev_viscosity_val,field_viscosity_val,sizeof(double)*dim0*dim1*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_inverse_val,field_inverse_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_u_val,field_u_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_oldu_val,field_oldu_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice, second_stream);
  cudaMemcpyAsync(dev_nu_val,field_nu_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_source_val,field_source_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_viscosity_val,field_viscosity_val,sizeof(double)*dim0*dim1*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_inverse_val,field_inverse_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_u_val,field_u_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_oldu_val,field_oldu_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice, second_stream);
  cudaMemcpyAsync(dev_nu_val,field_nu_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
  cudaMemcpyAsync(dev_source_val,field_source_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice,second_stream);
 
}
#define ADVECTION 1
#if (ADVECTION == 1)

GpuScalarField adv_host_t;
GpuScalarField adv_host_rhs;
GpuScalarField adv_host_density;
GpuScalarField adv_host_nvfrac;
GpuScalarField adv_host_source;

GpuVectorField adv_host_pos;
GpuVectorField adv_host_velocity;

GpuTensorField adv_host_diffu;

cudaStream_t adv_mem_stream;


extern "C" void  adv_create_stream(){
   cudaStreamCreateWithFlags(&adv_mem_stream,cudaStreamNonBlocking);
   return;
}
extern "C" void adv_destroy_stream_(){
 cudaStreamDestroy(adv_mem_stream);
return;
}

int8_t *adv_global_gpu_ptr=nullptr;
int8_t *adv_global_dev_shape_n = nullptr;
int8_t *adv_global_ndglno=nullptr;
int8_t *adv_global_field=nullptr;
int8_t *adv_dev_shape_n=nullptr;

double *global_adv_detwei=nullptr;
double *global_adv_dshape=nullptr;
double *global_adv_rhs = nullptr;
double *global_adv_matrix = nullptr;

double *global_assemble_matrix=nullptr;


extern "C" void dealloc_adv_mem_(){
 cudaFreeAsync(adv_global_gpu_ptr,adv_mem_stream);
 cudaFreeAsync(adv_global_dev_shape_n,adv_mem_stream);
 cudaFreeAsync(adv_global_field,adv_mem_stream);
 cudaFreeAsync(adv_global_ndglno,adv_mem_stream);
 cudaFreeAsync(adv_dev_shape_n,adv_mem_stream);
 cudaFreeAsync(global_adv_rhs,adv_mem_stream);
 cudaFreeAsync(global_adv_matrix,adv_mem_stream);
 return;
}
extern "C" void  adv_alloc_gpu_mem(int p_ele_num, int p_node_num){

   const int nloc = 4, ngi=11,dim=3,dim0=3,dim1=3;
   int ele_num = p_ele_num;
   int node_num = p_node_num;
   int64_t total_gpu_len = sizeof(double)*node_num  //t
    +sizeof(double)*node_num                //rhs
    +sizeof(double)*dim*node_num            //position
    +sizeof(double)*dim*node_num            //velocity
    +sizeof(double)*dim0*dim1*node_num      //diffusivity
    +sizeof(double)*node_num               //nvfrac //
    +sizeof(double)*node_num               //source //
    +sizeof(double);                //density //or sizeof(double)
    
    int64_t shape_len=sizeof(double)*nloc *ngi;  
    
    cudaStreamSynchronize(adv_mem_stream);
    
    cudaMallocAsync((void**)&adv_global_gpu_ptr,total_gpu_len, adv_mem_stream);
    cudaMallocAsync((void**)&adv_global_dev_shape_n,shape_len,adv_mem_stream); 
    int ndglno_len = sizeof(int)*nloc*ele_num;
    cudaMallocAsync((void**)&adv_global_ndglno,ndglno_len,adv_mem_stream); 

    int64_t dshape_size = dim * ngi * nloc; 
    int64_t detwei_size = ngi * ele_num;
    int64_t adv_shape_length = sizeof(double)*dshape_size //d_u_shape_dn
    + sizeof(double)*dshape_size + sizeof(double)*ngi    //d_lx_shape_dn,quadrature_weight
    + sizeof(double)*dshape_size *ele_num + sizeof(double)*detwei_size; //d_du_t,d_detwei
    cudaMallocAsync((void**)&adv_dev_shape_n,adv_shape_length,adv_mem_stream); 
    
    cudaMallocAsync(&global_adv_rhs,4*ele_num*sizeof(double),adv_mem_stream);
   
    cudaMallocAsync(&global_adv_matrix,4*4*ele_num*sizeof(double),adv_mem_stream);

    return;
}

extern "C" void adv_set_gpu_data_(int *p_ele_num, int *p_node_num,
int *field_mesh_ndglno,double*field_mesh_shape_n,
double *field_t_val,double *field_rhs_val,double*field_pos_val,double*field_velocity_val,
double *field_diffu_val,double*field_density_val,double *field_nvfrac_val,double *field_source_val,
int *nvfrac_associated, int*have_source,int *have_diffu){

   const int nloc = 4, ngi=11,dim=3,dim0=3,dim1=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   cudaStreamSynchronize(adv_mem_stream);
   int8_t * dev_ptr = adv_global_gpu_ptr;
   int8_t *dev_shape_n = adv_global_dev_shape_n;
   
    cudaMemcpyAsync(dev_shape_n,field_mesh_shape_n,4*11*sizeof(double),cudaMemcpyHostToDevice,adv_mem_stream);
    
    int *dev_ndglno = (int*)adv_global_ndglno;
    cudaMemcpyAsync(dev_ndglno,field_mesh_ndglno,sizeof(int)*nloc*ele_num,cudaMemcpyHostToDevice,adv_mem_stream);

   int64_t used_len = 0; 
   double *dev_t_val = (double*)(dev_ptr + used_len); 

       adv_host_t.mesh.shape.n = (double*)dev_shape_n;
       adv_host_t.mesh.ndglno= dev_ndglno;
       adv_host_t.mesh.shape.loc=nloc;
       adv_host_t.mesh.shape.ngi=ngi;
       adv_host_t.val=dev_t_val;
       adv_host_t.field_type=FIELD_TYPE_NORMAL;
       adv_host_t.mesh.elements=*p_ele_num;
       adv_host_t.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;

   double *dev_rhs_val = (double*)(dev_ptr + used_len); 

       adv_host_rhs.mesh.shape.n = (double*)dev_shape_n;
       adv_host_rhs.mesh.ndglno= dev_ndglno;
       adv_host_rhs.mesh.shape.loc=nloc;
       adv_host_rhs.mesh.shape.ngi=ngi;
       adv_host_rhs.val=dev_rhs_val;
       adv_host_rhs.field_type=FIELD_TYPE_NORMAL;
       adv_host_rhs.mesh.elements=*p_ele_num;
       adv_host_rhs.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;
     
     double *dev_pos_val=(double*)(dev_ptr + used_len);
      adv_host_pos.mesh.shape.n = (double*)dev_shape_n;
       adv_host_pos.mesh.ndglno= dev_ndglno;
       adv_host_pos.mesh.shape.loc=nloc;
       adv_host_pos.mesh.shape.ngi=ngi;
       adv_host_pos.dim=dim;
       adv_host_pos.val=dev_pos_val;
       adv_host_pos.field_type=FIELD_TYPE_NORMAL;
       adv_host_pos.mesh.elements=*p_ele_num;
       adv_host_pos.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*dim*node_num;
      
     double *dev_velocity_val = (double*)(dev_ptr + used_len);
      adv_host_velocity.mesh.shape.n = (double*)dev_shape_n;
      adv_host_velocity.mesh.ndglno= dev_ndglno;
      adv_host_velocity.mesh.shape.loc=nloc;
      adv_host_velocity.mesh.shape.ngi=ngi;
      adv_host_velocity.dim=dim;
      adv_host_velocity.val=dev_velocity_val;
      adv_host_velocity.field_type=FIELD_TYPE_NORMAL;
      adv_host_velocity.mesh.elements=*p_ele_num;
      adv_host_velocity.mesh.nodes=*p_node_num;
    
 used_len +=  sizeof(double)*dim*node_num;

      double *dev_diffu_val = (double*)(dev_ptr + used_len);
      adv_host_diffu.mesh.shape.n = (double*)dev_shape_n;
      adv_host_diffu.mesh.ndglno= dev_ndglno;
      adv_host_diffu.mesh.shape.loc=nloc;
      adv_host_diffu.mesh.shape.ngi=ngi;
      adv_host_diffu.dim[0]=dim0;
      adv_host_diffu.dim[1]=dim1;
      adv_host_diffu.val=dev_diffu_val;
      adv_host_diffu.field_type=FIELD_TYPE_NORMAL;
      adv_host_diffu.mesh.elements=*p_ele_num;
      adv_host_diffu.mesh.nodes=*p_node_num;
 
    used_len +=  sizeof(double)*dim0*dim1*node_num;
      
  double *dev_nvfrac_val=(double*)(dev_ptr + used_len);
  adv_host_nvfrac.mesh.shape.n = (double*)dev_shape_n;
  adv_host_nvfrac.mesh.ndglno= dev_ndglno;
  adv_host_nvfrac.mesh.shape.loc=nloc;
  adv_host_nvfrac.mesh.shape.ngi=ngi;
  adv_host_nvfrac.val=dev_nvfrac_val;
  adv_host_nvfrac.field_type=FIELD_TYPE_NORMAL;
  adv_host_nvfrac.mesh.elements=*p_ele_num;
  adv_host_nvfrac.mesh.nodes=*p_node_num;
  used_len +=  sizeof(double)*node_num;
  
  double *dev_source_val=(double*)(dev_ptr + used_len);
  adv_host_source.mesh.shape.n = (double*)dev_shape_n;
  adv_host_source.mesh.ndglno= dev_ndglno;
  adv_host_source.mesh.shape.loc=nloc;
  adv_host_source.mesh.shape.ngi=ngi;
  adv_host_source.val=dev_source_val;
  adv_host_source.field_type=FIELD_TYPE_NORMAL;
  adv_host_source.mesh.elements=*p_ele_num;
  adv_host_source.mesh.nodes=*p_node_num;
  used_len +=  sizeof(double)*node_num;

  double *dev_density_val = (double*)(dev_ptr+used_len);
  adv_host_density.mesh.shape.n = (double*)dev_shape_n;
  adv_host_density.mesh.ndglno= dev_ndglno;
  adv_host_density.mesh.shape.loc=nloc;
  adv_host_density.mesh.shape.ngi=ngi;
  adv_host_density.val=dev_density_val;
  adv_host_density.field_type=FIELD_TYPE_CONSTANT;
  adv_host_density.mesh.elements=*p_ele_num;
  adv_host_density.mesh.nodes=*p_node_num;
  used_len +=  sizeof(double);

  
  cudaMemcpy(adv_host_t.val,field_t_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(adv_host_rhs.val,field_rhs_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  if((*nvfrac_associated)==0){
  }else{
  cudaMemcpy(adv_host_nvfrac.val,field_nvfrac_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  }
  if((*have_source)==0){
  }else{
  cudaMemcpy(adv_host_source.val,field_source_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  }  
  cudaMemcpy(adv_host_pos.val,field_pos_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice);

   double *d_u_shape_dn=nullptr;
   double *d_lx_shape_dn=nullptr;
   double *d_quadrature_weight=nullptr; 
   int dshape_size = dim*ngi*nloc;
   int detwei_size = (ngi)*(*p_ele_num);
   int ele_count=*p_ele_num;
   used_len = 0;

   d_u_shape_dn =(double*) adv_dev_shape_n;
   used_len += sizeof(double)*dshape_size; 
   d_lx_shape_dn =(double*)(adv_dev_shape_n + used_len); 
   used_len += sizeof(double)*dshape_size;
   d_quadrature_weight = (double*)(adv_dev_shape_n + used_len);
   used_len += sizeof(double)*ngi;

 
    global_adv_dshape = (double*)(adv_dev_shape_n + used_len);
    used_len += sizeof(double)*dshape_size*(ele_count);
    global_adv_detwei = (double*)(adv_dev_shape_n + used_len);
    used_len += sizeof(double) * detwei_size;

    cudaStreamSynchronize(adv_mem_stream);
    int block_size=256;
    dim3 grid = ((ele_count)+block_size-1)/block_size;
    dim3 block = block_size;
   transform_to_physical_full_detwei<3,4,11><<<grid,block,0,adv_mem_stream>>>(adv_host_pos.val,d_u_shape_dn,d_lx_shape_dn,
    d_quadrature_weight, adv_host_pos.mesh.ndglno,ele_count,global_adv_dshape, global_adv_detwei);

  cudaMemcpy(adv_host_velocity.val,field_velocity_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice);
  if((*have_diffu)==0){
  }else{
  cudaMemcpy(adv_host_diffu.val,field_diffu_val,sizeof(double)*dim*dim*node_num,cudaMemcpyHostToDevice);
  }  
  cudaMemcpy(adv_host_density.val,field_density_val,sizeof(double),cudaMemcpyHostToDevice);//
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
return;
}


extern "C" void adv_shape_dn_to_device_(double *u_shape_dn,double *lx_shape_dn,
double *lx_shape_weight){
   const int nloc = 4, ngi=11,dim=3;
   double *d_u_shape_dn=nullptr;
   d_u_shape_dn =(double*) adv_dev_shape_n;
   int64_t dshape_size = dim * ngi * nloc; 
   cudaMemcpyAsync(d_u_shape_dn,u_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,adv_mem_stream);
   cudaMemcpyAsync(d_u_shape_dn + dshape_size,lx_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,adv_mem_stream);
   cudaMemcpyAsync(d_u_shape_dn + 2*dshape_size,lx_shape_weight, sizeof(double)*ngi,cudaMemcpyHostToDevice,adv_mem_stream);
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
}


extern "C" void gpu_adv_assemble_(int *p_ele_num,int *p_nodes,double *p_beta, 
double *p_dt_theta,double *host_rhs_val,int *have_source,int *have_diffu){
   cudaStreamSynchronize(adv_mem_stream);
   int ele_num = *p_ele_num;
   int node_num = *p_nodes;
   double beta = *p_beta;
   double dt_theta = *p_dt_theta;
   double *dev_detwei=nullptr;
   double *dev_rhs_addto=nullptr;
   double *dev_matrix_addto=nullptr;
   double *dev_du_t = nullptr;
   double *dev_dt_t = nullptr;
   printf("adv_assemble,ele_num=%d,node_num=%d\n",ele_num,node_num);fflush(stdout);
  dev_detwei = global_adv_detwei;
  dev_du_t = global_adv_dshape;
  dev_dt_t = global_adv_dshape;
   dev_rhs_addto = global_adv_rhs;
   cudaMemset(dev_rhs_addto,0,4*ele_num*sizeof(double)); 
   dev_matrix_addto = global_adv_matrix;
   global_assemble_matrix= global_adv_matrix; 
   cudaMemset(dev_matrix_addto,0,4*4*ele_num*sizeof(double));   
    
   dim3 threadsPerBlock(256,1,1);
   int num_of_block=(ele_num + 256 -1)/256;
   dim3 blocksPerGrid(num_of_block,1,1); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   gpu_adv_add_mass<<<blocksPerGrid,threadsPerBlock>>>(adv_host_t,ele_num,dev_detwei,dev_matrix_addto); 
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
   
   gpu_adv_add_advection<<<blocksPerGrid,threadsPerBlock>>>(adv_host_velocity,adv_host_t,ele_num,
   dev_du_t, dev_detwei,beta,dt_theta,dev_matrix_addto,dev_rhs_addto);   
   
   if(*have_diffu ==0){
     printf("no diffu!!!!!!!!!!!!!!!!!\n");
   }else{
   gpu_adv_add_diffusivity<3,4,4,11><<<blocksPerGrid,threadsPerBlock>>>(adv_host_diffu,adv_host_t,ele_num,dt_theta,
   dev_dt_t,dev_detwei,dev_matrix_addto,dev_rhs_addto);
   }
   if(*have_source ==0){
     printf("no source!!!!!!!!!!!!!!!!!\n");
   }else{
   gpu_adv_add_source<<<blocksPerGrid,threadsPerBlock>>>(adv_host_t,adv_host_source,ele_num,dev_detwei,dev_rhs_addto); 
   } 
   adv_rhs_addto_kernel<4> <<<blocksPerGrid,threadsPerBlock>>>(adv_host_rhs,ele_num,dev_rhs_addto); 
   cudaMemcpy(host_rhs_val,adv_host_rhs.val,node_num*sizeof(double),cudaMemcpyDeviceToHost);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("GPU  advection diffusion kernel took %f ms\n", milliseconds);
   return;
}

#endif

#define DIVERGENCE_MATRIX 1

#if (DIVERGENCE_MATRIX==1)


GpuScalarField div_host_density;
GpuScalarField div_host_olddensity;
GpuScalarField div_host_pressure;
GpuVectorField div_host_cord;

GpuVectorField div_host_velocity;

cudaStream_t div_stream;

extern "C" void  div_create_stream_(){
   cudaStreamCreateWithFlags(&div_stream,cudaStreamNonBlocking);
   return;
}
extern "C" void div_destroy_stream_(){
 cudaStreamDestroy(div_stream);
return;
}


int8_t *global_div_gpu_ptr=nullptr;
int8_t *global_div_ndglno=nullptr;
int8_t *global_div_field=nullptr;
int8_t *global_div_shape_n=nullptr;
int8_t *global_div_du_detwei=nullptr;

double *global_div_shape_dn=nullptr;
double *global_div_detwei=nullptr;
double *global_div_dshape=nullptr;
double *global_div_matrix = nullptr;

extern "C" void dealloc_div_mem_(){
 cudaFreeAsync(global_div_gpu_ptr,div_stream);
 cudaFreeAsync(global_div_shape_n,div_stream);
 cudaFreeAsync(global_div_du_detwei,div_stream);
 cudaFreeAsync(global_div_field,div_stream);
 cudaFreeAsync(global_div_ndglno,div_stream);
 cudaFreeAsync(global_div_matrix,div_stream);
 return;
}


extern "C" void  div_alloc_gpu_mem_(int *p_ele_num, int *p_node_num){

   const int nloc = 4, ngi=11,dim=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   int64_t total_gpu_len = sizeof(double)*node_num  //pressure
    +sizeof(double)*node_num                //density
    +sizeof(double)*node_num                //olddensity
    +sizeof(double)*dim*node_num            //position
    +sizeof(double)*dim*node_num;            //velocity
    
    int64_t shape_len=sizeof(double)*nloc *ngi;  
    cudaStreamSynchronize(div_stream);
    
    cudaMallocAsync((void**)&global_div_gpu_ptr,total_gpu_len, div_stream);
    
    cudaMallocAsync((void**)&global_div_shape_n,shape_len,div_stream); 
    int ndglno_len = sizeof(int)*nloc*ele_num;
    cudaMallocAsync((void**)&global_div_ndglno,ndglno_len,div_stream); 

    int64_t dshape_size = dim * ngi * nloc; 
    int64_t detwei_size = ngi * ele_num;
    int64_t div_du_detwei_length = sizeof(double)*dshape_size //d_u_shape_dn
    + sizeof(double)*dshape_size + sizeof(double)*ngi    //d_lx_shape_dn,quadrature_weight
    + sizeof(double)*dshape_size *ele_num + sizeof(double)*detwei_size; //d_du_t,d_detwei
    
    cudaMallocAsync((void**)&global_div_du_detwei,div_du_detwei_length,div_stream); 
    cudaMallocAsync(&global_div_matrix,3*4*4*ele_num*sizeof(double),div_stream);
    return;
}
extern "C" void div_shape_dn_to_device_(double *u_shape_dn,double *lx_shape_dn,
double *lx_shape_weight){
   const int nloc = 4, ngi=11,dim=3;
   double *d_u_shape_dn=nullptr;
   d_u_shape_dn =(double*) global_div_du_detwei;
   int64_t dshape_size = dim * ngi * nloc; 
   cudaMemcpyAsync(d_u_shape_dn,u_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,div_stream);
   cudaMemcpyAsync(d_u_shape_dn + dshape_size,lx_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,div_stream);
   cudaMemcpyAsync(d_u_shape_dn + 2*dshape_size,lx_shape_weight, sizeof(double)*ngi,cudaMemcpyHostToDevice,div_stream);
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
}

extern "C" void div_set_gpu_data_(int *p_ele_num, int *p_node_num,
int *field_mesh_ndglno,double*field_mesh_shape_n,
double *field_density_val, double *field_olddensity_val,
double *field_pressure_val,double*field_cord_val,double*field_velocity_val){

   const int nloc = 4, ngi=11,dim=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   cudaStreamSynchronize(div_stream);
   int8_t * dev_ptr = global_div_gpu_ptr;
   int8_t *dev_shape_n = global_div_shape_n;
   
   cudaMemcpyAsync(dev_shape_n,field_mesh_shape_n,4*11*sizeof(double),cudaMemcpyHostToDevice,div_stream);
   
   int *dev_ndglno = (int*)global_div_ndglno;
   cudaMemcpyAsync(dev_ndglno,field_mesh_ndglno,sizeof(int)*nloc*ele_num,cudaMemcpyHostToDevice,div_stream);

   int64_t used_len = 0; 
   double *dev_density_val = (double*)(dev_ptr + used_len); 

       div_host_density.mesh.shape.n = (double*)dev_shape_n;
       div_host_density.mesh.ndglno= dev_ndglno;
       div_host_density.mesh.shape.loc=nloc;
       div_host_density.mesh.shape.ngi=ngi;
       div_host_density.val=dev_density_val;
       div_host_density.field_type=FIELD_TYPE_NORMAL;
       div_host_density.mesh.elements=*p_ele_num;
       div_host_density.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;

   double *dev_olddensity_val = (double*)(dev_ptr + used_len); 

       div_host_olddensity.mesh.shape.n = (double*)dev_shape_n;
       div_host_olddensity.mesh.ndglno= dev_ndglno;
       div_host_olddensity.mesh.shape.loc=nloc;
       div_host_olddensity.mesh.shape.ngi=ngi;
       div_host_olddensity.val=dev_olddensity_val;
       div_host_olddensity.field_type=FIELD_TYPE_NORMAL;
       div_host_olddensity.mesh.elements=*p_ele_num;
       div_host_olddensity.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;

   double *dev_pressure_val = (double*)(dev_ptr + used_len); 

       div_host_pressure.mesh.shape.n = (double*)dev_shape_n;
       div_host_pressure.mesh.ndglno= dev_ndglno;
       div_host_pressure.mesh.shape.loc=nloc;
       div_host_pressure.mesh.shape.ngi=ngi;
       div_host_pressure.val=dev_pressure_val;
       div_host_pressure.field_type=FIELD_TYPE_NORMAL;
       div_host_pressure.mesh.elements=*p_ele_num;
       div_host_pressure.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;

     double *dev_cord_val=(double*)(dev_ptr + used_len);
      div_host_cord.mesh.shape.n = (double*)dev_shape_n;
       div_host_cord.mesh.ndglno= dev_ndglno;
       div_host_cord.mesh.shape.loc=nloc;
       div_host_cord.mesh.shape.ngi=ngi;
       div_host_cord.dim=dim;
       div_host_cord.val=dev_cord_val;
       div_host_cord.field_type=FIELD_TYPE_NORMAL;
       div_host_cord.mesh.elements=*p_ele_num;
       div_host_cord.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*dim*node_num;
      
   /*  double *dev_velocity_val = (double*)(dev_ptr + used_len);
      div_host_velocity.mesh.shape.n = (double*)dev_shape_n;
      div_host_velocity.mesh.ndglno= dev_ndglno;
      div_host_velocity.mesh.shape.loc=nloc;
      div_host_velocity.mesh.shape.ngi=ngi;
      div_host_velocity.dim=dim;
      div_host_velocity.val=dev_velocity_val;
      div_host_velocity.field_type=FIELD_TYPE_NORMAL;
      div_host_velocity.mesh.elements=*p_ele_num;
      div_host_velocity.mesh.nodes=*p_node_num;
      used_len +=  sizeof(double)*dim*node_num;
   */

  cudaMemcpy(div_host_cord.val,field_cord_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice);

   double *d_u_shape_dn=nullptr;
   double *d_lx_shape_dn=nullptr;
   double *d_quadrature_weight=nullptr; 
   int dshape_size = dim*ngi*nloc;
   int detwei_size = (ngi)*(*p_ele_num);
   int ele_count=*p_ele_num;
   used_len = 0;

   d_u_shape_dn =(double*) global_div_du_detwei;
   used_len += sizeof(double)*dshape_size; 
   d_lx_shape_dn =(double*)(global_div_du_detwei + used_len); 
   used_len += sizeof(double)*dshape_size;
   d_quadrature_weight = (double*)(global_div_du_detwei + used_len);
   used_len += sizeof(double)*ngi;
 
    global_div_dshape = (double*)(global_div_du_detwei + used_len);
    used_len += sizeof(double)*dshape_size*(ele_count);
    global_div_detwei = (double*)(global_div_du_detwei + used_len);
    used_len += sizeof(double) * detwei_size;
    cudaStreamSynchronize(div_stream);
    
    int block_size=256;
    dim3 grid = ((ele_count)+block_size-1)/block_size;
    dim3 block = block_size;
   transform_to_physical_full_detwei<3,4,11><<<grid,block,0,div_stream>>>(div_host_cord.val,d_u_shape_dn,d_lx_shape_dn,
    d_quadrature_weight, div_host_cord.mesh.ndglno,ele_count,global_div_dshape, global_div_detwei);
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
 
  cudaMemcpy(div_host_density.val,field_density_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(div_host_olddensity.val,field_olddensity_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(div_host_pressure.val,field_pressure_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);

}
 
extern "C" void gpu_div_mat_assemble_(int *p_ele_num,int *p_nodes,
                double *p_theta){

   cudaStreamSynchronize(div_stream);
   int ele_num = *p_ele_num;
   double theta = *p_theta;
   double *dev_detwei=nullptr;
   double *dev_ele_matrix=nullptr;
   dev_detwei = global_div_detwei;
   double *dev_div_dfield = global_div_dshape;
   dev_ele_matrix=(double*)global_div_matrix;
   global_assemble_matrix=global_div_matrix;
   double *dn=(double*)global_div_du_detwei;
   dim3 threadsPerBlock(256,1,1);
   int num_of_block=(ele_num + 256 -1)/256;
   dim3 blocksPerGrid(num_of_block,1,1); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   gpu_div_ele_mat<<<blocksPerGrid,threadsPerBlock>>>(div_host_pressure,div_host_density,div_host_olddensity,
   ele_num,theta,dn,dev_div_dfield,dev_detwei,dev_ele_matrix); 
   cudaError_t error = cudaGetLastError();
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   error = cudaGetLastError();
   printf("CUDA error: %s\n", cudaGetErrorString(error));
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("GPU  divergence matrix kernel took %f ms\n", milliseconds); fflush(stdout);
}


#endif

#define PROJECT_MATRIX 1
#if (PROJECT_MATRIX==1)


GpuScalarField prj_host_density;
GpuScalarField prj_host_olddensity;
GpuScalarField prj_host_pressure;
GpuScalarField prj_host_drhodp;
GpuScalarField prj_host_rhs;
GpuScalarField prj_host_eospressure;

GpuVectorField prj_host_cord;

cudaStream_t prj_stream;

extern "C" void  prj_create_stream_(){
   cudaStreamCreateWithFlags(&prj_stream,cudaStreamNonBlocking);
   return;
}
extern "C" void prj_destroy_stream_(){
 cudaStreamDestroy(prj_stream);
return;
}


int8_t *global_prj_gpu_ptr=nullptr;
int8_t *global_prj_ndglno=nullptr;
int8_t *global_prj_field=nullptr;
int8_t *global_prj_shape_n=nullptr;
int8_t *global_prj_du_detwei=nullptr;

double *global_prj_shape_dn=nullptr;
double *global_prj_detwei=nullptr;
double *global_prj_dshape=nullptr;

double *global_prj_matrix = nullptr;
double *global_prj_rhs = nullptr;

extern "C" void dealloc_prj_mem_(){
 cudaFreeAsync(global_prj_gpu_ptr,prj_stream);
 cudaFreeAsync(global_prj_shape_n,prj_stream);
 cudaFreeAsync(global_prj_du_detwei,prj_stream);
 cudaFreeAsync(global_prj_field,prj_stream);
 cudaFreeAsync(global_prj_ndglno,prj_stream);
 cudaFreeAsync(global_prj_matrix,prj_stream);
 cudaFreeAsync(global_prj_rhs,prj_stream);
 return;
}

extern "C" void  prj_alloc_gpu_mem_(int *p_ele_num, int *p_node_num){

   const int nloc = 4, ngi=11,dim=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   int64_t total_gpu_len = sizeof(double)*node_num  //pressure
    +sizeof(double)*node_num                //density
    +sizeof(double)*node_num                //olddensity
    +sizeof(double)*node_num                //drhodp
    +sizeof(double)*node_num                //eospressure
    +sizeof(double)*node_num                //rhs
    +sizeof(double)*dim*node_num;            //position
    
    int64_t shape_len=sizeof(double)*nloc *ngi;  
    cudaStreamSynchronize(prj_stream);
    
    cudaMallocAsync((void**)&global_prj_gpu_ptr,total_gpu_len, prj_stream);
    
    cudaMallocAsync((void**)&global_prj_shape_n,shape_len,prj_stream); 
    int ndglno_len = sizeof(int)*nloc*ele_num;
    cudaMallocAsync((void**)&global_prj_ndglno,ndglno_len,prj_stream); 
    int64_t dshape_size = dim * ngi * nloc; 
    int64_t detwei_size = ngi * ele_num;
    int64_t prj_du_detwei_length = sizeof(double)*dshape_size //d_u_shape_dn
    + sizeof(double)*dshape_size + sizeof(double)*ngi    //d_lx_shape_dn,quadrature_weight
    + sizeof(double)*dshape_size *ele_num + sizeof(double)*detwei_size; //d_du_t,d_detwei
    
    cudaMallocAsync(&global_prj_rhs,4*ele_num*sizeof(double),prj_stream);
    cudaMallocAsync((void**)&global_prj_du_detwei,prj_du_detwei_length,prj_stream); 
    cudaMallocAsync(&global_prj_matrix,4*4*ele_num*sizeof(double),prj_stream);
    return;
}


extern "C" void prj_shape_dn_to_device_(double *u_shape_dn,double *lx_shape_dn,
double *lx_shape_weight){
   const int nloc = 4, ngi=11,dim=3;
   double *d_u_shape_dn=nullptr;
   d_u_shape_dn =(double*) global_prj_du_detwei;
   int64_t dshape_size = dim * ngi * nloc; 
   cudaMemcpyAsync(d_u_shape_dn,u_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,prj_stream);
   cudaMemcpyAsync(d_u_shape_dn + dshape_size,lx_shape_dn, sizeof(double)*dshape_size,cudaMemcpyHostToDevice,prj_stream);
   cudaMemcpyAsync(d_u_shape_dn + 2*dshape_size,lx_shape_weight, sizeof(double)*ngi,cudaMemcpyHostToDevice,prj_stream);
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
}

extern "C" void prj_set_gpu_data_(int *p_ele_num, int *p_node_num,
int *field_mesh_ndglno,double*field_mesh_shape_n,
double *field_density_val, double *field_olddensity_val,
double *field_pressure_val,double *field_drhodp_val,
double *field_eospressure_val,double *field_rhs_val,double*field_cord_val){

   const int nloc = 4, ngi=11,dim=3;
   int ele_num = *p_ele_num;
   int node_num = *p_node_num;
   cudaStreamSynchronize(prj_stream);
   int8_t * dev_ptr = global_prj_gpu_ptr;
   int8_t *dev_shape_n = global_prj_shape_n;
    
   cudaMemcpyAsync(dev_shape_n,field_mesh_shape_n,4*11*sizeof(double),cudaMemcpyHostToDevice,prj_stream);
   
   int *dev_ndglno = (int*)global_prj_ndglno;
   cudaMemcpyAsync(dev_ndglno,field_mesh_ndglno,sizeof(int)*nloc*ele_num,cudaMemcpyHostToDevice,prj_stream);
   int64_t used_len = 0; 
   double *dev_density_val = (double*)(dev_ptr + used_len); 
       prj_host_density.mesh.shape.n = (double*)dev_shape_n;
       prj_host_density.mesh.ndglno= dev_ndglno;
       prj_host_density.mesh.shape.loc=nloc;
       prj_host_density.mesh.shape.ngi=ngi;
       prj_host_density.val=dev_density_val;
       prj_host_density.field_type=FIELD_TYPE_NORMAL;
       prj_host_density.mesh.elements=*p_ele_num;
       prj_host_density.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;


   double *dev_olddensity_val = (double*)(dev_ptr + used_len); 
       prj_host_olddensity.mesh.shape.n = (double*)dev_shape_n;
       prj_host_olddensity.mesh.ndglno= dev_ndglno;
       prj_host_olddensity.mesh.shape.loc=nloc;
       prj_host_olddensity.mesh.shape.ngi=ngi;
       prj_host_olddensity.val=dev_olddensity_val;
       prj_host_olddensity.field_type=FIELD_TYPE_NORMAL;
       prj_host_olddensity.mesh.elements=*p_ele_num;
       prj_host_olddensity.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;


   double *dev_pressure_val = (double*)(dev_ptr + used_len); 
       prj_host_pressure.mesh.shape.n = (double*)dev_shape_n;
       prj_host_pressure.mesh.ndglno= dev_ndglno;
       prj_host_pressure.mesh.shape.loc=nloc;
       prj_host_pressure.mesh.shape.ngi=ngi;
       prj_host_pressure.val=dev_pressure_val;
       prj_host_pressure.field_type=FIELD_TYPE_NORMAL;
       prj_host_pressure.mesh.elements=*p_ele_num;
       prj_host_pressure.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;
   
   double *dev_drhodp_val = (double*)(dev_ptr + used_len); 
       prj_host_drhodp.mesh.shape.n = (double*)dev_shape_n;
       prj_host_drhodp.mesh.ndglno= dev_ndglno;
       prj_host_drhodp.mesh.shape.loc=nloc;
       prj_host_drhodp.mesh.shape.ngi=ngi;
       prj_host_drhodp.val=dev_drhodp_val;
       prj_host_drhodp.field_type=FIELD_TYPE_NORMAL;
       prj_host_drhodp.mesh.elements=*p_ele_num;
       prj_host_drhodp.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;


   double *dev_eospressure_val = (double*)(dev_ptr + used_len); 
       prj_host_eospressure.mesh.shape.n = (double*)dev_shape_n;
       prj_host_eospressure.mesh.ndglno= dev_ndglno;
       prj_host_eospressure.mesh.shape.loc=nloc;
       prj_host_eospressure.mesh.shape.ngi=ngi;
       prj_host_eospressure.val=dev_eospressure_val;
       prj_host_eospressure.field_type=FIELD_TYPE_NORMAL;
       prj_host_eospressure.mesh.elements=*p_ele_num;
       prj_host_eospressure.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;
   
  double *dev_rhs_val = (double*)(dev_ptr + used_len); 
       prj_host_rhs.mesh.shape.n = (double*)dev_shape_n;
       prj_host_rhs.mesh.ndglno= dev_ndglno;
       prj_host_rhs.mesh.shape.loc=nloc;
       prj_host_rhs.mesh.shape.ngi=ngi;
       prj_host_rhs.val=dev_rhs_val;
       prj_host_rhs.field_type=FIELD_TYPE_NORMAL;
       prj_host_rhs.mesh.elements=*p_ele_num;
       prj_host_rhs.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*node_num;

     double *dev_cord_val=(double*)(dev_ptr + used_len);
       prj_host_cord.mesh.shape.n = (double*)dev_shape_n;
       prj_host_cord.mesh.ndglno= dev_ndglno;
       prj_host_cord.mesh.shape.loc=nloc;
       prj_host_cord.mesh.shape.ngi=ngi;
       prj_host_cord.dim=dim;
       prj_host_cord.val=dev_cord_val;
       prj_host_cord.field_type=FIELD_TYPE_NORMAL;
       prj_host_cord.mesh.elements=*p_ele_num;
       prj_host_cord.mesh.nodes=*p_node_num;
     used_len +=  sizeof(double)*dim*node_num;

  cudaMemcpy(prj_host_cord.val,field_cord_val,sizeof(double)*dim*node_num,cudaMemcpyHostToDevice);
  
    double *d_u_shape_dn=nullptr;
   double *d_lx_shape_dn=nullptr;
   double *d_quadrature_weight=nullptr; 
   int dshape_size = dim*ngi*nloc;
   int detwei_size = (ngi)*(*p_ele_num);
   int ele_count=*p_ele_num;
   used_len = 0;

   d_u_shape_dn =(double*) global_prj_du_detwei;
   used_len += sizeof(double)*dshape_size; 
   d_lx_shape_dn =(double*)(global_prj_du_detwei + used_len); 
   used_len += sizeof(double)*dshape_size;
   d_quadrature_weight = (double*)(global_prj_du_detwei + used_len);
   used_len += sizeof(double)*ngi;
 
    global_prj_dshape = (double*)(global_prj_du_detwei + used_len);
    used_len += sizeof(double)*dshape_size*(ele_count);
    global_prj_detwei = (double*)(global_prj_du_detwei + used_len);
    used_len += sizeof(double) * detwei_size;
    cudaStreamSynchronize(prj_stream);
    int block_size=256;
    dim3 grid = ((ele_count)+block_size-1)/block_size;
    dim3 block = block_size;
   transform_to_physical_full_detwei<3,4,11><<<grid,block,0,prj_stream>>>(prj_host_cord.val,d_u_shape_dn,d_lx_shape_dn,
    d_quadrature_weight,prj_host_cord.mesh.ndglno, ele_count,global_prj_dshape, global_prj_detwei);
  cudaMemcpy(prj_host_density.val,field_density_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(prj_host_olddensity.val,field_olddensity_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(prj_host_pressure.val,field_pressure_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(prj_host_drhodp.val,field_drhodp_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(prj_host_eospressure.val,field_eospressure_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
  cudaMemcpy(prj_host_rhs.val,field_rhs_val,sizeof(double)*node_num,cudaMemcpyHostToDevice);
    cudaStreamSynchronize(prj_stream);

}


extern "C" void gpu_prj_mat_assemble_(int *p_ele_num, int *p_nodes,
          double *p_atmos_pressure,double *p_dt, double *p_theta_div,
          double *p_theta_pg,double *host_rhs_val){

   cudaStreamSynchronize(prj_stream);
   int ele_num = *p_ele_num;
   int node_num = *p_nodes;
   double atmos_pressure = *p_atmos_pressure;
   double dt = *p_dt;
   double theta_div = *p_theta_div;
   double theta_pg = *p_theta_pg;
   double *dev_detwei=nullptr;
   double *dev_ele_matrix=nullptr;
   dev_detwei = global_prj_detwei;
   dev_ele_matrix=(double*)global_prj_matrix;
   double *dev_ele_rhs = (double*)global_prj_rhs; 
   
   global_assemble_matrix=global_prj_matrix;
 
   double invdt = (1.0)/dt;
   double factor =(1.0)/(dt*dt*theta_div*theta_pg);

   dim3 threadsPerBlock(256,1,1);
   int num_of_block=(ele_num + 256 -1)/256;
   dim3 blocksPerGrid(num_of_block,1,1); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   gpu_prj_ele_mat<<<blocksPerGrid,threadsPerBlock>>>(prj_host_density,prj_host_olddensity,prj_host_pressure,prj_host_drhodp,prj_host_eospressure, ele_num,atmos_pressure,invdt,factor,dev_detwei,dev_ele_matrix,dev_ele_rhs);
   
   adv_rhs_addto_kernel<4><<<blocksPerGrid,threadsPerBlock>>>(prj_host_rhs,ele_num,dev_ele_rhs);   
   cudaMemcpy(host_rhs_val,prj_host_rhs.val,node_num*sizeof(double),cudaMemcpyDeviceToHost);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaError_t error = cudaGetLastError();
   printf("CUDA error: %s@%d\n", cudaGetErrorString(error),__LINE__);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
}


#endif


//哈哈马哥进度，因为各个ele的元素互不影响，或者说内在联系还不清楚，
//在将ele的数据划分给线程时，没有多加考虑。但是如果之后发现transform有体现的话再说
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


int *adv_d_ndglno=nullptr;
double *adv_d_value=nullptr;
int *adv_d_findrm=nullptr;
int *adv_d_colm=nullptr;
extern double *global_assemble_matrix;
extern "C" void alloc_adv_assemble_(int* csr_findrm, int* csr_colm, int* mesh_ndglno,
    int* loc, int* ngi, int* ele_num, int* row_num,int* val_num){
    
    cudaMalloc((void**)&adv_d_value, sizeof(double)*(*val_num));//size(kt%val,1)
    cudaMalloc((void**)&adv_d_findrm, sizeof(int)*(*row_num));
    cudaMalloc((void**)&adv_d_colm, sizeof(int)*(*val_num));
    cudaMalloc((void**)&adv_d_ndglno, sizeof(int)*(*ele_num)*4);
    
    cudaMemcpy(adv_d_findrm,csr_findrm,sizeof(int)*(*row_num),cudaMemcpyHostToDevice);
    cudaMemcpy(adv_d_colm,csr_colm,sizeof(int)*(*val_num),cudaMemcpyHostToDevice);
    cudaMemcpy(adv_d_ndglno,mesh_ndglno, sizeof(int)*(*ele_num)*4,cudaMemcpyHostToDevice);
}
extern "C" void dealloc_adv_assemble_(){
     cudaFree(adv_d_value);
     cudaFree(adv_d_findrm);
     cudaFree(adv_d_colm);
     cudaFree(adv_d_ndglno);
}
extern "C" void assemble_csr_matrix_gpu_(double* csr_val, 
    int* loc, int* ngi, int* ele_num, int* row_num,int* val_num){
        //这不是coordinate里的ndglno，而是pressure_mesh%ndglno
    int* d_ndglno=adv_d_ndglno;
    int iidim=1; 
    int *dim=&iidim;
    double* d_value=adv_d_value;
    int* d_findrm=adv_d_findrm;
    int* d_colm=adv_d_colm;
    double *d_matrix_addto=global_assemble_matrix;
    cudaError_t error;
    //--------------------immediate

    cudaMemcpy(d_value,csr_val,sizeof(double)*(*val_num),cudaMemcpyHostToDevice);
    //dshape_tensor_dshape
    int num_block=(*ele_num + 256 - 1) / 256;
    dim3 grid(num_block);
    dim3 block(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    assemble_csr_matrix<<<grid, block>>>(d_matrix_addto,d_value,d_findrm,d_colm,d_ndglno,*ele_num, *dim, *loc, *ngi);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel assemble_csr_matrix took %f ms\n", milliseconds);
    // copy result value to CSR_matrix%val
    cudaMemcpy(csr_val, d_value, sizeof(double)*(*val_num),cudaMemcpyDeviceToHost);
    //check result
}

extern "C" void div_assemble_csr_matrix_gpu_(double* csr_val, 
    int* loc, int* ngi, int* ele_num, int* row_num,int* val_num,int *blocki){
        //这不是coordinate里的ndglno，而是pressure_mesh%ndglno
    int* d_ndglno=adv_d_ndglno;
    int iidim=3; 
    int *dim=&iidim;
    double* d_value=adv_d_value;
    int* d_findrm=adv_d_findrm;
    int* d_colm=adv_d_colm;
    double *d_matrix_addto=global_assemble_matrix;
    cudaError_t error;
    int bi = (*blocki - 1);
    cudaMemcpy(d_value,csr_val,sizeof(double)*(*val_num),cudaMemcpyHostToDevice);
    //dshape_tensor_dshape
    int num_block=(*ele_num + 256 - 1) / 256;
    dim3 grid(num_block);
    dim3 block(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    assemble_csr_matrix<<<grid, block>>>(d_matrix_addto,d_value,d_findrm,d_colm,d_ndglno,*ele_num, *dim, *loc, *ngi,bi);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel assemble_csr_matrix took %f ms\n", milliseconds);
    // copy result value to CSR_matrix%val
    cudaMemcpy(csr_val, d_value, sizeof(double)*(*val_num),cudaMemcpyDeviceToHost);
    //check result
}
    
    

extern double *kmk_d_dshape;
extern double *kmk_d_detwei;
extern double *kmk_d_hbar;
double *global_d_value=nullptr;
int *global_d_findrm=nullptr;
int *global_d_colm=nullptr;
int *global_d_ndglno=nullptr;
extern "C" void alloc_assemble_transfer_(double* csr_val, double* csr_findrm, double* csr_colm, int* mesh_ndglno,int* dim, int* loc, int* ngi, int* ele_num, int* row_num,int* val_num){
    cudaMalloc((void**)&global_d_value, sizeof(double)*(*val_num));//size(kt%val,1)
    cudaMalloc((void**)&global_d_findrm, sizeof(int)*(*row_num));
    cudaMalloc((void**)&global_d_colm, sizeof(int)*(*val_num));
    cudaMalloc((void**)&global_d_ndglno, sizeof(int)*(*ele_num)*4);
    
    cudaMemcpy(global_d_findrm,csr_findrm,sizeof(int)*(*row_num),cudaMemcpyHostToDevice);
    cudaMemcpy(global_d_colm,csr_colm,sizeof(int)*(*val_num),cudaMemcpyHostToDevice);
    cudaMemcpy(global_d_ndglno,mesh_ndglno, sizeof(int)*(*ele_num)*4,cudaMemcpyHostToDevice);
}
extern "C" void dealloc_assemble_(){
  cudaFree(global_d_value);
  cudaFree(global_d_findrm);
  cudaFree(global_d_colm);
  cudaFree(global_d_ndglno);
}

extern "C" void assemble_stiff_matrix_gpu_(double* csr_val, double* csr_findrm, double* csr_colm, int* mesh_ndglno,
    int* dim, int* loc, int* ngi, int* ele_num, int* row_num,int* val_num){
    //这不是coordinate里的ndglno，而是pressure_mesh%ndglno
    double* d_dshape=kmk_d_dshape;
    double* d_detwei=kmk_d_detwei;
    double* d_hbar=kmk_d_hbar;
    
    
    double* d_value=global_d_value;
    int* d_findrm=global_d_findrm;
    int* d_colm = global_d_colm;
    int* d_ndglno=global_d_ndglno;
    
    cudaError_t error;
    cudaMemcpy(d_value,csr_val,sizeof(double)*(*val_num),cudaMemcpyHostToDevice);
    int num_block=(*ele_num + 256 - 1) / 256;
    dim3 grid(num_block);
    dim3 block(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    assemble_stiff_matrix<<<grid, block>>>(d_dshape, d_hbar, d_dshape, d_detwei,d_value,d_findrm,d_colm,d_ndglno,*ele_num, *dim, *loc, *ngi);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel assemble_stiff_matrix took %f ms\n", milliseconds);

    // copy result value to CSR_matrix%val
    cudaMemcpy(csr_val, d_value, sizeof(double)*(*val_num),cudaMemcpyDeviceToHost);
}

extern "C" void allocate_cuda_pinned_mem_(double** invJ_cache, double** J_T_cache, double** detJ_cache, int* fdim, int* felements){
    int dim = *fdim;
    int ele = *felements;

    CHECK(cudaHostAlloc((void**)invJ_cache, dim * dim * ele * sizeof(double), cudaHostAllocDefault));

    CHECK(cudaHostAlloc((void**)J_T_cache, dim * dim * ele * sizeof(double), cudaHostAllocDefault));

    CHECK(cudaHostAlloc((void**)detJ_cache, ele * sizeof(double), cudaHostAllocDefault));

}

extern "C" void deallocate_cuda_pinned_mem_(double** invJ_cache, double** J_T_cache, double** detJ_cache){
    cudaFreeHost(*invJ_cache);
    cudaFreeHost(*J_T_cache);
    cudaFreeHost(*detJ_cache);
}

cudaStream_t stream_get_edge;
cudaStream_t stream_transform;

/////////
double *x_d_field_val=nullptr;
int *x_d_ndglno=nullptr;
double *x_d_shape_dn=nullptr;
double *x_d_lx_shape_dn=nullptr;
double *x_d_quadrature_weight=nullptr;
double *kmk_d_dshape=nullptr;
double *kmk_d_detwei=nullptr;
double *kmk_d_hbar=nullptr;

extern "C" void kmk_gpu_alloc_(int *dim, int *ngi,int *loc,
           int *ele_count, int *vertex_num){

    int dshape_size = (*dim)*(*ngi)*(*loc);
    int detwei_size = (*ngi)*(*ele_count);
    int ndglno_size = (*loc)*(*ele_count);
    cudaMalloc((void**)&x_d_field_val,sizeof(double)*(*vertex_num)*(*dim));
    cudaMalloc((void**)&x_d_shape_dn,sizeof(double)*dshape_size);
    cudaMalloc((void**)&x_d_lx_shape_dn,sizeof(double)*dshape_size);
    cudaMalloc((void**)&x_d_quadrature_weight,sizeof(double)*(*ngi));
    cudaMalloc((void**)&x_d_ndglno,sizeof(int)*(ndglno_size));
    
    cudaMalloc((void**)&kmk_d_dshape,sizeof(double)*dshape_size*(*ele_count));//存放计算结果，之后给后续kernel用
    cudaMalloc((void**)&kmk_d_detwei,sizeof(double)*detwei_size);
    CHECK(cudaMalloc((void**)&kmk_d_hbar, sizeof(double)*(*ele_count)*(*ngi)*(*dim)*(*dim)));
}

extern "C" void kmk_gpu_dealloc_(){
  cudaFree(x_d_field_val);
  cudaFree(x_d_shape_dn);
  cudaFree(x_d_lx_shape_dn);
  cudaFree(x_d_quadrature_weight);
  cudaFree(x_d_ndglno);
  cudaFree(kmk_d_dshape);
  cudaFree(kmk_d_detwei);
  cudaFree(kmk_d_hbar);
}


extern "C" void transform_to_physical_full_gpu_c_(double* field_val,double* shape_dn,double* lx_shape_dn,
        double* quadrature_weight, int * ndglno, int* vertex_num, int* ndglno_size,
        bool* x_spherical, bool* x_nonlinear, bool* m_nonlinear,
        int* dim, int* ngi, int* loc, int* ele_count,
        bool* detwei_flag){
        
        //这俩在本函数内创:double* dshape,double* detwei,ptr拷回地址
        //r_dshape/detwei对比计算结果


    int dshape_size = (*dim)*(*ngi)*(*loc);

    //计算使用数据
    double* d_field_val=x_d_field_val;
    double* d_shape_dn = x_d_shape_dn;
    double* d_lx_shape_dn=x_d_lx_shape_dn;
    //double* d_quadrature_wet;//
    int *d_ndglno = x_d_ndglno;

    cudaMemcpy(d_field_val, field_val, sizeof(double)*(*vertex_num)*(*dim), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape_dn, shape_dn, sizeof(double)*dshape_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lx_shape_dn, lx_shape_dn, sizeof(double)*dshape_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ndglno, ndglno, sizeof(int)*(*ndglno_size), cudaMemcpyHostToDevice);

    //GPU上计算结果
    double* d_dshape = kmk_d_dshape;
    double* d_detwei = kmk_d_detwei;
    
    bool detwei_exist=*detwei_flag; //在CPU判断，开启不同kernel

    int block_size=256;
    dim3 grid = ((*ele_count)+block_size-1)/block_size;
    dim3 block =block_size ;

    if(detwei_exist){
        //在fortran那看下有没present detwei，但是我们可以不用他的detwei，没用有危害吗？
        double* d_quadrature_weight=x_d_quadrature_weight;
        cudaMemcpy(d_quadrature_weight,quadrature_weight,sizeof(double)*(*ngi), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
//kernel
        transform_to_physical_full_detwei<3,4,11><<<grid,block>>>(d_field_val,d_shape_dn,d_lx_shape_dn,
        d_quadrature_weight, d_ndglno, *ele_count,
        d_dshape, d_detwei);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel transform_to_physical_full_detwei took %f ms\n", milliseconds);

    }else{
    }

}

