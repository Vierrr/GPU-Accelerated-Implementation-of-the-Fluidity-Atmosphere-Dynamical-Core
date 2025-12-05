#ifndef FIELDS_H
#define FIELDS_H

#include <string>
#include <cstdio>
using namespace std;

enum GpuFieldType {FIELD_TYPE_NORMAL=0, FIELD_TYPE_CONSTANT=1, FIELD_TYPE_PYTHON=2,FIELD_TYPE_DEFERRED=3};

enum GpuEleNumberingTypeEnum{
ELEMENT_LAGRANGIAN=1, ELEMENT_NONCONFORMING=2, ELEMENT_BUBBLE=3, 
ELEMENT_CONTROLVOLUMEBDY_SURFACE=4, ELEMENT_CONTROLVOLUME_SURFACE=5, 
ELEMENT_CONTROLVOLUME_SURFACE_BODYDERIVATIVES=6, ELEMENT_TRACE=7
};
enum GpuFamilyEnum{
FAMILY_SIMPLEX=1, FAMILY_CUBE=2
};

struct GpuEleNumberingType{
int faces,vertices,edges,boundaries;
int degree;
int dimension;
int nodes;
GpuEleNumberingTypeEnum type;
GpuFamilyEnum family;
int ***count2number;
int **number2count;
int *boundary_coord;
int *boundary_val; 
};

struct GpuQuadratureType{
 int dim;
 int degree;
 int vertices;
 int ngi;
 double  *weight;
 double **l; //!! Locations of quadrature points.
 char name[4];
 GpuFamilyEnum family; 
};

struct GpuPolynomial{
  double *coefs;
  int degree;
  GpuPolynomial(){
    degree=-1;
  };
};
struct GpuElementType{
//     !!< Type to encode shape and quadrature information for an element.
 int dim;// !! 2d or 3d?
 int loc;//    integer :: loc !! Number of nodes.
 int ngi;//    integer :: ngi !! Number of gauss points.
 int degree;//    integer :: degree !! Polynomial degree of element.
 //    !! Shape functions: n is for the primitive function, dn is for partial derivatives, dn_s is for partial derivatives on surfaces. 
 //    !! n is loc x ngi, dn is loc x ngi x dim
 //    !! n_s is loc x sngi, dn_s is loc x sngi x dim
 //    !! NOTE that both n_s and dn_s need to be reoriented before use so that they align with the arbitrary facet node ordering.
 double *n,*dn;//    real, pointer :: n(:,:)=>null(), dn(:,:,:)=>null()
 double *n_s;
 double *dn_s;//    real, pointer :: n_s(:,:)=>null(), dn_s(:,:,:)=>null()
 //    !! Polynomials defining shape functions and their derivatives.
 GpuPolynomial *spoly, *dspoly; //   type(polynomial), dimension(:,:), pointer :: spoly=>null(), dspoly=>null()
 //    !! Link back to the node numbering used for this element.
 GpuEleNumberingType *numbering; 
 //    type(ele_numbering_type), pointer :: numbering=>null()
 //    !! Link back to the quadrature used for this element.
 GpuQuadratureType *quadrature;//    type(quadrature_type) :: quadrature
 GpuQuadratureType *surface_quadrature;//    type(quadrature_type), pointer :: surface_quadrature=>null()
 //    !! Pointer to the superconvergence data for this element.
 //    type(superconvergence_type), pointer :: superconvergence=>null()
 //    !! Pointer to constraints data for this element
 //    type(constraints_type), pointer :: constraints=>null()
 //    !! Reference count to prevent memory leaks.
 //    type(refcount_type), pointer :: refcount=>null()
 //    !! Dummy name to satisfy reference counting
  char name[8];//   character(len=0) :: name
}; 

struct GpuMeshType{
//     !!< Mesh information for (among other things) fields.
 int *ndglno;//    integer, dimension(:), pointer :: ndglno
//     !! Flag for whether ndglno is allocated
 int  wrapped;
 //int     type(element_type) :: shape
 int elements;
 int nodes; 
 GpuElementType shape;
 char name[32]; //  character(len=FIELD_NAME_LEN) :: name
 GpuMeshType(){
  wrapped=1;
  elements=0;
  nodes=0;
 };
}; 

struct  GpuVectorField{
//dim x nonods vector values
  double *val;
  int  wrapped;
  GpuFieldType field_type;
  int dim;
  GpuMeshType mesh;
  GpuVectorField(){
  wrapped=1;
  field_type=FIELD_TYPE_NORMAL;
  dim=0;
 };
 void serialize(const char *varname){
  string binname=varname;
  binname +=".vector.dat";
  FILE *serial = fopen(binname.c_str(),"wb");
  fwrite(&dim, sizeof(dim),1,serial);
  fwrite(&mesh.elements,sizeof(int),1,serial);
  fwrite(&mesh.nodes,sizeof(int),1,serial);
  fwrite(&field_type,sizeof(field_type),1,serial);
  fwrite(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fwrite(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fwrite(val,sizeof(double),dim * mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 };
  
  void unserialize(const char *varname){
  string binname=varname;
  binname +=".vector.dat";
  FILE *serial = fopen(binname.c_str(),"rb"); 
  fread(&dim, sizeof(dim),1,serial);
  fread(&mesh.elements,sizeof(int),1,serial);
  fread(&mesh.nodes,sizeof(int),1,serial);
  fread(&field_type,sizeof(field_type),1,serial);
  fread(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fread(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fread(val,sizeof(double),dim * mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 }


};

struct  GpuTensorField{
//dim x nonods vector values
  double *val;
  int  wrapped;
  GpuFieldType field_type;
  int dim[2];
  GpuMeshType mesh;
  GpuTensorField(){
  wrapped=1;
  field_type=FIELD_TYPE_NORMAL;
  dim[0]=0;dim[1]=0;
 };
 void serialize(const char *varname){
  string binname=varname;
  binname +=".tensor.dat";
  FILE *serial = fopen(binname.c_str(),"wb");
  fwrite(&dim[0], sizeof(dim[0]),1,serial);
  fwrite(&dim[1], sizeof(dim[1]),1,serial);
  fwrite(&mesh.elements,sizeof(int),1,serial);
  fwrite(&mesh.nodes,sizeof(int),1,serial);
  fwrite(&field_type,sizeof(field_type),1,serial);
  fwrite(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fwrite(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fwrite(val,sizeof(double),dim[0]*dim[1] * mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 };
  
  void unserialize(const char *varname){
  string binname=varname;
  binname +=".tensor.dat";
  FILE *serial = fopen(binname.c_str(),"rb"); 
  fread(&dim[0], sizeof(dim[0]),1,serial);
  fread(&dim[1], sizeof(dim[1]),1,serial);
  fread(&mesh.elements,sizeof(int),1,serial);
  fread(&mesh.nodes,sizeof(int),1,serial);
  fread(&field_type,sizeof(field_type),1,serial);
  fread(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fread(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fread(val,sizeof(double),dim[0]*dim[1] * mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 }

};


struct GpuScalarField{
  double *val;
  int  wrapped;
  GpuFieldType field_type;
  GpuMeshType mesh;
  GpuScalarField(){
  wrapped=1;
  field_type=FIELD_TYPE_NORMAL;
 };
 void serialize(const char *varname){
  string binname=varname;
  binname +=".scalar.dat";
  FILE *serial = fopen(binname.c_str(),"wb");
  fwrite(&mesh.elements,sizeof(int),1,serial);
  fwrite(&mesh.nodes,sizeof(int),1,serial);
  fwrite(&field_type,sizeof(field_type),1,serial);
  fwrite(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fwrite(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fwrite(val,sizeof(double), mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 };
  
  void unserialize(const char *varname){
  string binname=varname;
  binname +=".scalar.dat";
  FILE *serial = fopen(binname.c_str(),"rb"); //for copy and paste
  fread(&mesh.elements,sizeof(int),1,serial);
  fread(&mesh.nodes,sizeof(int),1,serial);
  fread(&field_type,sizeof(field_type),1,serial);
  fread(mesh.shape.n,sizeof(double),mesh.shape.loc*mesh.shape.ngi,serial);
  fread(mesh.ndglno,sizeof(int),mesh.shape.loc*mesh.elements,serial); 
  fread(val,sizeof(double),mesh.nodes,serial);
  fclose(serial);
  serial=nullptr;
 }
};


#endif 
