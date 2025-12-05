
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <cmath>

extern "C" {

void write_vtk_(double *points,int *p_numpoint,int *ndglno, int *p_numele, double *val,int *bcndglno,int *p_bcnum,char *name,int*namelen){
fflush(stdout);
int numpoint=(*p_numpoint)/3;
int numele=(*p_numele);
printf("bcnum=%d\n",*p_bcnum);fflush(stdout);

std::string vtkname(name,*namelen);
vtkname+=".vtk";
FILE *vtkfile=fopen(vtkname.c_str(),"wb");
fprintf(vtkfile,"# vtk DataFile Version 2.0\n%s\nASCII\n",vtkname.c_str());
fprintf(vtkfile,"DATASET UNSTRUCTURED_GRID\n");

fprintf(vtkfile,"\n");
fprintf(vtkfile,"POINTS %d double\n",numpoint);
for(int ii = 0; ii < numpoint; ++ii){
fprintf(vtkfile,"%f %f %f\n",points[ii*3], points[ii*3 + 1], points[ii*3+2]);
}
fprintf(vtkfile,"\n");
int bcnum=*p_bcnum;

fprintf(vtkfile,"CELLS %d %d\n",numele+bcnum,numele*5 + bcnum * 4);
for(int ii = 0; ii < numele; ++ii){
fprintf(vtkfile,"%d\t%d %d %d %d\n",4,ndglno[ii*4]-1, ndglno[ii*4 + 1]-1, ndglno[ii*4 + 2]-1, ndglno[ii*4 + 3]-1);
}
for(int ii = 0; ii < bcnum; ++ii){
fprintf(vtkfile,"%d\t%d %d %d\n",3,bcndglno[ii*3]-1, bcndglno[ii*3 + 1]-1, bcndglno[ii*3 + 2]-1);
}
fprintf(vtkfile,"\n");

fprintf(vtkfile,"CELL_TYPES %d\n",numele + bcnum);

for(int ii = 0; ii < numele; ++ii){
fprintf(vtkfile,"10\n");
}
for(int ii = 0; ii < bcnum; ++ii){
fprintf(vtkfile,"5\n");
}
fprintf(vtkfile,"\n");

fprintf(vtkfile,"POINT_DATA %d\n",numpoint);
fprintf(vtkfile,"SCALARS rhs double 1\nLOOKUP_TABLE default\n");

for(int ii = 0; ii < numpoint; ++ii){
fprintf(vtkfile,"%f\n",val[ii]);
}
fprintf(vtkfile,"\n");

fprintf(vtkfile,"CELL_DATA %d\n",numele + bcnum);
fprintf(vtkfile,"SCALARS bc double 1\nLOOKUP_TABLE default\n");

for(int ii = 0; ii < numele; ++ii){
fprintf(vtkfile,"10\n");
}
for(int ii = 0; ii < bcnum; ++ii){
fprintf(vtkfile,"100\n");
}
fprintf(vtkfile,"\n");

fclose(vtkfile);
vtkfile=nullptr;

}

}//extern "C"

extern "C"{
static std::vector<double> all_init_val; 
static std::map<std::string,int> field_index_map;
static int init_count = 0;
void easy_extract_initial_condition(int pointx) {
    // Extract the variables from a file (initial_condition_deltaz1000.txt)
    std::string initial_condition = "initial_condition_deltaz1000.txt";
    std::ifstream file_to_read(initial_condition);
    while (!file_to_read.eof()) {
        double Pres_trans, WaVa_trans, Dens_trans, PoTem_trans, Pai_trans, Temp_trans;
        file_to_read >> Pres_trans >> WaVa_trans >> Dens_trans >> PoTem_trans >> Pai_trans >> Temp_trans;
        all_init_val.push_back(Pres_trans);
        all_init_val.push_back(WaVa_trans);
        all_init_val.push_back(Dens_trans);
        all_init_val.push_back(PoTem_trans);
        all_init_val.push_back(Pai_trans);
        Temp_trans /=100.0;
        all_init_val.push_back(Temp_trans);
    }
    file_to_read.close();

    field_index_map.insert(std::make_pair("InitialP",0));
    field_index_map.insert(std::make_pair("InitialWaterVapor",1));
    field_index_map.insert(std::make_pair("InitialDensity",2));
    field_index_map.insert(std::make_pair("InitialPT",3));
    field_index_map.insert(std::make_pair("InitialPai",4));
    field_index_map.insert(std::make_pair("InitialT",5));
}




void cpp_field_init_(double *X, double *Y, double *Z,
                double *result,int *num_nodes,int *flag,
                int *name_len,char*field_name){
    if(all_init_val.size()==0){
      easy_extract_initial_condition(0);
    }
    std::string field_str(field_name,*name_len); 
    auto iter = field_index_map.find(field_str);
    int index = 0;//field_index_map[field_str]; 
    if(iter != field_index_map.end()){
       index = iter->second;
       *flag = 1; 
    } else{
       *flag = 0;
       return;
    }
    double domainx_cal = 80000.0;
    double domainy_cal = 80000.0;
    double domainz_cal = 20000.0;
    int pointx = 21;
    int pointy = 21;
    int pointz = 21;
    double dx = domainx_cal / (pointx - 1);
    double dy = domainy_cal / (pointy - 1);
    double dz = domainz_cal / (pointz - 1);
    for(int i = 0; i < *num_nodes; ++i){
    //int x_pos = std::floor(X[i]/dx);
    //int y_pos = std::floor(Y[i]/dy);  
    int z_pos = std::floor(Z[i]/dz);  //only z
    int val_pos = 6*z_pos + index; 
    result[i]=all_init_val[val_pos];
    }
}


std::vector<std::vector<double>> extract_partial_basic_state_PT_Z(int pointx, int pointx_par, int pointz_par) {
    // Extract the partial basic state from MATLAB
    std::string partialbasicstate = "partial_basic_state_Z_1000_8000.txt";
    std::vector<std::vector<double>> arr(pointx_par);
    std::vector<std::vector<double>> arr_final(pointx_par);

    for (int i = 0; i < pointx_par; ++i) {
        std::vector<double> col;
        arr[i] = col;
        arr_final[i] = col;
    }

    std::ifstream file_to_read(partialbasicstate);

    if (file_to_read.is_open()) {
        std::string line;
        while (std::getline(file_to_read, line)) {
            std::vector<double> arr_inter;
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                arr_inter.push_back(val);
            }

            for (int k = 0; k < pointx_par; ++k) {
                arr[k].push_back(arr_inter[k]);
            }
        }

        for (int p = 0; p < pointx_par; ++p) {
            arr_final[p] = std::vector<double>(arr[p].begin(), arr[p].end());
        }
    }
    else {
        std::cerr << "Unable to open file: " << partialbasicstate << std::endl;
    }

    std::vector<std::vector<double>>::size_type num = arr_final[0].size();
    std::vector<std::vector<double>> output(pointx_par, std::vector<double>(num, 0.0));

    for (std::vector<std::vector<double>>::size_type i = 0; i < num; ++i) {
        for (std::vector<std::vector<double>>::size_type q = 0; q < pointx_par; ++q) {
            output[q][i] = arr_final[q][i];
        }
    }

    return output;
}

std::vector<std::vector<double>> extract_partial_basic_state_PT_X(int pointx, int pointx_par, int pointz_par) {
    // Extract the partial basic state from MATLAB
    std::string partialbasicstate = "partial_basic_state_X_1000_8000.txt";
    std::vector<std::vector<double>> arr(pointx_par);
    std::vector<std::vector<double>> arr_final(pointx_par);

    for (int i = 0; i < pointx_par; ++i) {
        std::vector<double> col;
        arr[i] = col;
        arr_final[i] = col;
    }

    std::ifstream file_to_read(partialbasicstate);

    if (file_to_read.is_open()) {
        std::string line;
        while (std::getline(file_to_read, line)) {
            std::vector<double> arr_inter;
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                arr_inter.push_back(val);
            }

            for (int k = 0; k < pointx_par; ++k) {
                arr[k].push_back(arr_inter[k]);
            }
        }

        for (int p = 0; p < pointx_par; ++p) {
            arr_final[p] = std::vector<double>(arr[p].begin(), arr[p].end());
        }
    }
    else {
        std::cerr << "Unable to open file: " << partialbasicstate << std::endl;
    }

    std::vector<std::vector<double>>::size_type num = arr_final[0].size();
    std::vector<std::vector<double>> output(pointx_par, std::vector<double>(num, 0.0));

    for (std::vector<std::vector<double>>::size_type i = 0; i < num; ++i) {
        for (std::vector<std::vector<double>>::size_type q = 0; q < pointx_par; ++q) {
            output[q][i] = arr_final[q][i];
        }
    }

    return output;
}

double interp_verti_coeff_partial_PT_Z(std::vector<std::vector<double>> partial_basic_state, double domainx_cal, double domainz_cal, double height, double x_position, double dz_par, int pointz_par, int k3, double dx_par, int pointx_par, int m3) {
    std::vector<std::vector<double>> PoTem_partial_inter(pointx_par, std::vector<double>(pointz_par, 0.0));
    double wz_partial_pote_temp = 0.0;
    double wx_partial_pote_temp = 0.0;
    double PoTem_partial = 0.0;

    for (int k = 0; k < pointz_par; ++k) {
        if (height >= k * dz_par) {
            k3 += 1;
        }
    }

    int k2 = k3 - 1;

    if (k3 == pointz_par) {
        wz_partial_pote_temp = 1.0;
    } else if (k3 == 1) {
        wz_partial_pote_temp = 0.0;
    } else {
        for(int i=0;i<pointx_par;++i){
            wz_partial_pote_temp = (partial_basic_state[(i-1+pointx_par)%pointx_par][k3] - partial_basic_state[(i-1+pointx_par)%pointx_par][k2]) / (k3 * dz_par - k2 * dz_par);
        }
    }

    for (int i = 0; i < pointx_par; ++i) {
        if (height == 0.0) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][0] = partial_basic_state[(i-1+pointx_par)%pointx_par][0];
        } else if (height == domainz_cal) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][pointz_par - 1] = partial_basic_state[(i-1+pointx_par)%pointx_par][pointz_par - 1];
        } else {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][0] = partial_basic_state[(i-1+pointx_par)%pointx_par][k2] + wz_partial_pote_temp * (height - k2 * dz_par);
        }
    }

    for (int m = 0; m < pointx_par; ++m) {
        if (x_position >= m * dx_par) {
            m3 += 1;
        }
    }

    int m2 = m3 - 1;

    if (m3 == pointx_par) {
        PoTem_partial = PoTem_partial_inter[pointx_par - 1][0];
    } else if (m3 == 1) {
        PoTem_partial = PoTem_partial_inter[0][0];
    } else {
        wx_partial_pote_temp = (PoTem_partial_inter[m3][0] - PoTem_partial_inter[m2][0]) / (m3 * dx_par - m2 * dx_par);
        PoTem_partial = PoTem_partial_inter[m2][0] + wx_partial_pote_temp * (x_position - m2 * dx_par);
    }

    return PoTem_partial;
}

double interp_verti_coeff_partial_PT_X(std::vector<std::vector<double>> partial_basic_state, double domainx_cal, double domainz_cal, double height, double x_position, double dz_par, int pointz_par, int k3, double dx_par, int pointx_par, int m3) {
    std::vector<std::vector<double>> PoTem_partial_inter(pointx_par, std::vector<double>(pointz_par, 0.0));
    double wz_partial_pote_temp = 0.0;
    double wx_partial_pote_temp = 0.0;
    double PoTem_partial = 0.0;

    for (int k = 0; k < pointz_par; ++k) {
        if (height >= k * dz_par) {
            k3 += 1;
        }
    }

    int k2 = k3 - 1;

    if (k3 == pointz_par) {
        wz_partial_pote_temp = 1.0;
    } else if (k3 == 1) {
        wz_partial_pote_temp = 0.0;
    } else {
        for(int i=0;i<pointx_par;++i){
            wz_partial_pote_temp = (partial_basic_state[(i-1+pointx_par)%pointx_par][k3] - partial_basic_state[(i-1+pointx_par)%pointx_par][k2]) / (k3 * dz_par - k2 * dz_par);
        }
    }

    for (int i = 0; i < pointx_par; ++i) {
        if (height == 0.0) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][0] = partial_basic_state[(i-1+pointx_par)%pointx_par][0];
        } else if (height == domainz_cal) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][pointz_par - 1] = partial_basic_state[(i-1+pointx_par)%pointx_par][pointz_par - 1];
        } else {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par][0] = partial_basic_state[(i-1+pointx_par)%pointx_par][k2] + wz_partial_pote_temp * (height - k2 * dz_par);
        }
    }

    for (int m = 0; m < pointx_par; ++m) {
        if (x_position >= m * dx_par) {
            m3 += 1;
        }
    }

    int m2 = m3 - 1;

    if (m3 == pointx_par) {
        PoTem_partial = PoTem_partial_inter[pointx_par - 1][0];
    } else if (m3 == 1) {
        PoTem_partial = PoTem_partial_inter[0][0];
    } else {
        wx_partial_pote_temp = (PoTem_partial_inter[m3][0] - PoTem_partial_inter[m2][0]) / (m3 * dx_par - m2 * dx_par);
        PoTem_partial = PoTem_partial_inter[m2][0] + wx_partial_pote_temp * (x_position - m2 * dx_par);
    }

    return PoTem_partial;
}

double temperature_perturbation_source(double height, double r_cal, double r_cal_left, double r_cal_right, double xlocation) {
    int kind = 1;
    double delta_theta, rh, zc, rz, R_theta, TemPer_Klemp;

    if (kind == 1) {
        // Parameters
        delta_theta = 3.0;
        rh = 10000.0;
        zc = 1500.0;
        rz = 1500.0;
        R_theta = std::sqrt(std::pow(r_cal / rh, 2.0) + std::pow((height - zc) / rz, 2.0));

        // PT Perturbation from Klemp 2015
        if (R_theta < 1.0) {
            TemPer_Klemp = delta_theta * std::pow(std::cos(M_PI / 2.0 * R_theta), 2.0);
        } else {
            TemPer_Klemp = 0.0;
        }
    } else if (kind == 2) {
        // Parameters
        delta_theta = 3.0;
        rh = 5000.0;
        zc = 1500.0;
        rz = 1500.0;

        if (xlocation <= 40000.0) {
            R_theta = std::sqrt(std::pow(r_cal_left / rh, 2.0) + std::pow((height - zc) / rz, 2.0));
        } else {
            R_theta = std::sqrt(std::pow(r_cal_right / rh, 2.0) + std::pow((height - zc) / rz, 2.0));
        }

        // PT Perturbation from Klemp 2015
        if (R_theta < 1.0) {
            TemPer_Klemp = delta_theta * std::pow(std::cos(M_PI / 2.0 * R_theta), 2.0);
        } else {
            TemPer_Klemp = 0.0;
        }
    }

    return TemPer_Klemp;
}

std::vector<std::vector<double>> extract_initial_condition(int pointx) {
    // Extract the variables from a file (initial_condition_deltaz1000.txt)
    std::string initial_condition = "initial_condition_deltaz1000.txt";
    std::ifstream file_to_read(initial_condition);

    std::vector<double> Pres_init_temp1;
    std::vector<double> WaVa_init_temp1;
    std::vector<double> Dens_init_temp1;
    std::vector<double> PoTem_init_temp1;
    std::vector<double> Pai_init_temp1;
    std::vector<double> Temp_init_temp1;

    while (!file_to_read.eof()) {
        double Pres_trans, WaVa_trans, Dens_trans, PoTem_trans, Pai_trans, Temp_trans;
        file_to_read >> Pres_trans >> WaVa_trans >> Dens_trans >> PoTem_trans >> Pai_trans >> Temp_trans;

        Pres_init_temp1.push_back(Pres_trans);
        WaVa_init_temp1.push_back(WaVa_trans);
        Dens_init_temp1.push_back(Dens_trans);
        PoTem_init_temp1.push_back(PoTem_trans);
        Pai_init_temp1.push_back(Pai_trans);
        Temp_init_temp1.push_back(Temp_trans);
    }
    file_to_read.close();

    std::vector<double> Pres_init_temp2(Pres_init_temp1.begin(), Pres_init_temp1.end());
    std::vector<double> WaVa_init_temp2(WaVa_init_temp1.begin(), WaVa_init_temp1.end());
    std::vector<double> Dens_init_temp2(Dens_init_temp1.begin(), Dens_init_temp1.end());
    std::vector<double> PoTem_init_temp2(PoTem_init_temp1.begin(), PoTem_init_temp1.end());
    std::vector<double> Pai_init_temp2(Pai_init_temp1.begin(), Pai_init_temp1.end());
    std::vector<double> Temp_init_temp2(Temp_init_temp1.begin(), Temp_init_temp1.end());

    // Perform the necessary calculations
    for (int i = 0; i < Pres_init_temp2.size(); ++i) {
        Pres_init_temp2[i] /= 100.0;
    }

    int num = Pres_init_temp2.size();

    std::vector<std::vector<double>> output(6, std::vector<double>(num, 0.0));

    for (int i = 0; i < num; ++i) {
        output[0][i] = Pres_init_temp2[i];
        output[1][i] = WaVa_init_temp2[i];
        output[2][i] = Dens_init_temp2[i];
        output[3][i] = PoTem_init_temp2[i];
        output[4][i] = Pai_init_temp2[i];
        output[5][i] = Temp_init_temp2[i];
    }

    return output;
}

std::vector<std::vector<double>> extract_partial_basic_state(int pointx) {
    // Extract the partial basic state from a file
    std::string partialBasicState = "partial_basic_state.txt";
    std::ifstream fileToRead(partialBasicState);
    if (!fileToRead.is_open()) {
        std::cerr << "Failed to open file: " << partialBasicState << std::endl;
        exit(1);
    }

    std::vector<double> WaVaPartialTemp1;
    std::vector<double> PoTemPartialTemp1;
    std::vector<double> TempPartialTemp1;
    double pwvTrans, pptTrans, ptTrans;

    while (fileToRead >> pwvTrans >> pptTrans >> ptTrans) {
        WaVaPartialTemp1.push_back(pwvTrans);
        PoTemPartialTemp1.push_back(pptTrans);
        TempPartialTemp1.push_back(ptTrans);
    }

    int num = WaVaPartialTemp1.size();
    std::vector<std::vector<double>> output(3, std::vector<double>(num, 0.0));

    for (int i = 0; i < num; i++) {
        output[0][i] = PoTemPartialTemp1[i];
        output[1][i] = WaVaPartialTemp1[i];
        output[2][i] = TempPartialTemp1[i];
    }

    return output;
}

std::vector<double> interp_vert_coeff_initial(std::vector<std::vector<double>> initial_condition, double domainz_cal, double pointz, double height, int k1) {
    std::vector<double> output(6, 0.0);
    double dz_deltaz = 1000.0;
    int pointz_deltaz = static_cast<int>(domainz_cal / dz_deltaz) + 1;

    double wz_pres, wz_wava, wz_dens, wz_pote, wz_pai, wz_tem; // Declare the variables here

    for (int l = 0; l < pointz_deltaz; l++) {
        if (height >= l * dz_deltaz) {
            k1 += 1;
        }
        int k0 = k1 - 1;

        if (k1 == pointz_deltaz) {
            output[0] = 1.0;
            output[1] = 1.0;
            output[2] = 1.0;
            output[3] = 1.0;
            output[4] = 1.0;
            output[5] = 1.0;
        } else if (k1 == 1) {
            output[0] = 0.0;
            output[1] = 0.0;
            output[2] = 0.0;
            output[3] = 0.0;
            output[4] = 0.0;
            output[5] = 0.0;
        } else {
            double deltaz = k1 * dz_deltaz - k0 * dz_deltaz;
            wz_pres = (initial_condition[0][k1] - initial_condition[0][k0]) / deltaz;
            wz_wava = (initial_condition[1][k1] - initial_condition[1][k0]) / deltaz;
            wz_dens = (initial_condition[2][k1] - initial_condition[2][k0]) / deltaz;
            wz_pote = (initial_condition[3][k1] - initial_condition[3][k0]) / deltaz;
            wz_pai = (initial_condition[4][k1] - initial_condition[4][k0]) / deltaz;
            wz_tem = (initial_condition[5][k1] - initial_condition[5][k0]) / deltaz;
        }

        if (height == 0.0) {
            output[0] = initial_condition[0][0];
            output[1] = initial_condition[1][0];
            output[2] = initial_condition[2][0];
            output[3] = initial_condition[3][0];
            output[4] = initial_condition[4][0];
            output[5] = initial_condition[5][0];
        } else if (height == domainz_cal) {
            output[0] = initial_condition[0][pointz_deltaz - 1];
            output[1] = initial_condition[1][pointz_deltaz - 1];
            output[2] = initial_condition[2][pointz_deltaz - 1];
            output[3] = initial_condition[3][pointz_deltaz - 1];
            output[4] = initial_condition[4][pointz_deltaz - 1];
            output[5] = initial_condition[5][pointz_deltaz - 1];
        } else {
            double deltaz = height - k0 * dz_deltaz;
            output[0] = initial_condition[0][k0] + wz_pres * deltaz;
            output[1] = initial_condition[1][k0] + wz_wava * deltaz;
            output[2] = initial_condition[2][k0] + wz_dens * deltaz;
            output[3] = initial_condition[3][k0] + wz_pote * deltaz;
            output[4] = initial_condition[4][k0] + wz_pai * deltaz;
            output[5] = initial_condition[5][k0] + wz_tem * deltaz;
        }
    }
    return output;
}

std::vector<double> interp_verti_coeff_partial(std::vector<std::vector<double>> partial_basic_state, double domainz_cal, double height, double dz_par, int pointz_par, int k3) {
    for (int l = 0; l < pointz_par; ++l) {
        if (height >= l * dz_par) {
            k3 += 1;
        }
    }
    int k2 = k3 - 1;

    double wz_partial_pote_temp, wz_partial_wate_vapo;

    if (k3 == pointz_par) {
        wz_partial_pote_temp = 1;
        wz_partial_wate_vapo = 1;
    }
    else if (k3 == 1) {
        wz_partial_pote_temp = 0;
        wz_partial_wate_vapo = 0;
    }
    else {
        wz_partial_pote_temp = (partial_basic_state[0][k3] - partial_basic_state[0][k2]) / (k3 * dz_par - k2 * dz_par);
        wz_partial_wate_vapo = (partial_basic_state[1][k3] - partial_basic_state[1][k2]) / (k3 * dz_par - k2 * dz_par);
    }

    double PoTem_partial, WaVa_partial;

    if (height == 0.0) {
        PoTem_partial = partial_basic_state[0][0];
        WaVa_partial = partial_basic_state[1][0];
    }
    else if (height == domainz_cal) {
        PoTem_partial = partial_basic_state[0][pointz_par - 1];
        WaVa_partial = partial_basic_state[1][pointz_par - 1];
    }
    else {
        PoTem_partial = partial_basic_state[0][k2] + wz_partial_pote_temp * (height - k2 * dz_par);
        WaVa_partial = partial_basic_state[1][k2] + wz_partial_wate_vapo * (height - k2 * dz_par);
    }

    std::vector<double> output = {PoTem_partial, WaVa_partial};

    return output;
}



std::vector<std::vector<double>> extract_partial_basic_state_P(int pointx, int pointx_par, int pointz_par) {
    std::string partialbasicstate = "partial_basic_state_P_1000_8000.txt";
    std::vector<std::vector<double>> arr(pointx_par);
    std::vector<std::vector<double>> arr_final(pointx_par);
    
    std::ifstream file_to_read(partialbasicstate);
    // if (!file_to_read.is_open()) {
    //     std::cerr << "无法打开文件。" << std::endl;
    //     return {};
    // }

    std::string line;
    while (std::getline(file_to_read, line)) {
        std::istringstream iss(line);
        std::vector<double> arr_inter;
        double value;
        while (iss >> value) {
            arr_inter.push_back(value);
        }
        for (int k = 0; k < pointx_par; ++k) {
            arr[k].push_back(arr_inter[k]);
        }
    }
    file_to_read.close();

    for (int p = 0; p < pointx_par; ++p) {
        arr_final[p] = arr[p];
    }
    
    int num = arr_final[0].size();
    std::vector<std::vector<double>> output(pointx_par, std::vector<double>(num, 0.0));

    for (int i = 0; i < num; ++i) {
        for (int q = 0; q < pointx_par; ++q) {
            output[q][i] = arr_final[q][i];
        }
    }
    return output;
}

double interp_verti_coeff_partial_P
(   const std::vector<std::vector<double>>& partial_basic_state, 
    double domainx_cal, 
    double domainz_cal, 
    double height, 
    double x_position, 
    double dz_par, 
    int pointz_par, 
    double k3, 
    double dx_par, 
    int pointx_par, 
    double m3) 
{
    std::vector<double> PoTem_partial_inter(pointx_par, 0.0);
    int k2 = 0, m2 = 0;
    double wz_partial_pote_temp = 0.0;

    // Calculate k3
    for (int k = 0; k < pointz_par; ++k) {
        if (height >= k * dz_par) {
            k3++;
        }
        k2 = k3 - 1;
    }

    // Calculate wz_partial_pote_temp
    if (k3 == pointz_par) {
        wz_partial_pote_temp = 1;
    } else if (k3 == 1) {
        wz_partial_pote_temp = 0;
    } else {
        for (int i = 0; i < pointx_par; ++i) {
            wz_partial_pote_temp = (partial_basic_state[(i-1+pointx_par)%pointx_par][k3] - partial_basic_state[(i-1+pointx_par)%pointx_par][k2]) / (k3 * dz_par - k2 * dz_par);
        }
    }

    // Calculate PoTem_partial_inter
    for (int i = 0; i < pointx_par; ++i) {
        if (height == 0.0) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par] = partial_basic_state[(i-1+pointx_par)%pointx_par][0];
        } else if (height == domainz_cal) {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par] = partial_basic_state[(i-1+pointx_par)%pointx_par][pointz_par - 1];
        } else {
            PoTem_partial_inter[(i-1+pointx_par)%pointx_par] = partial_basic_state[(i-1+pointx_par)%pointx_par][k2] + wz_partial_pote_temp * (height - k2 * dz_par);
        }
    }

    // Calculate m3 and m2
    for (int m = 0; m < pointx_par; ++m) {
        if (x_position >= m * dx_par) {
            m3++;
        }
        m2 = m3 - 1;
    }

    // Calculate PoTem_partial
    double PoTem_partial = 0.0;
    if (m3 == pointx_par) {
        PoTem_partial = PoTem_partial_inter[pointx_par - 1];
    } else if (m3 == 1) {
        PoTem_partial = PoTem_partial_inter[0];
    } else {
        double wx_partial_pote_temp = (PoTem_partial_inter[m3] - PoTem_partial_inter[m2]) / (m3 * dx_par - m2 * dx_par);
        PoTem_partial = PoTem_partial_inter[m2] + wx_partial_pote_temp * (x_position - m2 * dx_par);
    }

    // Output
    return PoTem_partial;
}

void cpp_code_(int* cnt,double*Pres_prog,double*PoTem_prog,double*Dens_prog,double*ClWa_prog,double*RaWa_prog,double*WaVa_prog,double*Velo_prog,double*Coor_prog,double*ans){
    double domainx_cal = 1350000.0;
    double domainy_cal = 1350000.0;
    double domainz_cal = 20000.0;
    int pointx = 21;
    int pointy = 21;
    int pointz = 21;
    double dx = domainx_cal / (pointx - 1);
    double dy = domainy_cal / (pointy - 1);
    double dz = domainz_cal / (pointz - 1);
    std::vector<double> Pres_init_temp1;
    std::vector<double> WaVa_init_temp1;
    std::vector<double> Dens_init_temp1;
    std::vector<double> PoTem_init_temp1;
    std::vector<double> Pai_init_temp1;
    std::vector<double> Temp_init_temp1;
    std::string initial_condition = "initial_condition_deltaz1000.txt";
    std::ifstream infile(initial_condition);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double Pres_trans, WaVa_trans, Dens_trans, PoTem_trans, Pai_trans, Temp_trans;
        if (!(iss >> Pres_trans >> WaVa_trans >> Dens_trans >> PoTem_trans >> Pai_trans >> Temp_trans)) {
            std::cerr << "Error reading line: " << line << std::endl;
            continue; // skip the problematic line
        }
        Pres_init_temp1.push_back(Pres_trans);
        WaVa_init_temp1.push_back(WaVa_trans);
        Dens_init_temp1.push_back(Dens_trans);
        PoTem_init_temp1.push_back(PoTem_trans);
        Pai_init_temp1.push_back(Pai_trans);
        Temp_init_temp1.push_back(Temp_trans);
    }
    // Close the file
    infile.close();
    std::vector<double> Pres_init_temp0 = Pres_init_temp1;
    std::vector<double> WaVa_init_temp2 = WaVa_init_temp1;
    std::vector<double> Dens_init_temp2 = Dens_init_temp1;
    std::vector<double> PoTem_init_temp2 = PoTem_init_temp1;
    std::vector<double> Pai_init_temp2 = Pai_init_temp1;
    std::vector<double> Temp_init_temp2 = Temp_init_temp1;
    std::vector<double> Pres_init_temp2;
    for (auto pres_0 : Pres_init_temp0) {
        Pres_init_temp2.push_back(pres_0/100.0);
    }
        double deltaDensity = 1.0;
	double dz_par = 1000.0;
	int pointz_par = 21;
	double dx_par = 8000.0;
	int pointx_par = 11;
        //double count=0;
	std::vector<std::vector<double>>tmp=extract_partial_basic_state_P(pointx, pointx_par, pointz_par);
	for (int n = 0; n < *cnt;n++) 
        {
		//time_t start_time, end_time;
                //time(&start_time);
		int t1=n*3,t2=n*3+1,t3=n*3+2;
		double xhorizon = Coor_prog[t1];
		double yhorizon = Coor_prog[t2];
		double height = Coor_prog[t3];
		double r_cal = std::sqrt(((xhorizon - (domainx_cal / 2.0) * (domainx_cal / 2.0)) + (yhorizon - (domainy_cal / 2.0) * (domainy_cal / 2.0))));
		double k1 = 0;
		double k3 = 0;
		double m3 = 0;
		//# Interpolation
                int k0;
		for (int m = 0;m < pointz;m++) 
                {
			if (height >= m * dz) 
                        {
				k1 += 1;
			}
			k0 = k1 - 1;
		}
                double wz_pres ;
		double wz_dens ;
		double wz_wava ;
		double wz_pote ;
		if (k1 == pointz) 
                {
			wz_pres = 1;
			wz_dens = 1;
			wz_wava = 1;
		    wz_pote = 1;
		}
		else if (k1 == 1) 
                {
			wz_pres = 0;
			wz_dens = 0;
			wz_wava = 0;
			wz_pote = 0;
		}
		else 
                {
			wz_pres = (Pres_init_temp2[k1] - Pres_init_temp2[k0]) / (k1 * dz - k0 * dz);
			wz_dens = (Dens_init_temp2[k1] - Dens_init_temp2[k0]) / (k1 * dz - k0 * dz);
			wz_wava = (WaVa_init_temp2[k1] - WaVa_init_temp2[k0]) / (k1 * dz - k0 * dz);
			wz_pote = (PoTem_init_temp2[k1] - PoTem_init_temp2[k0]) / (k1 * dz - k0 * dz);
		}
                double Pres_init ;
		double Dens_init ;
		double WaVa_init ;
		double PoTem_init;
		if (height == 0.0) 
                {
			Pres_init = Pres_init_temp2[0];
			Dens_init = Dens_init_temp2[0];
			WaVa_init = WaVa_init_temp2[0];
			PoTem_init = PoTem_init_temp2[0];
		}
                else if (height == 20000.0)
                {
			Pres_init = Pres_init_temp2[pointz - 1];
			Dens_init = Dens_init_temp2[pointz - 1];
			WaVa_init = WaVa_init_temp2[pointz - 1];
			PoTem_init = PoTem_init_temp2[pointz - 1];
		}
                else
                {
			Pres_init = Pres_init_temp2[k0] + wz_pres * (height - k0 * dz);
			Dens_init = Dens_init_temp2[k0] + wz_dens * (height - k0 * dz);
			WaVa_init = WaVa_init_temp2[k0] + wz_pres * (height - k0 * dz);
			PoTem_init = PoTem_init_temp2[k0] + wz_dens * (height - k0 * dz);
		}
                double g = 9.8e-00;
	        double PoTem_Pert=PoTem_prog[n]/PoTem_init;
		double delta_g = 0.608 * WaVa_prog[n] - ClWa_prog[n] - RaWa_prog[n];
                double finalvalue11 = g * PoTem_Pert * 1.0e-00;
                double finalvalue12 = finalvalue11 + g * delta_g * 1.0e-00;
	    //time(&end_time);
            //double elapsed_time = difftime(end_time, start_time);
            //count=count+elapsed_time;
	    //大约一秒
                //std::vector<std::vector<double>>tmp=extract_partial_basic_state_P(pointx, pointx_par, pointz_par);
		double dpdx = -1.0e00 * interp_verti_coeff_partial_P(tmp
                , domainx_cal, domainz_cal, height, xhorizon, dz_par, pointz_par, k3, dx_par, pointx_par, m3) ;
            //time(&end_time);
            //double elapsed_time = difftime(end_time, start_time);
            //count=count+elapsed_time; 
	double dpdx1 = 0.0;
        if(xhorizon > 40000.0)
        {
            dpdx1 = 1.0e01 / 60.0;
        }
        else if(xhorizon < 40000.0)
        {
            dpdx1 = -1.0e01 / 60.0;
        }
        else
        {
            dpdx1 = 0.0;
        }

        double f = 0.0;
        if (xhorizon > 40000.0) 
        {
            f = 0.0;
        }
        else if (xhorizon < 40000.0) 
        {
            f = -0.0;
        }
        else
        {
            f = 0.0;
        }
        double ux = Velo_prog[n*3+0];
        double uy = Velo_prog[n*3+1];
        double w = Velo_prog[n*3+2];
        double times = 1.0;// # (good result : times = 100.0, uy = const15.0absolutevalue)
        double artificialwatertimes = 1.0;// # (good result : times = 10.0)
        double value_x = 0.0;
        if (WaVa_prog[n] >= 0.001)   
        {
            value_x = dpdx + times * f * uy * artificialwatertimes;
        }
        else
        {
            value_x = dpdx + times * f * uy;
        }
        double value_y = - f * ux;
        double coeffx = 0.0;
        double coeffy = 0.0;
        //#    if abs.(xhorizon - domainx_cal/2.0)>=25000.0 or abs.(yhorizon - domainy_cal/2.0)>=25000.0:
        if (fabs(xhorizon - 40000.0) >= 35000.0 || fabs(yhorizon - 40000.0) >= 35000.0) 
        {
            coeffx = 0.0;
            coeffy = 0.0;
        }
        else if ((fabs(xhorizon - 40000.0) >= 25000.0 && fabs(xhorizon - 40000.0) < 35000.0) || (fabs(yhorizon - 40000.0) >= 25000.0 && fabs(yhorizon - 40000.0) < 35000.0)) 
        {
            coeffx = 1.0 - (fabs(xhorizon - 40000.0) - 25000.0) / 10000.0;
            coeffy = 1.0 - (fabs(yhorizon - 40000.0) - 25000.0) / 10000.0;
        }
        else if(fabs(xhorizon - 40000.0) < 25000.0 || fabs(yhorizon - 40000.0) < 25000.0)
        {
            coeffx = 1.0;
            coeffy = 1.0;
        }
        ans[n*3+0]= value_x * coeffx;
        ans[n*3+1]= value_y * coeffy;
        ans[n*3+2]= finalvalue12;
    }
    //time(&end_time);
    //double elapsed_time = difftime(end_time, start_time);
    //printf("经过了 %.2lf 秒\n", count);
}
void cpp_code_water_(int* cnt,double*Pres_prog,double*PoTem_prog,double*Dens_prog,double*RaWa_prog,double*WaVa_prog,double*Velo_prog,double*Coor_prog,double*ans){
    double domainx_cal = 1350000.0;
    double domainy_cal = 1350000.0;
    double domainz_cal = 20000.0;
    int pointx = 21;
    int pointy = 21;
    int pointz = 21;
    double dx = domainx_cal / (pointx - 1);
    double dy = domainy_cal / (pointy - 1);
    double dz = domainz_cal / (pointz - 1);
    // Definition of new variables of cumulus convection
    double alpha = 1.0e-03;
    double beta = 0.5e-06;
    double L = 2.5e10;
    double cp = 1.004e07;
    double PR = 1000.0;
    double R = 287.0e04;
    double deltaDensity = 1.0;
    double dz_par = 100.0;
    int pointz_par = 201;
    std::vector<std::vector<double>> initial_condition = extract_initial_condition(pointx);
    std::vector<std::vector<double>> partial_basic_state = extract_partial_basic_state(pointx);
    
    for (int n = 0; n < *cnt; ++n) {
        int t1=n*3,t2=n*3+1,t3=n*3+2;
		double xhorizon = Coor_prog[t1];
		double yhorizon = Coor_prog[t2];
		double height = Coor_prog[t3];
        double r_cal = std::sqrt((std::pow(xhorizon - (domainx_cal / 2.0), 2.0) + std::pow(yhorizon - (domainy_cal / 2.0), 2.0)));
        int k1 = 0;
        int k3 = 0;
    
        // Interpolation
        std::vector<double> initial_vert_coeff = interp_vert_coeff_initial(initial_condition, domainz_cal, pointz, height, k1);
        std::vector<double> partial_vert_coeff = interp_verti_coeff_partial(partial_basic_state, domainz_cal, height, dz_par, pointz_par, k3);
    
        // Vertical Basic State
        double w = Velo_prog[n*3+2];
    
        // Final Value
        int choice_restrict = 2;
        double restrict_velocity = 0.1;
        double finalvalue2;
    
        if (choice_restrict == 1) {
            if (w <= restrict_velocity && w >= -restrict_velocity) {
                finalvalue2 = -w * partial_vert_coeff[1];
            } else {
                if (w >= 0.0) {
                    finalvalue2 = -restrict_velocity * partial_vert_coeff[1];
                } else {
                    finalvalue2 = restrict_velocity * partial_vert_coeff[1];
                }
            }
        } else if (choice_restrict == 2) {
            finalvalue2 = -w * partial_vert_coeff[1];
        }
        ans[n]=finalvalue2;
    }
}
void cpp_code_temp_(int* cnt,double*Pres_prog,double*PoTem_prog,double*Dens_prog,double*RaWa_prog,double*WaVa_prog,double*Velo_prog,double*Coor_prog,double*ans){
    double domainx_cal = 80000.0;
    double domainy_cal = 80000.0;
    double domainz_cal = 20000.0;
    int pointx = 21;
    int pointy = 21;
    int pointz = 21;
    double dx = domainx_cal / (pointx - 1);
    double dy = domainy_cal / (pointy - 1);
    double dz = domainz_cal / (pointz - 1);

    double deltaDensity = 1.0;
    double dz_par_supercell = 1000.0;
    int pointz_par_supercell = 21;
    double dx_par_supercell = 8000.0;
    int pointx_par_supercell = 11;
    double dz_par_typhoon = 100.0;
    int pointz_par_typhoon = 201;
    std::vector<std::vector<double>> partial_basic_state = extract_partial_basic_state(pointx);
    std::vector<std::vector<double>> partial_basic_state_PT_Z = extract_partial_basic_state_PT_Z(pointx, pointx_par_supercell, pointz_par_supercell);
    std::vector<std::vector<double>> partial_basic_state_PT_X = extract_partial_basic_state_PT_X(pointx, pointx_par_supercell, pointz_par_supercell);
    for (int n = 0; n < *cnt; ++n) {
        int t1=n*3,t2=n*3+1,t3=n*3+2;
	double xhorizon = Coor_prog[t1];
	double yhorizon = Coor_prog[t2];
	double height = Coor_prog[t3];
        double r_cal = std::sqrt(std::pow(xhorizon - (domainx_cal / 2.0), 2.0) + std::pow(yhorizon - (domainy_cal / 2.0), 2.0));
        double r_cal_right = std::sqrt(std::pow(xhorizon - (domainx_cal / 2.0 + 5000.0), 2.0) + std::pow(yhorizon - (domainy_cal / 2.0), 2.0));
        double r_cal_left = std::sqrt(std::pow(xhorizon - (domainx_cal / 2.0 - 5000.0), 2.0) + std::pow(yhorizon - (domainy_cal / 2.0), 2.0));
	int k3 = 0;
        int m3 = 0;

        std::vector<double> partial_vert_coeff = interp_verti_coeff_partial(partial_basic_state, domainz_cal, height, dz_par_typhoon, pointz_par_typhoon, k3);
        double partial_vert_coeff_PT_Z = interp_verti_coeff_partial_PT_Z(partial_basic_state_PT_Z, domainx_cal, domainz_cal, height, xhorizon, dz_par_supercell, pointz_par_supercell, k3, dx_par_supercell, pointx_par_supercell, m3);
        double partial_vert_coeff_PT_X = interp_verti_coeff_partial_PT_X(partial_basic_state_PT_X, domainx_cal, domainz_cal, height, xhorizon, dz_par_supercell, pointz_par_supercell, k3, dx_par_supercell, pointx_par_supercell, m3);
        double TemPer = temperature_perturbation_source(height, r_cal, r_cal_left, r_cal_right, xhorizon);

        // Get velocity components
        double w = Velo_prog[n*3+2];
        double ux = Velo_prog[n*3+0];
        double uy = Velo_prog[n*3+1];
        
        
        // Calculate the final value
        double value_typhoon = -w * partial_vert_coeff[0];
        double value_supercell = -w * partial_vert_coeff_PT_Z - ux * partial_vert_coeff_PT_X;
        double value = value_typhoon + value_supercell;

        ans[n]=value;
    }
}




}


