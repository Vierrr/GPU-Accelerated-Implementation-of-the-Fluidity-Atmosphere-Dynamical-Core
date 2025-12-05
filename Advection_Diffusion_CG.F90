!     Copyright (C) 2006 Imperial College London and others.
!     
!     Please see the AUTHORS file in the main source directory for a full list
!     of copyright holders.
!     
!     Prof. C Pain
!     Applied Modelling and Computation Group
!     Department of Earth Science and Engineering
!     Imperial College London
!     
!     amcgsoftware@imperial.ac.uk
!     
!     This library is free software; you can redistribute it and/or
!     modify it under the terms of the GNU Lesser General Public
!     License as published by the Free Software Foundation,
!     version 2.1 of the License.
!     
!     This library is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!     Lesser General Public License for more details.
!     
!     You should have received a copy of the GNU Lesser General Public
!     License along with this library; if not, write to the Free Software
!     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!     USA

#include "fdebug.h"

module advection_diffusion_cg
  
  use fldebug
  use global_parameters, only : FIELD_NAME_LEN, OPTION_PATH_LEN, COLOURING_CG1
  use futils, only: int2str
  use quadrature
  use elements
  use spud
  use integer_set_module
#ifdef _OPENMP
  use omp_lib
#endif
  use sparse_tools
  use transform_elements
  use fetools
  use fields
  use profiler
  use sparse_tools_petsc
  use state_module
  use boundary_conditions
  use field_options
  use sparsity_patterns_meshes
  use boundary_conditions_from_options
  use petsc_solve_state_module
  use upwind_stabilisation
  use multiphase_module
  use colouring
  use iso_c_binding  
  implicit none
  include "mpif.h" 
  private
  
  public :: solve_field_equation_cg, advection_diffusion_cg_check_options, &
  !2022-04-02-001S================= Max Min and Effective Mesh Sizes Computing !@#$% jxli Apr 2022
            average_mesh_size
  !2022-04-02-001E================= Max Min and Effective Mesh Sizes Computing !@#$% jxli Apr 2022
  
  character(len = *), parameter, public :: advdif_cg_m_name = "AdvectionDiffusionCGMatrix"
  character(len = *), parameter, public :: advdif_cg_rhs_name = "AdvectionDiffusionCGRHS"
  character(len = *), parameter, public :: advdif_cg_delta_t_name = "AdvectionDiffusionCGChange"
  character(len = *), parameter, public :: advdif_cg_velocity_name = "AdvectionDiffusionCGVelocity"
  
  ! Stabilisation schemes
  integer, parameter :: STABILISATION_NONE = 0, &
    & STABILISATION_STREAMLINE_UPWIND = 1, STABILISATION_SUPG = 2
  
  ! Boundary condition types
  integer, parameter :: BC_TYPE_NEUMANN = 1, BC_TYPE_WEAKDIRICHLET = 2, BC_TYPE_INTERNAL = 3, &
                        BC_TYPE_ROBIN = 4
  
  ! Global variables, set by assemble_advection_diffusion_cg for use by
  ! assemble_advection_diffusion_element_cg and
  ! assemble_advection_diffusion_face_cg
  
  ! Local timestep
  real :: local_dt
  ! Implicitness/explicitness factor * timestep
  real :: dt_theta
  ! Implicitness/explicitness factor
  real :: theta
  ! Conservative/non-conservative discretisation factor
  real :: beta
  ! Stabilisation scheme
  integer :: stabilisation_scheme
  integer :: nu_bar_scheme
  real :: nu_bar_scale
  
  ! equation type
  integer :: equation_type
  ! Implicitness/explicitness factor for density
  real :: density_theta
  ! Which terms do we have?
  
  ! Mass term?
  logical :: have_mass
  ! Lump mass?
  logical :: lump_mass
  ! Advection?
  logical :: have_advection
  ! Integrate advection by parts?
  logical :: integrate_advection_by_parts
  ! Source?
  logical :: have_source
  integer :: have_source_int, nvfrac_associated,have_diffu_int
  ! Add source directly to the right hand side?
  logical :: add_src_directly_to_rhs
  ! Absorption?
  logical :: have_absorption
  ! Diffusivity?
  logical :: have_diffusivity
  ! Isotropic diffusivity?
  logical :: isotropic_diffusivity
  ! Is the mesh moving?
  logical :: move_mesh
  ! Is this material_phase compressible?
  logical :: compressible = .false.
  ! Are we running a multiphase flow simulation?
  logical :: multiphase

  INTERFACE
  SUBROUTINE adv_create_stream_c() &
     BIND(C, name="adv_create_stream")
  END SUBROUTINE adv_create_stream_c
  END INTERFACE
  INTERFACE
  SUBROUTINE adv_alloc_gpu_mem_c(ele_num,nodes) &
     BIND(C, name="adv_alloc_gpu_mem")
     IMPORT :: C_INT
     INTEGER(KIND=C_INT), VALUE   :: ele_num,nodes
  END SUBROUTINE adv_alloc_gpu_mem_c
  END INTERFACE


contains

  !subroutine solve_field_equation_cg(field_name, state, istate, dt, velocity_name, iterations_taken) ! FA Original Code
  subroutine solve_field_equation_cg(field_name, state, istate, dt, velocity_name, iterations_taken, counterG) !@#$% jxli Apr 2023
    !!< Construct and solve the advection-diffusion equation for the given
    !!< field using a continuous Galerkin discretisation. Based on
    !!< Advection_Diffusion_DG and Momentum_CG.
    
    character(len = *), intent(in) :: field_name
    type(state_type), dimension(:), intent(inout) :: state
    integer, intent(in) :: istate
    real, intent(in) :: dt
    character(len = *), optional, intent(in) :: velocity_name
    integer, intent(out), optional :: iterations_taken
    
    type(csr_matrix) :: matrix
    type(scalar_field) :: delta_t, rhs
    type(scalar_field), pointer :: t
    
    ! Do we open MRF_paramerisation? Yes, this is used for transfer countergradient. !@#$% jxli Apr 2023
    type(tensor_field), intent(out)     :: counterG

    ewrite(1, *) "In solve_field_equation_cg"
    
    ewrite(2, *) "Solving advection-diffusion equation for field " // &
      & trim(field_name) // " in state " // trim(state(istate)%name)

    call initialise_advection_diffusion_cg(field_name, t, delta_t, matrix, rhs, state(istate))
    
    call profiler_tic(t, "assembly")
    !S2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023
    !call assemble_advection_diffusion_cg(t, matrix, rhs, state(istate), dt, velocity_name = velocity_name)  ! FA Original Code   
    call assemble_advection_diffusion_cg(t, matrix, rhs, state(istate), dt, velocity_name = velocity_name, counterG=counterG) !@#$% jxli Apr 2023 
    !E2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023    

    ! Note: the assembly of the heat transfer term is done here to avoid 
    ! passing in the whole state array to assemble_advection_diffusion_cg.
    if(have_option("/multiphase_interaction/heat_transfer") .and. &
       equation_type_index(trim(t%option_path)) == FIELD_EQUATION_INTERNALENERGY) then
      call add_heat_transfer(state, istate, t, matrix, rhs)
    end if
    call profiler_toc(t, "assembly")
    
    call profiler_tic(t, "solve_total")
    call solve_advection_diffusion_cg(t, delta_t, matrix, rhs, state(istate), &
                                      iterations_taken = iterations_taken)
    call profiler_toc(t, "solve_total")

    call profiler_tic(t, "assembly")
    call apply_advection_diffusion_cg_change(t, delta_t, dt)
    
    call finalise_advection_diffusion_cg(delta_t, matrix, rhs)
    call profiler_toc(t, "assembly")

    ewrite(1, *) "Exiting solve_field_equation_cg"
    
  end subroutine solve_field_equation_cg
  
  subroutine initialise_advection_diffusion_cg(field_name, t, delta_t, matrix, rhs, state)
    character(len = *), intent(in) :: field_name
    type(scalar_field), pointer :: t
    type(scalar_field), intent(out) :: delta_t
    type(csr_matrix), intent(out) :: matrix
    type(scalar_field), intent(out) :: rhs
    type(state_type), intent(inout) :: state
    
    integer :: stat
    type(csr_sparsity), pointer :: sparsity
    type(scalar_field), pointer :: t_old
        
    t => extract_scalar_field(state, field_name)
    if(t%mesh%continuity /= 0) then
      FLExit("CG advection-diffusion requires a continuous mesh")
    end if
        
    t_old => extract_scalar_field(state, "Old" // field_name, stat = stat)
    if(stat == 0) then
      assert(t_old%mesh == t%mesh)
      ! Reset t to value at the beginning of the timestep
      call set(t, t_old)
    end if
    
    sparsity => get_csr_sparsity_firstorder(state, t%mesh, t%mesh)
    
    call allocate(matrix, sparsity, name = advdif_cg_m_name)
    call allocate(rhs, t%mesh, name = advdif_cg_rhs_name)
    call allocate(delta_t, t%mesh, name = trim(field_name)//advdif_cg_delta_t_name)
    
    call set_advection_diffusion_cg_initial_guess(delta_t)
    
  end subroutine initialise_advection_diffusion_cg
  
  subroutine finalise_advection_diffusion_cg(delta_t, matrix, rhs)
    type(scalar_field), intent(inout) :: delta_t
    type(csr_matrix), intent(inout) :: matrix
    type(scalar_field), intent(inout) :: rhs
    
    call deallocate(matrix)
    call deallocate(rhs)
    call deallocate(delta_t)
    
  end subroutine finalise_advection_diffusion_cg
  
  subroutine set_advection_diffusion_cg_initial_guess(delta_t)
    type(scalar_field), intent(inout) :: delta_t
    
    call zero(delta_t)
    
  end subroutine set_advection_diffusion_cg_initial_guess
  
  !subroutine assemble_advection_diffusion_cg(t, matrix, rhs, state, dt, velocity_name)  ! FA Original Code   
  subroutine assemble_advection_diffusion_cg(t, matrix, rhs, state, dt, velocity_name, counterG) !@#$% jxli Apr 2023 

    type(scalar_field), intent(inout) :: t
    type(csr_matrix), intent(inout) :: matrix
    type(scalar_field), intent(inout) :: rhs
    type(state_type), intent(inout) :: state
    real, intent(in) :: dt
    character(len = *), optional, intent(in) :: velocity_name

    character(len = FIELD_NAME_LEN) :: lvelocity_name, velocity_equation_type
    integer :: i, j, stat
    integer, dimension(:), allocatable :: t_bc_types
    type(scalar_field) :: t_bc, t_bc_2
    type(scalar_field), pointer :: absorption, sinking_velocity, source
    type(tensor_field), pointer :: diffusivity
    type(vector_field) :: velocity
    type(vector_field), pointer :: gravity_direction, temp_velocity_ptr, velocity_ptr, grid_velocity
    type(vector_field), pointer :: positions, old_positions, new_positions
    type(scalar_field), target :: dummydensity
    type(scalar_field), pointer :: density, olddensity
    character(len = FIELD_NAME_LEN) :: density_name
    type(scalar_field), pointer :: pressure
    
    ! Volume fraction fields for multiphase flow simulation
    type(scalar_field), pointer :: vfrac
    type(scalar_field) :: nvfrac ! Non-linear version

    !! Coloring  data structures for OpenMP parallization
    type(integer_set), dimension(:), pointer :: colours
    integer :: clr, nnid, len, ele
    integer :: num_threads, thread_num
    !! Did we successfully prepopulate the transform_to_physical_cache?
    logical :: cache_valid
    logical :: have_mrf
    type(element_type), dimension(:), allocatable :: supg_element
  
    ! Do we open MRF_paramerisation? Yes, this is used for transfer countergradient. !@#$% jxli Apr 2023
    type(tensor_field), intent(out)     :: counterG
    integer, dimension(:), pointer :: element_nodes
    INTEGER :: clock_rate
    real:: start_time, end_time
    real elapsed_time;
    ewrite(1, *) "In assemble_advection_diffusion_cg"
    
    assert(mesh_dim(rhs) == mesh_dim(t))
    assert(ele_count(rhs) == ele_count(t))
    
    if(present(velocity_name)) then
      lvelocity_name = velocity_name
    else
      lvelocity_name = "NonlinearVelocity"
    end if

#ifdef _OPENMP
    num_threads = omp_get_max_threads()
#else
    num_threads = 1
#endif
    ! Step 1: Pull fields out of state
    
    ! Coordinate
    positions => extract_vector_field(state, "Coordinate")
    ewrite_minmax(positions)
    assert(positions%dim == mesh_dim(t))
    assert(ele_count(positions) == ele_count(t))
    
    ! Velocity    
    velocity_ptr => extract_vector_field(state, lvelocity_name, stat = stat)
    if(stat == 0) then
      assert(velocity_ptr%dim == mesh_dim(t))
      assert(ele_count(velocity_ptr) == ele_count(t))
      
      ewrite(2, *) "Velocity:"
      ewrite_minmax(velocity_ptr)

      if (have_option(trim(t%option_path) // &
        "/prognostic/spatial_discretisation/continuous_galerkin/advection_terms/only_sinking_velocity")) then
        ewrite(2, *) "No advection set for field"
        call allocate(velocity, mesh_dim(t), t%mesh, name = advdif_cg_velocity_name)
        call zero(velocity)
      else
        call allocate(velocity, velocity_ptr%dim, velocity_ptr%mesh, name = advdif_cg_velocity_name)
        call set(velocity, velocity_ptr)
      end if
    else
      ewrite(2, *) "No velocity"
      call allocate(velocity, mesh_dim(t), t%mesh, name = advdif_cg_velocity_name)
      call zero(velocity)
    end if
        
    ! Source
    source => extract_scalar_field(state, trim(t%name) // "Source", stat = stat)
    have_source = stat == 0
    if(have_source) then
      assert(mesh_dim(source) == mesh_dim(t))
      assert(ele_count(source) == ele_count(t))
      
      add_src_directly_to_rhs = have_option(trim(source%option_path)//'/diagnostic/add_directly_to_rhs')
      
      if (add_src_directly_to_rhs) then 
         ewrite(2, *) "Adding Source field directly to the right hand side"
         assert(node_count(source) == node_count(t))
      end if
      
      ewrite_minmax(source)
    else
      ewrite(2, *) "No source"
      
      add_src_directly_to_rhs = .false.
    end if
    
    ! Absorption
    absorption => extract_scalar_field(state, trim(t%name) // "Absorption", stat = stat)
    have_absorption = stat == 0
    if(have_absorption) then
      assert(mesh_dim(absorption) == mesh_dim(t))
      assert(ele_count(absorption) == ele_count(t))
    
      ewrite_minmax(absorption)
    else
      ewrite(2, *) "No absorption"
    end if

    ! Sinking velocity
    sinking_velocity => extract_scalar_field(state, trim(t%name) // "SinkingVelocity", stat = stat)
    if(stat == 0) then
      ewrite_minmax(sinking_velocity)
      
      gravity_direction => extract_vector_field(state, "GravityDirection")
      ! this may perform a "remap" internally from CoordinateMesh to VelocitMesh
      call addto(velocity, gravity_direction, scale = sinking_velocity)
      ewrite_minmax(velocity)
    else
      ewrite(2, *) "No sinking velocity"
    end if
    
    ! Diffusivity
    diffusivity => extract_tensor_field(state, trim(t%name) // "Diffusivity", stat = stat)
    have_diffusivity = stat == 0
    if(have_diffusivity) then
      assert(all(diffusivity%dim == mesh_dim(t)))
      assert(ele_count(diffusivity) == ele_count(t))
      
      isotropic_diffusivity = option_count(complete_field_path(diffusivity%option_path)) &
        & == option_count(trim(complete_field_path(diffusivity%option_path)) // "/value/isotropic")
        
      if(isotropic_diffusivity) then
        ewrite(2, *) "Isotropic diffusivity"
        assert(all(diffusivity%dim > 0))
        ewrite_minmax(diffusivity%val(1, 1, :))
      else
        ewrite_minmax(diffusivity)
      end if
    else
      isotropic_diffusivity = .false.
      ewrite(2, *) "No diffusivity"
    end if

    ! Step 2: Pull options out of the options tree
    
    call get_option(trim(t%option_path) // "/prognostic/temporal_discretisation/theta", theta)
    assert(theta >= 0.0 .and. theta <= 1.0)
    ewrite(2, *) "Theta = ", theta
    dt_theta = dt * theta
    local_dt = dt
    
    call get_option(trim(t%option_path) // "/prognostic/spatial_discretisation/conservative_advection", beta)
    assert(beta >= 0.0 .and. beta <= 1.0)
    ewrite(2, *) "Beta = ", beta
    
    have_advection = .not. have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/advection_terms/exclude_advection_terms")
    if(have_advection) then
      ewrite(2, *) "Including advection"
      
      integrate_advection_by_parts = have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/advection_terms/integrate_advection_by_parts")
      if(integrate_advection_by_parts) then
        ewrite(2, *) "Integrating advection terms by parts"
      end if
    else
      integrate_advection_by_parts = .false.
      ewrite(2, *) "Excluding advection"
    end if
    
    have_mass = .not. have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/mass_terms/exclude_mass_terms")
    if(have_mass) then
      ewrite(2, *) "Including mass"
      
      lump_mass = have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/mass_terms/lump_mass_matrix")
      if(lump_mass) then
        ewrite(2, *) "Lumping mass"
      end if
    else
      lump_mass = .false.
      ewrite(2, *) "Excluding mass"
    end if
    
    ! are we moving the mesh?
    move_mesh = (have_option("/mesh_adaptivity/mesh_movement") .and. have_mass)
    if(move_mesh) then
      ewrite(2,*) "Moving the mesh"
      old_positions => extract_vector_field(state, "OldCoordinate")
      ewrite_minmax(old_positions)
      new_positions => extract_vector_field(state, "IteratedCoordinate")
      ewrite_minmax(new_positions)
      
      ! Grid velocity
      grid_velocity => extract_vector_field(state, "GridVelocity")
      assert(grid_velocity%dim == mesh_dim(t))
      assert(ele_count(grid_velocity) == ele_count(t))
      
      ewrite(2, *) "Grid velocity:"    
      ewrite_minmax(grid_velocity)
    else
      ewrite(2,*) "Not moving the mesh"
    end if
    
    allocate(supg_element(num_threads))
    if(have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/stabilisation/streamline_upwind")) then
      ewrite(2, *) "Streamline upwind stabilisation"
      stabilisation_scheme = STABILISATION_STREAMLINE_UPWIND
      call get_upwind_options(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/stabilisation/streamline_upwind", &
          & nu_bar_scheme, nu_bar_scale)
      if(move_mesh) then
        FLExit("Haven't thought about how mesh movement works with stabilisation yet.")
      end if
    else if(have_option(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/stabilisation/streamline_upwind_petrov_galerkin")) then
      ewrite(2, *) "SUPG stabilisation"
      stabilisation_scheme = STABILISATION_SUPG
      call get_upwind_options(trim(t%option_path) // "/prognostic/spatial_discretisation/continuous_galerkin/stabilisation/streamline_upwind_petrov_galerkin", &
          & nu_bar_scheme, nu_bar_scale)
      ! Note this is not mixed mesh safe (but then nothing really is)
      ! You need 1 supg_element per thread.
      do i = 1, num_threads
         supg_element(i)=make_supg_element(ele_shape(t,1))
      end do
      if(move_mesh) then
        FLExit("Haven't thought about how mesh movement works with stabilisation yet.")
      end if
    else
      ewrite(2, *) "No stabilisation"
      stabilisation_scheme = STABILISATION_NONE
    end if
    
    ! PhaseVolumeFraction for multiphase flow simulations
    if(option_count("/material_phase/vector_field::Velocity/prognostic") > 1) then
       multiphase = .true.
       vfrac => extract_scalar_field(state, "PhaseVolumeFraction")
       call allocate(nvfrac, vfrac%mesh, "NonlinearPhaseVolumeFraction")
       call zero(nvfrac)
       call get_nonlinear_volume_fraction(state, nvfrac)
      
       ewrite_minmax(nvfrac)
    else
       multiphase = .false.
       call allocate(nvfrac, t%mesh, "DummyNonlinearPhaseVolumeFraction", field_type=FIELD_TYPE_CONSTANT)
       call set(nvfrac, 1.0)
    end if
      
    call allocate(dummydensity, t%mesh, "DummyDensity", field_type=FIELD_TYPE_CONSTANT)
    call set(dummydensity, 1.0)
    ! find out equation type and hence if density is needed or not
    equation_type=equation_type_index(trim(t%option_path))
    select case(equation_type)
    case(FIELD_EQUATION_ADVECTIONDIFFUSION)
      ewrite(2,*) "Solving advection-diffusion equation"
      ! density not needed so use a constant field for assembly
      density => dummydensity
      olddensity => dummydensity
      density_theta = 1.0
      pressure => dummydensity

    case(FIELD_EQUATION_INTERNALENERGY)
      ewrite(2,*) "Solving internal energy equation"
      if(move_mesh) then
        FLExit("Haven't implemented a moving mesh energy equation yet.")
      end if
      
      ! Get old and current densities
      call get_option(trim(t%option_path)//'/prognostic/equation[0]/density[0]/name', &
                      density_name)
      density=>extract_scalar_field(state, trim(density_name))
      ewrite_minmax(density)
      olddensity=>extract_scalar_field(state, "Old"//trim(density_name))
      ewrite_minmax(olddensity)
      
      if(have_option(trim(state%option_path)//'/equation_of_state/compressible')) then         
         call get_option(trim(density%option_path)//"/prognostic/temporal_discretisation/theta", density_theta)
         compressible = .true.
         
         ! We always include the p*div(u) term if this is the compressible phase.
         pressure=>extract_scalar_field(state, "Pressure")
         ewrite_minmax(pressure)
      else
         ! Since the particle phase is always incompressible then its Density
         ! will not be prognostic. Just use a fixed theta value of 1.0.
         density_theta = 1.0
         compressible = .false.
         
         ! Don't include the p*div(u) term if this is the incompressible particle phase.
         pressure => dummydensity
      end if

    case(FIELD_EQUATION_KEPSILON)
      ewrite(2,*) "Solving k-epsilon equation"
      if(move_mesh) then
        FLExit("Haven't implemented a moving mesh k-epsilon equation yet.")
      end if
      
      ! Depending on the equation type, extract the density or set it to some dummy field allocated above
      temp_velocity_ptr => extract_vector_field(state, "Velocity")
      call get_option(trim(temp_velocity_ptr%option_path)//"/prognostic/equation[0]/name", velocity_equation_type)
      select case(velocity_equation_type)
         case("LinearMomentum")
            density=>extract_scalar_field(state, "Density")
            olddensity => dummydensity
            density_theta = 1.0
         case("Boussinesq")
            density=>dummydensity
            olddensity => dummydensity
            density_theta = 1.0
         case("Drainage")
            density=>dummydensity
            olddensity => dummydensity
            density_theta = 1.0
         case default
            ! developer error... out of sync options input and code
            FLAbort("Unknown equation type for velocity")
      end select
      ewrite_minmax(density)

    case default
      FLExit("Unknown field equation type for cg advection diffusion.")
    end select
    
    ! Step 3: Assembly
    
    call zero(matrix)
    call zero(rhs)
    
    call profiler_tic(t, "advection_diffusion_loop_overhead")

#ifdef _OPENMP
    cache_valid = prepopulate_transform_cache(positions)
#endif
    if (have_option('/material_phase[0]/subgridscale_parameterisations/MRF/')) then
       have_mrf=.true.
    else
       have_mrf=.false.
    end if

    call get_mesh_colouring(state, t%mesh, COLOURING_CG1, colours)
    call profiler_toc(t, "advection_diffusion_loop_overhead")

    call profiler_tic(t, "advection_diffusion_loop")
    have_source_int = have_source
    have_diffu_int = have_diffusivity
    nvfrac_associated = ASSOCIATED(nvfrac%val)
    write(*,*)"have_****",have_source_int,nvfrac_associated,have_diffusivity
 
    !!!!!lils gpu
    start_time = MPI_Wtime()
    call adv_create_stream_c() 
    call adv_alloc_gpu_mem_c(element_count(t%mesh),t%mesh%nodes)  
    call adv_shape_dn_to_device(positions%mesh%shape%dn,positions%mesh%shape%dn,positions%mesh%shape%quadrature%weight) 
    call adv_set_gpu_data(element_count(t%mesh),t%mesh%nodes,&
    &t%mesh%ndglno, t%mesh%shape%n, t%val, rhs%val,positions%val,velocity%val,&
    &diffusivity%val,density%val,nvfrac%val,source%val,&
    &nvfrac_associated,have_source_int,have_diffu_int)
    !!!!!lils gpu 
!    !$OMP PARALLEL DEFAULT(SHARED) &
!    !$OMP PRIVATE(clr, len, nnid, ele, thread_num)

!#ifdef _OPENMP    
!    thread_num = omp_get_thread_num()
!#else
!    thread_num=0
!#endif
!    colour_loop: do clr = 1, size(colours)

!      len = key_count(colours(clr))
!      !$OMP DO SCHEDULE(STATIC)
!      element_loop: do ele = 1, element_count(t%mesh)
         !ele = fetch(colours(clr), nnid)
         !S2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023
!         call assemble_advection_diffusion_element_cg(ele, t, matrix, rhs, loop_rhs_addto(:,ele),loop_matrix_addto(:,:,ele),&
!              loop_detwei(:,ele),loop_du_t(:,:,:,ele),loop_dt_t(:,:,:,ele), & 
!              positions, old_positions, new_positions, &
!              velocity, grid_velocity, &
!              source, absorption, diffusivity, &
!              density, olddensity, pressure, nvfrac, &
              !supg_element(thread_num+1)) ! FA Original Code  
!              supg_element(thread_num+1), counterG,have_mrf) !@#$% jxli Apr 2023 
              !E2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023
!      end do element_loop
!      !$OMP END DO

!    end do colour_loop
!    !$OMP END PARALLEL
  
   call alloc_adv_assemble(matrix%sparsity%findrm, matrix%sparsity%colm, t%mesh%ndglno, &
   ele_loc(t, 1), ele_ngi(t, 1), element_count(t%mesh), size(matrix%sparsity%findrm,1),size(matrix%val,1))
   call gpu_adv_assemble(element_count(t%mesh),t%mesh%nodes,&
   beta,dt_theta,rhs%val,have_source_int,have_diffu_int)
   
 !  rhs_addto_loop: do ele = 1, element_count(t%mesh)
 !   element_nodes => ele_nodes(t, ele)
    !call addto(rhs, element_nodes, loop_rhs_addto(:,ele))
 !  end do rhs_addto_loop 
   
 !  matrix_addto_loop: do ele = 1, element_count(t%mesh)
 !   element_nodes => ele_nodes(t, ele)
 !   call addto(matrix, element_nodes, element_nodes, loop_matrix_addto(:,:,ele))
 !  end do matrix_addto_loop 
   
   call assemble_csr_matrix_gpu(matrix%val,&
   ele_loc(t, 1), ele_ngi(t, 1), element_count(t%mesh), size(matrix%sparsity%findrm,1),size(matrix%val,1))
    end_time = MPI_Wtime()
    elapsed_time = end_time - start_time
    write(*,*)"GPU assemble advection diffusion time:",elapsed_time
   !!!!lils gpu
   call dealloc_adv_mem()
   call dealloc_adv_assemble()
   call adv_destroy_stream()  
   !!!!lils gpu
 
 call profiler_toc(t, "advection_diffusion_loop")

    ! Add the source directly to the rhs if required 
    ! which must be included before dirichlet BC's.
    if (add_src_directly_to_rhs) call addto(rhs, source)
    
    
    ! Step 4: Boundary conditions
    
    if( &
      & (integrate_advection_by_parts .and. have_advection) &
      & .or. have_diffusivity &
      & ) then
    
      allocate(t_bc_types(surface_element_count(t)))
      call get_entire_boundary_condition(t, &
                                         (/ "neumann      ", &
                                            "weakdirichlet", &
                                            "internal     ", &
                                            "robin        "/), &
                                          t_bc, &
                                          t_bc_types, &
                                          boundary_second_value = t_bc_2)

      if(any(t_bc_types /= 0)) then
        call ewrite_bc_counts(2, t_bc_types)
      end if
    
      do i = 1, surface_element_count(t)
        if(t_bc_types(i)==BC_TYPE_INTERNAL) cycle
        call assemble_advection_diffusion_face_cg(i, t_bc_types(i), t, t_bc, t_bc_2, &
                                                  matrix, rhs, &
                                                  positions, velocity, grid_velocity, &
                                                  density, olddensity, nvfrac)
      end do
    
      call deallocate(t_bc)
      call deallocate(t_bc_2)
      deallocate(t_bc_types)
    
    end if
    
    ewrite(2, *) "Applying strong Dirichlet boundary conditions"
    call apply_dirichlet_conditions(matrix, rhs, t, dt)
    
    ewrite_minmax(rhs)
    
    call deallocate(velocity)
    call deallocate(nvfrac)
    call deallocate(dummydensity)
    if (stabilisation_scheme == STABILISATION_SUPG) then
       do i = 1, num_threads
          call deallocate(supg_element(i))
       end do
    end if
    deallocate(supg_element)

    ewrite(1, *) "Exiting assemble_advection_diffusion_cg"
    
  end subroutine assemble_advection_diffusion_cg
  
  subroutine ewrite_bc_counts(debug_level, bc_types)
    !!< A simple subroutine to count and output the number of elements with
    !!< each boundary conditions (combines counts into a single surface
    !!< element loop).
    
    integer, intent(in) :: debug_level
    integer, dimension(:), intent(in) :: bc_types
    
    integer :: i, nneumann, nweak_dirichlet, ninternal, nrobin
    
    if(debug_level > current_debug_level) return
    
    nneumann = 0
    nweak_dirichlet = 0
    ninternal = 0
    nrobin = 0
    do i = 1, size(bc_types)
      select case(bc_types(i))
        case(BC_TYPE_NEUMANN)
          nneumann = nneumann + 1
        case(BC_TYPE_WEAKDIRICHLET)
          nweak_dirichlet = nweak_dirichlet + 1
        case(BC_TYPE_INTERNAL)
          ninternal = ninternal + 1
        case(BC_TYPE_ROBIN)
          nrobin = nrobin + 1
        case(0)
        case default
          ! this is a code error
          ewrite(-1, *) "For boundary condition type: ", bc_types(i)
          FLAbort("Unrecognised boundary condition type")
      end select
    end do
    
    ewrite(debug_level, *) "Surface elements with Neumann boundary condition: ", nneumann
    ewrite(debug_level, *) "Surface elements with weak Dirichlet boundary condition: ", nweak_dirichlet
    ewrite(debug_level, *) "Surface elements with internal or periodic boundary condition: ", ninternal
    ewrite(debug_level, *) "Surface elements with Robin boundary condition: ", nrobin
    
  end subroutine ewrite_bc_counts
  
  subroutine assemble_advection_diffusion_element_cg(ele, t, matrix, rhs, rhs_addto,matrix_addto,detwei,du_t,dt_t,&
                                      positions, old_positions, new_positions, &
                                      velocity, grid_velocity, &
                                      source, absorption, diffusivity, &
                                      !density, olddensity, pressure, nvfrac, supg_shape) ! FA Original Code
                                      density, olddensity, pressure, nvfrac, supg_shape, counterG,have_mrf) !@#$% jxli Apr 2023 
    integer, intent(in) :: ele
    type(scalar_field), intent(in) :: t
    type(csr_matrix), intent(inout) :: matrix
    type(scalar_field), intent(inout) :: rhs
    type(vector_field), intent(in) :: positions
    type(vector_field), pointer :: old_positions, new_positions
    type(vector_field), intent(in) :: velocity
    type(vector_field), pointer :: grid_velocity
    type(scalar_field), intent(in) :: source
    type(scalar_field), intent(in) :: absorption
    type(tensor_field), intent(in) :: diffusivity
    type(scalar_field), intent(in) :: density
    type(scalar_field), intent(in) :: olddensity
    type(scalar_field), intent(in) :: pressure
    type(scalar_field), intent(in) :: nvfrac
    type(element_type), intent(inout) :: supg_shape
    real, dimension(ele_loc(t, ele)),intent(inout) :: rhs_addto
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)),intent(inout) :: matrix_addto
    real, dimension(ele_loc(velocity, ele), ele_ngi(velocity, ele), mesh_dim(t)),intent(inout) :: du_t 
    real, dimension(ele_loc(t, ele), ele_ngi(t, ele), mesh_dim(t)),intent(inout) :: dt_t
    real, dimension(ele_ngi(t, ele)),intent(inout) :: detwei
    
    logical, intent(in) :: have_mrf
    integer, dimension(:), pointer :: element_nodes
    !real, dimension(ele_ngi(t, ele)) :: detwei, 
    real, dimension(ele_ngi(t, ele)) ::detwei_old, detwei_new
    !real, dimension(ele_loc(t, ele), ele_ngi(t, ele), mesh_dim(t)) :: dt_t
    real, dimension(ele_loc(density, ele), ele_ngi(density, ele), mesh_dim(density)) :: drho_t
    !real, dimension(ele_loc(velocity, ele), ele_ngi(velocity, ele), mesh_dim(t)) :: du_t 
    real, dimension(ele_loc(positions, ele), ele_ngi(velocity, ele), mesh_dim(t)) :: dug_t 
    ! Derivative of shape function for nvfrac field
    real, dimension(ele_loc(nvfrac, ele), ele_ngi(nvfrac, ele), mesh_dim(nvfrac)) :: dnvfrac_t
    
    real, dimension(mesh_dim(t), mesh_dim(t), ele_ngi(t, ele)) :: j_mat 
    type(element_type) :: test_function
    type(element_type), pointer :: t_shape
    
    ! What we will be adding to the matrix and RHS - assemble these as we
    ! go, so that we only do the calculations we really need
   ! real, dimension(ele_loc(t, ele)),intent(inout) :: rhs_addto
   ! real, dimension(ele_loc(t, ele), ele_loc(t, ele)),intent(inout) :: matrix_addto
    
    ! Do we open MRF_paramerisation? Yes, this is used for transfer countergradient. !@#$% jxli Apr 2023
    type(tensor_field), intent(out)     :: counterG

#ifdef DDEBUG
    assert(ele_ngi(positions, ele) == ele_ngi(t, ele))
    assert(ele_ngi(velocity, ele) == ele_ngi(t, ele))
    if(have_diffusivity) then
      assert(ele_ngi(diffusivity, ele) == ele_ngi(t, ele))
    end if
    if(have_source) then
      assert(ele_ngi(source, ele) == ele_ngi(t, ele))
    end if
    if(have_absorption) then
      assert(ele_ngi(absorption, ele) == ele_ngi(t, ele))
    end if
    if(move_mesh) then
      ! the following has been assumed in the declarations above
      assert(ele_loc(grid_velocity, ele) == ele_loc(positions, ele))
      assert(ele_ngi(grid_velocity, ele) == ele_ngi(velocity, ele))
    end if
#endif

    matrix_addto = 0.0
    rhs_addto = 0.0
    
    t_shape => ele_shape(t, ele)
      
    ! Step 1: Transform
    
    if(.not. have_advection .and. .not. have_diffusivity) then
      call transform_to_physical(positions, ele, detwei = detwei)
    else if(any(stabilisation_scheme == (/STABILISATION_STREAMLINE_UPWIND, STABILISATION_SUPG/))) then
      call transform_to_physical(positions, ele, t_shape, &
        & dshape = dt_t, detwei = detwei, j = j_mat)
    else
    !here  write(*,*)"which transform_to_physical "
    !  call transform_to_physical(positions, ele, t_shape, &
    !    & dshape = dt_t, detwei = detwei)
    end if
    
    if(have_advection.or.(equation_type==FIELD_EQUATION_INTERNALENERGY).or.equation_type==FIELD_EQUATION_KEPSILON) then
     !here write(*,*)"which transform_to_physical "
    !  call transform_to_physical(positions, ele, &
    !       & ele_shape(velocity, ele), dshape = du_t)
    end if
    
    if(have_advection.and.move_mesh.and..not.integrate_advection_by_parts) then
      call transform_to_physical(positions, ele, &
          & ele_shape(grid_velocity, ele), dshape = dug_t)
    end if    
    
    if(move_mesh) then
      call transform_to_physical(old_positions, ele, detwei=detwei_old)
      call transform_to_physical(new_positions, ele, detwei=detwei_new)
    end if
    
    if(have_advection.and.(equation_type==FIELD_EQUATION_INTERNALENERGY .or. equation_type==FIELD_EQUATION_KEPSILON)) then
      if(ele_shape(density, ele)==t_shape) then
        drho_t = dt_t
      else
        call transform_to_physical(positions, ele, &
          & ele_shape(density, ele), dshape = drho_t)
      end if
    end if
    
    if(have_advection .and. multiphase .and. (equation_type==FIELD_EQUATION_INTERNALENERGY)) then
      ! If the field and nvfrac meshes are different, then we need to
      ! compute the derivatives of the nvfrac shape functions.    
      if(ele_shape(nvfrac, ele) == t_shape) then
         dnvfrac_t = dt_t
      else
         call transform_to_physical(positions, ele, ele_shape(nvfrac, ele), dshape=dnvfrac_t)
      end if
    end if

    ! Step 2: Set up test function
    
    select case(stabilisation_scheme)
      case(STABILISATION_SUPG)
        if(have_diffusivity) then
          call supg_test_function(supg_shape, t_shape, dt_t, ele_val_at_quad(velocity, ele), j_mat, diff_q = ele_val_at_quad(diffusivity, ele), &
            & nu_bar_scheme = nu_bar_scheme, nu_bar_scale = nu_bar_scale)
        else
          call supg_test_function(supg_shape, t_shape, dt_t, ele_val_at_quad(velocity, ele), j_mat, &
            & nu_bar_scheme = nu_bar_scheme, nu_bar_scale = nu_bar_scale)
        end if
        test_function = supg_shape
      case default
      !here write(*,*)"which test_function"
        test_function = t_shape
    end select
    ! Important note: with SUPG the test function derivatives have not been
    ! modified - i.e. dt_t is currently used everywhere. This is fine for P1,
    ! but is not consistent for P>1.

    ! Step 3: Assemble contributions
    
    ! Mass
    if(have_mass)then
     !here write(*,*)"have mass"
!     call add_mass_element_cg(ele, test_function, t, density, olddensity,  nvfrac, detwei, detwei_old, detwei_new, matrix_addto, rhs_addto)
    endif 
    ! Advection
    if(have_advection)then
      !here write(*,*)"have advection"
      
!      call add_advection_element_cg(ele, test_function, t, &
!                                        velocity, grid_velocity, diffusivity, &
!                                        density, olddensity, nvfrac, &
!                                        dt_t, du_t, dug_t, drho_t, dnvfrac_t, detwei, j_mat, matrix_addto, rhs_addto)
    endif    
    ! Absorption
    if(have_absorption)then
     !not write(*,*)"have absorption"
     call add_absorption_element_cg(ele, test_function, t, absorption, detwei, matrix_addto, rhs_addto)
    endif 
    ! Diffusivity
    if(have_diffusivity) then
     !here write(*,*)"have diffusivity"
    ! call add_diffusivity_element_cg(ele, t, diffusivity, dt_t, nvfrac, detwei, matrix_addto, rhs_addto)
    endif 
    ! Source
    if(have_source .and. (.not. add_src_directly_to_rhs)) then 
    !here  write(*,*)"have source"
    ! call add_source_element_cg(ele, test_function, t, source, detwei, rhs_addto)
    end if
    
    !!****************************************************************************************************
    !!***** option: atmosphere, source term generating a new subroutine  !@#$% jxli Apr 2023 *************
    !!****************************************************************************************************

    !S2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023
    ! countergradient term in MRF parameterisation
    if (have_mrf) then
      if(ele==1)then
      write(*,*)"have_mrf"
      endif
       call add_MRF_countergradient_correction_advection_element_cg(ele, t, dt_t, detwei, rhs_addto, counterG)
    end if
    !E2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023

    ! Pressure
    if(equation_type==FIELD_EQUATION_INTERNALENERGY .and. compressible) then
     !not write(*,*)"add_pressure"
       call add_pressurediv_element_cg(ele, test_function, t, velocity, pressure, nvfrac, du_t, detwei, rhs_addto)
    end if
                                                                                  
    
    ! Step 4: Insertion
            
!    element_nodes => ele_nodes(t, ele)
!    call addto(matrix, element_nodes, element_nodes, matrix_addto)
!    call addto(rhs, element_nodes, rhs_addto)
    !call exit(0)
  end subroutine assemble_advection_diffusion_element_cg
  
  subroutine add_mass_element_cg(ele, test_function, t, density, olddensity, nvfrac, detwei, detwei_old, detwei_new, matrix_addto, rhs_addto)
    integer, intent(in) :: ele
    type(element_type), intent(in) :: test_function
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: density, olddensity
    type(scalar_field), intent(in) :: nvfrac    
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei, detwei_old, detwei_new
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)), intent(inout) :: matrix_addto
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
    
    integer :: i
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)) :: mass_matrix
    
    real, dimension(ele_ngi(density,ele)) :: density_at_quad
    
    assert(have_mass)
    
    select case(equation_type)
    case(FIELD_EQUATION_INTERNALENERGY)
      assert(ele_ngi(density, ele)==ele_ngi(olddensity, ele))
      
      density_at_quad = ele_val_at_quad(olddensity, ele)

      if(move_mesh) then
        ! needs to be evaluated at t+dt
        mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei_new*density_at_quad)
      else
        if(multiphase) then
          mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei*density_at_quad*ele_val_at_quad(nvfrac, ele))
        else
          mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei*density_at_quad)
        end if
      end if
    case(FIELD_EQUATION_KEPSILON)      
      density_at_quad = ele_val_at_quad(density, ele)
      mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei*density_at_quad)
    case default
    
      if(move_mesh) then
        ! needs to be evaluated at t+dt
        mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei_new)
      else
        mass_matrix = shape_shape(test_function, ele_shape(t, ele), detwei)
      end if
      
    end select
    
    if(lump_mass) then
      do i = 1, size(matrix_addto, 1)
        matrix_addto(i, i) = matrix_addto(i, i) + sum(mass_matrix(i, :))
      end do
    else
      matrix_addto = matrix_addto + mass_matrix
    end if
  
    if(move_mesh) then
      ! In the unaccelerated form we solve:
      !  /
      !  |  N^{n+1} T^{n+1}/dt - N^{n} T^n/dt + ... = f
      !  /
      ! so in accelerated form:
      !  /
      !  |  N^{n+1} dT + (N^{n+1}- N^{n}) T^n/dt + ... = f
      !  /
      ! where dT=(T^{n+1}-T^{n})/dt is the acceleration.
      ! Put the (N^{n+1}-N^{n}) T^n term on the rhs
      mass_matrix = shape_shape(test_function, ele_shape(t, ele), (detwei_new-detwei_old))
      if(lump_mass) then
        rhs_addto = rhs_addto - sum(mass_matrix, 2)*ele_val(t, ele)/local_dt
      else
        rhs_addto = rhs_addto - matmul(mass_matrix, ele_val(t, ele))/local_dt
      end if
    end if

  end subroutine add_mass_element_cg
  
  subroutine add_advection_element_cg(ele, test_function, t, &
                                velocity, grid_velocity, diffusivity, &
                                density, olddensity, nvfrac, &
                                dt_t, du_t, dug_t, drho_t, dnvfrac_t, detwei, j_mat, matrix_addto, rhs_addto)
    integer, intent(in) :: ele
    type(element_type), intent(in) :: test_function
    type(scalar_field), intent(in) :: t
    type(vector_field), intent(in) :: velocity
    type(vector_field), pointer :: grid_velocity
    type(tensor_field), intent(in) :: diffusivity
    type(scalar_field), intent(in) :: density, olddensity
    type(scalar_field), intent(in) :: nvfrac
    real, dimension(ele_loc(t, ele), ele_ngi(t, ele), mesh_dim(t)), intent(in) :: dt_t
    real, dimension(ele_loc(velocity, ele), ele_ngi(velocity, ele), mesh_dim(t)) :: du_t
    real, dimension(:, :, :) :: dug_t
    real, dimension(ele_loc(density, ele), ele_ngi(density, ele), mesh_dim(density)), intent(in) :: drho_t
    real, dimension(:, :, :), intent(in) :: dnvfrac_t
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei
    real, dimension(mesh_dim(t), mesh_dim(t), ele_ngi(t, ele)), intent(in) :: j_mat 
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)), intent(inout) :: matrix_addto
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
    
    real, dimension(ele_loc(t, ele), ele_loc(t,ele)) :: advection_mat
    real, dimension(velocity%dim, ele_ngi(velocity, ele)) :: velocity_at_quad
    real, dimension(ele_ngi(velocity, ele)) :: velocity_div_at_quad
    type(element_type), pointer :: t_shape
    
    real, dimension(ele_ngi(density, ele)) :: density_at_quad
    real, dimension(velocity%dim, ele_ngi(density, ele)) :: densitygrad_at_quad
    real, dimension(ele_ngi(density, ele)) :: udotgradrho_at_quad
    
    real, dimension(ele_ngi(t, ele)) :: nvfrac_at_quad
    real, dimension(velocity%dim, ele_ngi(t, ele)) :: nvfracgrad_at_quad
    real, dimension(ele_ngi(t, ele)) :: udotgradnvfrac_at_quad
    
    assert(have_advection)
    
    t_shape => ele_shape(t, ele)
    
    velocity_at_quad = ele_val_at_quad(velocity, ele)
    if(move_mesh) then
      velocity_at_quad = velocity_at_quad - ele_val_at_quad(grid_velocity, ele)
    end if
    
    select case(equation_type)
    case(FIELD_EQUATION_INTERNALENERGY)
      assert(ele_ngi(density, ele)==ele_ngi(olddensity, ele))
      
      density_at_quad = density_theta*ele_val_at_quad(density, ele)&
                       +(1.-density_theta)*ele_val_at_quad(olddensity, ele)
      densitygrad_at_quad = density_theta*ele_grad_at_quad(density, ele, drho_t) &
                           +(1.-density_theta)*ele_grad_at_quad(olddensity, ele, drho_t)
      udotgradrho_at_quad = sum(densitygrad_at_quad*velocity_at_quad, 1)
      
      if(multiphase) then
         nvfrac_at_quad = ele_val_at_quad(nvfrac, ele)
      end if

    case(FIELD_EQUATION_KEPSILON)
      density_at_quad = ele_val_at_quad(density, ele)
      densitygrad_at_quad = ele_grad_at_quad(density, ele, drho_t)
      udotgradrho_at_quad = sum(densitygrad_at_quad*velocity_at_quad, 1)
    end select
                
    if(integrate_advection_by_parts) then
      ! element advection matrix
      !    /                                        /
      !  - | (grad N_A dot nu) N_B dV - (1. - beta) | N_A ( div nu ) N_B dV
      !    /                                        /
      select case(equation_type)
      case(FIELD_EQUATION_INTERNALENERGY)
        if(multiphase) then      
           advection_mat = -dshape_dot_vector_shape(dt_t, velocity_at_quad, t_shape, detwei*density_at_quad*nvfrac_at_quad)
        else
           advection_mat = -dshape_dot_vector_shape(dt_t, velocity_at_quad, t_shape, detwei*density_at_quad)
        end if
           
        if(abs(1.0 - beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          
          if(multiphase) then
          
             nvfracgrad_at_quad = ele_grad_at_quad(nvfrac, ele, dnvfrac_t)
             udotgradnvfrac_at_quad = sum(nvfracgrad_at_quad*velocity_at_quad, 1)
             
             advection_mat = advection_mat - (1.0-beta) * ( shape_shape(test_function, t_shape, detwei*velocity_div_at_quad*density_at_quad*nvfrac_at_quad) &
                                                           + shape_shape(test_function, t_shape, detwei*nvfrac_at_quad*udotgradrho_at_quad) &
                                                           + shape_shape(test_function, t_shape, detwei*density_at_quad*udotgradnvfrac_at_quad) )
          else
             advection_mat = advection_mat - (1.0-beta) * shape_shape(test_function, t_shape, (velocity_div_at_quad*density_at_quad + udotgradrho_at_quad)*detwei)
          end if
        end if
      case(FIELD_EQUATION_KEPSILON)
        advection_mat = -dshape_dot_vector_shape(dt_t, velocity_at_quad, t_shape, detwei*density_at_quad)
        if(abs(1.0 - beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          advection_mat = advection_mat &
                    - (1.0-beta) * shape_shape(test_function, t_shape, (velocity_div_at_quad*density_at_quad &
                                                                      +udotgradrho_at_quad)* detwei)
        end if
      case default
        advection_mat = -dshape_dot_vector_shape(dt_t, velocity_at_quad, t_shape, detwei)
        if(abs(1.0 - beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          advection_mat = advection_mat &
                    - (1.0-beta)*shape_shape(test_function, t_shape, velocity_div_at_quad*detwei)
        end if
      end select
    else
      ! element advection matrix
      !  /                                 /
      !  | N_A (nu dot grad N_B) dV + beta | N_A ( div nu ) N_B dV
      !  /                                 /
      select case(equation_type)
      case(FIELD_EQUATION_INTERNALENERGY)
        
        if(multiphase) then     
           ! vfrac*rho*nu*grad(internalenergy)
           advection_mat = shape_vector_dot_dshape(test_function, velocity_at_quad, dt_t, detwei*density_at_quad*nvfrac_at_quad)
        else
           advection_mat = shape_vector_dot_dshape(test_function, velocity_at_quad, dt_t, detwei*density_at_quad)
        end if
        
        if(abs(beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          
          if(multiphase) then
             ! advection_mat + internalenergy*div(vfrac*rho*nu)
             ! Split up div(vfrac*rho*nu) = vfrac*rho*div(nu) + nu*grad(vfrac*rho) = vfrac*rho*div(nu) + nu*(vfrac*grad(rho) + rho*grad(nvfrac))
             
             nvfracgrad_at_quad = ele_grad_at_quad(nvfrac, ele, dnvfrac_t)
             udotgradnvfrac_at_quad = sum(nvfracgrad_at_quad*velocity_at_quad, 1)
             
             advection_mat = advection_mat + beta * ( shape_shape(test_function, t_shape, detwei*velocity_div_at_quad*density_at_quad*nvfrac_at_quad) &
                                                     + shape_shape(test_function, t_shape, detwei*nvfrac_at_quad*udotgradrho_at_quad) &
                                                     + shape_shape(test_function, t_shape, detwei*density_at_quad*udotgradnvfrac_at_quad) )
          else
             advection_mat = advection_mat + beta*shape_shape(test_function, t_shape, (velocity_div_at_quad*density_at_quad &
                                                              +udotgradrho_at_quad)*detwei)
          end if
        end if
      case(FIELD_EQUATION_KEPSILON)
        advection_mat = shape_vector_dot_dshape(test_function, velocity_at_quad, dt_t, detwei*density_at_quad)
        if(abs(beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          advection_mat = advection_mat &
                    + beta*shape_shape(test_function, t_shape, (velocity_div_at_quad*density_at_quad &
                                                                +udotgradrho_at_quad)*detwei)
        end if
      case default
        advection_mat = shape_vector_dot_dshape(test_function, velocity_at_quad, dt_t, detwei)
        if(abs(beta) > epsilon(0.0)) then
          velocity_div_at_quad = ele_div_at_quad(velocity, ele, du_t)
          advection_mat = advection_mat &
                    + beta*shape_shape(test_function, t_shape, velocity_div_at_quad*detwei)
        end if
        if(move_mesh) then
          advection_mat = advection_mat &
                    - shape_shape(test_function, t_shape, ele_div_at_quad(grid_velocity, ele, dug_t)*detwei)
        end if
      end select
    end if
    
    ! Stabilisation
    select case(stabilisation_scheme)
      case(STABILISATION_STREAMLINE_UPWIND)
        if(have_diffusivity) then
          advection_mat = advection_mat + &
            & element_upwind_stabilisation(t_shape, dt_t, velocity_at_quad, j_mat, detwei, &
            & diff_q = ele_val_at_quad(diffusivity, ele), nu_bar_scheme = nu_bar_scheme, nu_bar_scale = nu_bar_scale)
        else
          advection_mat = advection_mat + &
            & element_upwind_stabilisation(t_shape, dt_t, velocity_at_quad, j_mat, detwei, &
            & nu_bar_scheme = nu_bar_scheme, nu_bar_scale = nu_bar_scale)
        end if
      case default
    end select
    
      
    if(abs(dt_theta) > epsilon(0.0)) then
      matrix_addto = matrix_addto + dt_theta * advection_mat
    end if
    
    rhs_addto = rhs_addto - matmul(advection_mat, ele_val(t, ele))
  end subroutine add_advection_element_cg
  
  !S2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023
  ! add the countergradient correction term to Advection equation 
  subroutine add_MRF_countergradient_correction_advection_element_cg(ele, t, dt_t, detwei, rhs_addto, counterG)

    integer, intent(in)                                                        :: ele
    type(scalar_field), intent(in)                                             :: t
    real, dimension(ele_loc(t, ele), ele_ngi(t, ele), mesh_dim(t)), intent(in) :: dt_t
    real, dimension(ele_ngi(t, ele)), intent(in)                               :: detwei
    real, dimension(ele_loc(t, ele)), intent(inout)                            :: rhs_addto
    real, dimension(3, ele_ngi(t, ele))                                        :: vector
    type(tensor_field), intent(out)                                            :: counterG

    vector(1, :) = 0.0
    vector(2, :) = 0.0
    vector(3, :) = counterG%val(3, 3, :) ! countergradient correction term
    rhs_addto = rhs_addto - dshape_dot_vector_rhs(dt_t, vector, detwei)

  end subroutine add_MRF_countergradient_correction_advection_element_cg
  !E2023-04~No.005================= MRF Package CounterG !@#$% jxli Apr 2023

  subroutine add_source_element_cg(ele, test_function, t, source, detwei, rhs_addto)
    integer, intent(in) :: ele
    type(element_type), intent(in) :: test_function
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: source
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
   
    assert(have_source)
   
    rhs_addto = rhs_addto + shape_rhs(test_function, detwei * ele_val_at_quad(source, ele))
    
  end subroutine add_source_element_cg
  
  subroutine add_absorption_element_cg(ele, test_function, t, absorption, detwei, matrix_addto, rhs_addto)
    integer, intent(in) :: ele
    type(element_type), intent(in) :: test_function
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: absorption
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)), intent(inout) :: matrix_addto
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
    
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)) ::  absorption_mat
    
    assert(have_absorption)
    
    absorption_mat = shape_shape(test_function, ele_shape(t, ele), detwei * ele_val_at_quad(absorption, ele))
    
    if(abs(dt_theta) > epsilon(0.0)) matrix_addto = matrix_addto + dt_theta * absorption_mat
    
    rhs_addto = rhs_addto - matmul(absorption_mat, ele_val(t, ele))
    
  end subroutine add_absorption_element_cg
  
  subroutine add_diffusivity_element_cg(ele, t, diffusivity, dt_t, nvfrac, detwei, matrix_addto, rhs_addto)
    integer, intent(in) :: ele
    type(scalar_field), intent(in) :: t, nvfrac
    type(tensor_field), intent(in) :: diffusivity
    real, dimension(ele_loc(t, ele), ele_ngi(t, ele), mesh_dim(t)), intent(in) :: dt_t
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)), intent(inout) :: matrix_addto
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
    
    real, dimension(diffusivity%dim(1), diffusivity%dim(2), ele_ngi(diffusivity, ele)) :: diffusivity_gi
    real, dimension(ele_loc(t, ele), ele_loc(t, ele)) :: diffusivity_mat
    
    assert(have_diffusivity)
    
    diffusivity_gi = ele_val_at_quad(diffusivity, ele)

    if(isotropic_diffusivity) then
      assert(size(diffusivity_gi, 1) > 0)
      if(multiphase .and. equation_type==FIELD_EQUATION_INTERNALENERGY) then
         ! This allows us to use the Diffusivity term as the heat flux term
         ! in the multiphase InternalEnergy equation: div( (k/Cv) * vfrac * grad(ie) ).
         ! The user needs to input k/Cv for the prescribed diffusivity,
         ! where k is the effective conductivity and Cv is the specific heat
         ! at constant volume. We've assumed this will always be isotropic here.
         ! The division by Cv is needed because the heat flux
         ! is defined in terms of temperature T = ie/Cv.
         diffusivity_mat = dshape_dot_dshape(dt_t, dt_t, detwei * diffusivity_gi(1, 1, :) * ele_val_at_quad(nvfrac, ele))
      else
         diffusivity_mat = dshape_dot_dshape(dt_t, dt_t, detwei * diffusivity_gi(1, 1, :))
      end if
    else
      diffusivity_mat = dshape_tensor_dshape(dt_t, diffusivity_gi, dt_t, detwei)
    end if

    if(abs(dt_theta) > epsilon(0.0)) matrix_addto = matrix_addto + dt_theta * diffusivity_mat
    
    rhs_addto = rhs_addto - matmul(diffusivity_mat, ele_val(t, ele))
    
  end subroutine add_diffusivity_element_cg
  
  subroutine add_pressurediv_element_cg(ele, test_function, t, velocity, pressure, nvfrac, du_t, detwei, rhs_addto)
  
    integer, intent(in) :: ele
    type(element_type), intent(in) :: test_function
    type(scalar_field), intent(in) :: t
    type(vector_field), intent(in) :: velocity
    type(scalar_field), intent(in) :: pressure
    type(scalar_field), intent(in) :: nvfrac
    real, dimension(ele_loc(velocity, ele), ele_ngi(velocity, ele), mesh_dim(t)), intent(in) :: du_t
    real, dimension(ele_ngi(t, ele)), intent(in) :: detwei
    real, dimension(ele_loc(t, ele)), intent(inout) :: rhs_addto
    
    assert(equation_type==FIELD_EQUATION_INTERNALENERGY)
    assert(ele_ngi(pressure, ele)==ele_ngi(t, ele))
    
    if(multiphase) then
       ! -p * vfrac * div(nu)
       rhs_addto = rhs_addto - shape_rhs(test_function, ele_div_at_quad(velocity, ele, du_t) * ele_val_at_quad(pressure, ele) * detwei * ele_val_at_quad(nvfrac, ele))
    else
       rhs_addto = rhs_addto - shape_rhs(test_function, ele_div_at_quad(velocity, ele, du_t) * ele_val_at_quad(pressure, ele) * detwei)
    end if
    
  end subroutine add_pressurediv_element_cg
  
  subroutine assemble_advection_diffusion_face_cg(face, bc_type, t, t_bc, t_bc_2, matrix, rhs, positions, velocity, grid_velocity, density, olddensity, nvfrac)
    integer, intent(in) :: face
    integer, intent(in) :: bc_type
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: t_bc
    type(scalar_field), intent(in) :: t_bc_2
    type(csr_matrix), intent(inout) :: matrix
    type(scalar_field), intent(inout) :: rhs
    type(vector_field), intent(in) :: positions
    type(vector_field), intent(in) :: velocity
    type(vector_field), pointer :: grid_velocity
    type(scalar_field), intent(in) :: density
    type(scalar_field), intent(in) :: olddensity
    type(scalar_field), intent(in) :: nvfrac
    
    integer, dimension(face_loc(t, face)) :: face_nodes
    real, dimension(face_ngi(t, face)) :: detwei
    real, dimension(mesh_dim(t), face_ngi(t, face)) :: normal
    
    ! What we will be adding to the matrix and RHS - assemble these as we
    ! go, so that we only do the calculations we really need
    real, dimension(face_loc(t, face)) :: rhs_addto
    real, dimension(face_loc(t, face), face_loc(t, face)) :: matrix_addto
    
    assert(any(bc_type == (/0, BC_TYPE_NEUMANN, BC_TYPE_WEAKDIRICHLET, BC_TYPE_ROBIN/)))
    assert(face_ngi(positions, face) == face_ngi(t, face))
    assert(face_ngi(velocity, face) == face_ngi(t, face))

    matrix_addto = 0.0
    rhs_addto = 0.0
    
    ! Step 1: Transform
    
    if(have_advection .and. integrate_advection_by_parts) then
      call transform_facet_to_physical(positions, face, &
        & detwei_f = detwei, normal = normal)
    else if(have_diffusivity.and.((bc_type == BC_TYPE_NEUMANN).or.(bc_type == BC_TYPE_ROBIN))) then
      call transform_facet_to_physical(positions, face, &
        & detwei_f = detwei)
    end if
    
    ! Note that with SUPG the surface element test function is not modified
          
    ! Step 2: Assemble contributions
    
    ! Advection
    if(have_advection .and. integrate_advection_by_parts) &
      call add_advection_face_cg(face, bc_type, t, t_bc, velocity, grid_velocity, density, olddensity, nvfrac, detwei, normal, matrix_addto, rhs_addto)
    
    ! Diffusivity
    if(have_diffusivity) call add_diffusivity_face_cg(face, bc_type, t, t_bc, t_bc_2, detwei, matrix_addto, rhs_addto)
    
    ! Step 3: Insertion
    
    face_nodes = face_global_nodes(t, face)
    call addto(matrix, face_nodes, face_nodes, matrix_addto)
    call addto(rhs, face_nodes, rhs_addto)
    
  end subroutine assemble_advection_diffusion_face_cg
  
  subroutine add_advection_face_cg(face, bc_type, t, t_bc, velocity, grid_velocity, density, olddensity, nvfrac, detwei, normal, matrix_addto, rhs_addto)
    integer, intent(in) :: face
    integer, intent(in) :: bc_type
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: t_bc
    type(vector_field), intent(in) :: velocity
    type(vector_field), pointer :: grid_velocity
    type(scalar_field), intent(in) :: density
    type(scalar_field), intent(in) :: olddensity
    type(scalar_field), intent(in) :: nvfrac
    real, dimension(face_ngi(t, face)), intent(in) :: detwei
    real, dimension(mesh_dim(t), face_ngi(t, face)), intent(in) :: normal
    real, dimension(face_loc(t, face), face_loc(t, face)), intent(inout) :: matrix_addto
    real, dimension(face_loc(t, face)), intent(inout) :: rhs_addto
    
    real, dimension(velocity%dim, face_ngi(velocity, face)) :: velocity_at_quad
    real, dimension(face_loc(t, face), face_loc(t,face)) :: advection_mat
    type(element_type), pointer :: t_shape
    
    real, dimension(face_ngi(density, face)) :: density_at_quad
    
    assert(have_advection)
    assert(integrate_advection_by_parts)
    
    t_shape => face_shape(t, face)
    
    velocity_at_quad = face_val_at_quad(velocity, face)
    if(move_mesh) then
      velocity_at_quad = velocity_at_quad - face_val_at_quad(grid_velocity, face)
    end if
    select case(equation_type)
    case(FIELD_EQUATION_INTERNALENERGY)
      density_at_quad = density_theta*face_val_at_quad(density, face) &
                       +(1.0-density_theta)*face_val_at_quad(olddensity, face)
                       
      if(multiphase) then
         advection_mat = shape_shape(t_shape, t_shape, detwei * sum(velocity_at_quad * normal, 1) * density_at_quad * face_val_at_quad(nvfrac, face))
      else
         advection_mat = shape_shape(t_shape, t_shape, detwei * sum(velocity_at_quad * normal, 1) * density_at_quad)
      end if
    case default
      
      advection_mat = shape_shape(t_shape, t_shape, detwei * sum(velocity_at_quad * normal, 1))
      
    end select
    
    if(abs(dt_theta) > epsilon(0.0)) then
      if(bc_type == BC_TYPE_WEAKDIRICHLET) then
        rhs_addto = rhs_addto - theta * matmul(advection_mat, ele_val(t_bc, face) - face_val(t, face))
      else
        matrix_addto = matrix_addto + dt_theta * advection_mat
      end if
    end if
    
    rhs_addto = rhs_addto - matmul(advection_mat, face_val(t, face))

  end subroutine add_advection_face_cg
  
  subroutine add_diffusivity_face_cg(face, bc_type, t, t_bc, t_bc_2, detwei, matrix_addto, rhs_addto)
    integer, intent(in) :: face
    integer, intent(in) :: bc_type
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(in) :: t_bc
    type(scalar_field), intent(in) :: t_bc_2
    real, dimension(face_ngi(t, face)), intent(in) :: detwei
    real, dimension(face_loc(t, face), face_loc(t, face)), intent(inout) :: matrix_addto    
    real, dimension(face_loc(t, face)), intent(inout) :: rhs_addto
    
    real, dimension(face_loc(t, face), face_loc(t,face)) :: robin_mat
    type(element_type), pointer :: t_shape

    assert(have_diffusivity)

    t_shape => face_shape(t, face)

    if(bc_type == BC_TYPE_NEUMANN) then
      rhs_addto = rhs_addto + shape_rhs(t_shape, detwei * ele_val_at_quad(t_bc, face))
    else if(bc_type == BC_TYPE_ROBIN) then
      rhs_addto = rhs_addto + shape_rhs(t_shape, detwei * ele_val_at_quad(t_bc, face))
      robin_mat = shape_shape(t_shape, t_shape, detwei * ele_val_at_quad(t_bc_2, face))   
      if (abs(dt_theta) > epsilon(0.0)) then 
         matrix_addto = matrix_addto + dt_theta * robin_mat
      end if 
      ! this next term is due to solving the acceleration form of the equation
      rhs_addto = rhs_addto - matmul(robin_mat, face_val(t, face))      
    else if(bc_type == BC_TYPE_WEAKDIRICHLET) then
      ! Need to add stuff here once transform_to_physical can supply gradients
      ! on faces to ensure that weak bcs work
      FLExit("Weak Dirichlet boundary conditions with diffusivity are not supported by CG advection-diffusion")
    end if

  end subroutine add_diffusivity_face_cg
     
  subroutine solve_advection_diffusion_cg(t, delta_t, matrix, rhs, state, iterations_taken)
    type(scalar_field), intent(in) :: t
    type(scalar_field), intent(inout) :: delta_t
    type(csr_matrix), intent(in) :: matrix
    type(scalar_field), intent(in) :: rhs
    type(state_type), intent(in) :: state
    integer, intent(out), optional :: iterations_taken
    
    call petsc_solve(delta_t, matrix, rhs, state, option_path = t%option_path, &
                     iterations_taken = iterations_taken)
    
    ewrite_minmax(delta_t)
    
  end subroutine solve_advection_diffusion_cg
  
  subroutine apply_advection_diffusion_cg_change(t, delta_t, dt)
    type(scalar_field), intent(inout) :: t
    type(scalar_field), intent(in) :: delta_t
    real, intent(in) :: dt
    
    ewrite_minmax(t)
    
    call addto(t, delta_t, dt)
    
    ewrite_minmax(t)
    
  end subroutine apply_advection_diffusion_cg_change
    
  subroutine advection_diffusion_cg_check_options
    !!< Check CG advection-diffusion specific options
    
    character(len = FIELD_NAME_LEN) :: field_name, state_name, mesh_0, mesh_1
    character(len = OPTION_PATH_LEN) :: path
    integer :: i, j, stat
    real :: beta, l_theta
    
    if(option_count("/material_phase/scalar_field/prognostic/spatial_discretisation/continuous_galerkin") == 0) then
      ! Nothing to check
      return
    end if
    
    ewrite(2, *) "Checking CG advection-diffusion options"
            
    if(option_count("/material_phase/scalar_field::" // advdif_cg_rhs_name) > 0) then
      FLExit("The scalar field name " // advdif_cg_rhs_name // " is reserved")
    end if
    
    if(option_count("/material_phase/scalar_field::" // advdif_cg_delta_t_name) > 0) then
      FLExit("The scalar field name " // advdif_cg_delta_t_name // " is reserved")
    end if
    
    do i = 0, option_count("/material_phase") - 1
      path = "/material_phase[" // int2str(i) // "]"
      call get_option(trim(path) // "/name", state_name)
      
      do j = 0, option_count(trim(path) // "/scalar_field") - 1
        path = "/material_phase[" // int2str(i) // "]/scalar_field[" // int2str(j) // "]"
        call get_option(trim(path) // "/name", field_name)
        
        if(field_name /= "Pressure") then
        
          path = trim(path) // "/prognostic"
          
          if(have_option(trim(path) // "/spatial_discretisation/continuous_galerkin").and.&
             have_option(trim(path) // "/equation[0]")) then       
            call get_option(trim(path) // "/spatial_discretisation/conservative_advection", beta, stat)
            if(stat == SPUD_NO_ERROR) then
              if(beta < 0.0 .or. beta > 1.0) then
              
                call field_error(state_name, field_name, &
                  & "Conservative advection factor (beta) must be >= 0.0 and <= 1.0")
              end if
            else
              call field_error(state_name, field_name, &
                & "Conservative advection factor (beta) required")
            end if
            
            call get_option(trim(path) // "/temporal_discretisation/theta", l_theta, stat)
            if(stat == SPUD_NO_ERROR) then
              if(l_theta < 0. .or. l_theta > 1.0) then
                call field_error(state_name, field_name, &
                  &"Implicitness factor (theta) must be >= 0.0 and <= 1.0")
              end if
            else
              call field_error(state_name, field_name, &
                & "Implicitness factor (theta) required")
            end if
            if(have_option(trim(path) // "/spatial_discretisation/continuous_galerkin/mass_terms/exclude_mass_terms") .and. &
              & abs(l_theta - 1.0) > epsilon(0.0)) then
              call field_warning(state_name, field_name, &
                & "Implicitness factor (theta) should = 1.0 when excluding mass")
            end if
  
            if(have_option(trim(path) // "/spatial_discretisation/continuous_galerkin/stabilisation/streamline_upwind_petrov_galerkin") .and. &
              & have_option(trim(path) // "/spatial_discretisation/continuous_galerkin/advection_terms/integrate_advection_by_parts")) then
              call field_warning(state_name, field_name, &
                & "SUPG stabilisation should only be used with advection not integrated by parts")
            end if
  
            if(option_count(trim(path) // "/boundary_conditions/type::dirichlet/apply_weakly") > 0 &
              & .and. have_option(trim(path) // "/tensor_field::Diffusivity")) then
              call field_error(state_name, field_name, &
                & "Weak Dirichlet boundary conditions with diffusivity are not supported by CG advection-diffusion")
            end if
                 
            if (have_option(trim(path) // "/scalar_field::SinkingVelocity")) then
               call get_option(trim(complete_field_path(trim(path) // &
                    "/scalar_field::SinkingVelocity"))//"/mesh[0]/name", &
                    mesh_0, stat)
               if(stat == SPUD_NO_ERROR) then
                  call get_option(trim(complete_field_path("/material_phase[" // int2str(i) // &
                       "]/vector_field::Velocity")) // "/mesh[0]/name", mesh_1)
                  if(trim(mesh_0) /= trim(mesh_1)) then
                     call field_warning(state_name, field_name, &
                          & "SinkingVelocity is on a different mesh to the Velocity field. This could cause problems")
                  end if
               end if
            end if
            if(have_option(trim(path) // "/spatial_discretisation/continuous_galerkin/advection_terms/exclude_advection_terms")) then
              if(have_option(trim(path) // "/scalar_field::SinkingVelocity")) then
                call field_warning(state_name, field_name, &
                  & "SinkingVelocity set, but advection terms have been excluded - SinkingVelocity will have no effect")
              end if
            end if
  
            if(option_count(trim(path) // "/boundary_conditions/type::neumann") > 0 &
              & .and. .not. (have_option(trim(path) // "/tensor_field::Diffusivity") &
              & .or. have_option(trim(path) // "/subgridscale_parameterisation::k-epsilon") &
              & .or. have_option(trim(path) // "/subgridscale_parameterisation::GLS"))) then
                call field_warning(state_name, field_name, &
                & "Neumann boundary condition set, but have no diffusivity - boundary condition will not be applied")
            end if
          end if
        end if
      end do
    end do
    
    ewrite(2, *) "Finished checking CG advection-diffusion options"
    
  contains
  
    subroutine field_warning(state_name, field_name, msg)
      character(len = *), intent(in) :: state_name
      character(len = *), intent(in) :: field_name
      character(len = *), intent(in) :: msg
      
      ewrite(0, *) "Warning: For field " // trim(field_name) // " in state " // trim(state_name)
      ewrite(0, *) trim(msg)
    
    end subroutine field_warning
  
    subroutine field_error(state_name, field_name, msg)
      character(len = *), intent(in) :: state_name
      character(len = *), intent(in) :: field_name
      character(len = *), intent(in) :: msg
      
      ewrite(-1, *) "For field " // trim(field_name) // " in state " // trim(state_name)
      FLExit(trim(msg))
    
    end subroutine field_error
  
  end subroutine advection_diffusion_cg_check_options


  !2022-04-02-001S================= Max Min and Effective Mesh Sizes Computing !@#$% jxli Apr 2022
  subroutine average_mesh_size(state)

     type(state_type), intent(in)            :: state

     integer                                 :: stat, node_number, point_loc, i, if_count, dimD
     type(vector_field), pointer             :: positions_local
     type(vector_field)                      :: positions
     type(mesh_type), pointer                :: mesh
     integer, dimension(:), pointer          :: ele_number, face_number
     real, dimension(3,4)                    :: element_position_matrix_3D
     real, dimension(2,3)                    :: element_position_matrix
     real                                    :: distance, distance_min, distance_max, effective_distance

     type(scalar_field), pointer             :: effe_size_local, mesh_size_local, min_size_local, max_size_local

     real, dimension(2)                      :: domain_x, domain_y, domain_z

     ewrite(1,*) 'Start calculate average adaptive mesh size'

     effe_size_local => extract_scalar_field(state,'EffectiveMeshSize',stat=stat)
     !call allocate(effe_size, effe_size_local%mesh, 'RemappedEffectiveMeshSize')
     !call set(effe_size, 0.0)
     !call remap_field(effe_size_local, effe_size)

     mesh_size_local => extract_scalar_field(state,'MeshSize',stat=stat)
     !call allocate(mesh_size, mesh_size_local%mesh, 'RemappedMeshSize')
     !call set(mesh_size, 0.0)
     !call remap_field(mesh_size_local, mesh_size)

     min_size_local => extract_scalar_field(state,'MinMeshSize',stat=stat)
     !call allocate(min_size, min_size_local%mesh, 'RemappedMinMeshSize')
     !call set(min_size, 0.0)
     !call remap_field(min_size_local, min_size)

     max_size_local => extract_scalar_field(state,'MaxMeshSize',stat=stat)
     !call allocate(max_size, max_size_local%mesh, 'RemappedMaxMeshSize')
     !call set(max_size, 0.0)
     !call remap_field(max_size_local, max_size)

     call get_option("/geometry/dimension", dimD)

     if (dimD == 1) then
         print*, 'Mountain Wave Damping can be applied only for 2D currently.'
         stop
     else if (dimD == 2) then

         positions_local => extract_vector_field(state, 'Coordinate', stat=stat)
         call allocate(positions, positions_local%dim, positions_local%mesh, 'RemappedCoordinate')
         positions%val = 0.0
         call remap_field(positions_local, positions)

         mesh            => positions_local%mesh
         call add_nelist(mesh)

         distance_min = 1000000.0
         distance_max = 0.0
         effective_distance = 0.0
         if_count = 0

if (have_option("/material_phase/subgridscale_parameterisations/mesh_size_computing/domain_x/")) then
    call get_option("/material_phase/subgridscale_parameterisations/mesh_size_computing/domain_x/id_vector/", domain_x) 
!end if
!if (have_option("/material_phase/subgridscale_parameterisations/mesh_size_computing/domain_y/")) then
    call get_option("/material_phase/subgridscale_parameterisations/mesh_size_computing/domain_y/id_vector/", domain_y) 


         do node_number = 1, node_count(positions_local)
             distance = 0.0
                           !!< Return a pointer to a vector containing the element numbers of the elements containing node_number.
             ele_number => node_neigh(mesh, node_number)

             !face_number => ele_faces(mesh, ele_number(1)) ! Return a pointer to a vector containing the face numbers of the faces adjacent to ele_number.
             !print*, 'miao',face_val(positions, face_number(1)) ! all points z position in one face
             
             element_position_matrix = ele_val(positions, ele_number(1)) ! all position in one element   

if ( (element_position_matrix(1,1) >= domain_x(1) .and. element_position_matrix(1,1) <= domain_x(2) .and.  &
      element_position_matrix(2,1) >= domain_y(1) .and. element_position_matrix(2,1) <= domain_y(2) ) &
 .or.(element_position_matrix(1,2) >= domain_x(1) .and. element_position_matrix(1,2) <= domain_x(2) .and.  &
      element_position_matrix(2,2) >= domain_y(1) .and. element_position_matrix(2,2) <= domain_y(2) ) &
 .or.(element_position_matrix(1,3) >= domain_x(1) .and. element_position_matrix(1,3) <= domain_x(2) .and.  &
      element_position_matrix(2,3) >= domain_y(1) .and. element_position_matrix(2,3) <= domain_y(2) ) ) then !#100
             if_count = if_count + 1
             point_loc = 0
             do i = 1, 3
                 if ( ( (element_position_matrix(1,i) - positions%val(1,node_number))**2.0 &
                      + (element_position_matrix(2,i) - positions%val(2,node_number))**2.0 ) < 1.0e-03 ) then
                     point_loc = i
                 end if
             end do

             if (point_loc == 0) then
                 ewrite(1,*) 'Element Error'
                 stop
             else
                 do i = 1, 3
                     if ( point_loc .ne. i) then
                         distance = distance + sqrt( (element_position_matrix(1,i) - positions%val(1,node_number))**2.0 &
                                                   + (element_position_matrix(2,i) - positions%val(2,node_number))**2.0 )
                     end if
                 end do
                 distance = distance * 1.0 / 2.0
                 effective_distance = effective_distance + distance
             end if

             if (distance >= distance_max) distance_max = distance
             if (distance <= distance_min) distance_min = distance
             mesh_size_local%val(node_number) = distance  

             !if ( PT_local%val(node_number) >= deltaPT_max ) deltaPT_max = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 >= velocity_max**2 ) velocity_max = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
             !if ( PT_local%val(node_number) <= deltaPT_min ) deltaPT_min = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 <= velocity_min**2 ) velocity_min = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
end if !#100
         end do
         effective_distance = effective_distance / if_count

         do node_number = 1, node_count(positions_local)
!             ele_number => node_neigh(mesh, node_number)

             !face_number => ele_faces(mesh, ele_number(1)) ! Return a pointer to a vector containing the face numbers of the faces adjacent to ele_number.
             !print*, 'miao',face_val(positions, face_number(1)) ! all points z position in one face
             
!             element_position_matrix = ele_val(positions, ele_number(1)) ! all position in one element   
!if ( (element_position_matrix(1,1) >= domain_x(1) .and. element_position_matrix(1,1) <= domain_x(2) .and.  &
!      element_position_matrix(2,1) >= domain_y(1) .and. element_position_matrix(2,1) <= domain_y(2) ) &
! .or.(element_position_matrix(1,2) >= domain_x(1) .and. element_position_matrix(1,2) <= domain_x(2) .and.  &
!      element_position_matrix(2,2) >= domain_y(1) .and. element_position_matrix(2,2) <= domain_y(2) ) &
! .or.(element_position_matrix(1,3) >= domain_x(1) .and. element_position_matrix(1,3) <= domain_x(2) .and.  &
!      element_position_matrix(2,3) >= domain_y(1) .and. element_position_matrix(2,3) <= domain_y(2) ) ) then !#200
             max_size_local%val (node_number) = distance_max
             min_size_local%val (node_number) = distance_min
             effe_size_local%val(node_number) = effective_distance
             !viscosity = viscosity_ref * (PT_local%val(node_number)-deltaPT_min)/deltaPT_ref * effective_distance/mesh_size_ref * velocity_max/velocity_ref
             !viscosity = max(viscosity_ref, viscosity)
             !effe_size_local%val(node_number) = viscosity
!end if !#200
         end do
else
         do node_number = 1, node_count(positions_local)
             distance = 0.0
                           !!< Return a pointer to a vector containing the element numbers of the elements containing node_number.
             ele_number => node_neigh(mesh, node_number)

             !face_number => ele_faces(mesh, ele_number(1)) ! Return a pointer to a vector containing the face numbers of the faces adjacent to ele_number.
             !print*, 'miao',face_val(positions, face_number(1)) ! all points z position in one face
             
             element_position_matrix = ele_val(positions, ele_number(1)) ! all position in one element   

             if_count = if_count + 1
             point_loc = 0
             do i = 1, 3
                 if ( ( (element_position_matrix(1,i) - positions%val(1,node_number))**2.0 &
                      + (element_position_matrix(2,i) - positions%val(2,node_number))**2.0 ) < 1.0e-03 ) then
                     point_loc = i
                 end if
             end do

             if (point_loc == 0) then
                 ewrite(1,*) 'Element Error'
                 stop
             else
                 do i = 1, 3
                     if ( point_loc .ne. i) then
                         distance = distance + sqrt( (element_position_matrix(1,i) - positions%val(1,node_number))**2.0 &
                                                   + (element_position_matrix(2,i) - positions%val(2,node_number))**2.0 )
                     end if
                 end do
                 distance = distance * 1.0 / 2.0
                 effective_distance = effective_distance + distance
             end if

             if (distance >= distance_max) distance_max = distance
             if (distance <= distance_min) distance_min = distance
             mesh_size_local%val(node_number) = distance  

             !if ( PT_local%val(node_number) >= deltaPT_max ) deltaPT_max = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 >= velocity_max**2 ) velocity_max = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
             !if ( PT_local%val(node_number) <= deltaPT_min ) deltaPT_min = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 <= velocity_min**2 ) velocity_min = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
         end do

         effective_distance = effective_distance / if_count

         do node_number = 1, node_count(positions_local)
             max_size_local%val (node_number) = distance_max
             min_size_local%val (node_number) = distance_min
             effe_size_local%val(node_number) = effective_distance
             !viscosity = viscosity_ref * (PT_local%val(node_number)-deltaPT_min)/deltaPT_ref * effective_distance/mesh_size_ref * velocity_max/velocity_ref
             !viscosity = max(viscosity_ref, viscosity)
             !effe_size_local%val(node_number) = viscosity
         end do

end if








         call deallocate ( positions )

     else if (dimD == 3) then

         positions_local => extract_vector_field(state, 'Coordinate', stat=stat)
         call allocate(positions, positions_local%dim, positions_local%mesh, 'RemappedCoordinate')
         positions%val = 0.0
         call remap_field(positions_local, positions)

         mesh            => positions_local%mesh
         call add_nelist(mesh)

         distance_min = 1000000.0
         distance_max = 0.0
         effective_distance = 0.0
         if_count = 0

         do node_number = 1, node_count(positions_local)
             distance = 0.0
                           !!< Return a pointer to a vector containing the element numbers of the elements containing node_number.
             ele_number => node_neigh(mesh, node_number)

             !face_number => ele_faces(mesh, ele_number(1)) ! Return a pointer to a vector containing the face numbers of the faces adjacent to ele_number.
             !print*, 'miao',face_val(positions, face_number(1)) ! all points z position in one face
             
             element_position_matrix_3D = ele_val(positions, ele_number(1)) ! all position in one element   

             if_count = if_count + 1
             point_loc = 0
             do i = 1, 4
                 if ( ( (element_position_matrix_3D(1,i) - positions%val(1,node_number))**2.0 &
                      + (element_position_matrix_3D(2,i) - positions%val(2,node_number))**2.0 &
                      + (element_position_matrix_3D(3,i) - positions%val(3,node_number))**2.0 ) < 1.0e-03 ) then
                     point_loc = i
                 end if
             end do

             if (point_loc == 0) then
                 ewrite(1,*) 'Element Error'
                 stop
             else
                 do i = 1, 4
                     if ( point_loc .ne. i) then
                         distance = distance + sqrt( (element_position_matrix_3D(1,i) - positions%val(1,node_number))**2.0 &
                                                   + (element_position_matrix_3D(2,i) - positions%val(2,node_number))**2.0 &
                                                   + (element_position_matrix_3D(3,i) - positions%val(3,node_number))**2.0 )
                     end if
                 end do
                 distance = distance * 1.0 / 3.0
                 effective_distance = effective_distance + distance
             end if

             if (distance >= distance_max) distance_max = distance
             if (distance <= distance_min) distance_min = distance
             mesh_size_local%val(node_number) = distance  

             !if ( PT_local%val(node_number) >= deltaPT_max ) deltaPT_max = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 >= velocity_max**2 ) velocity_max = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
             !if ( PT_local%val(node_number) <= deltaPT_min ) deltaPT_min = PT_local%val(node_number)
             !if ( velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2 <= velocity_min**2 ) velocity_min = sqrt(velocity_local%val(1, node_number)**2+velocity_local%val(2, node_number)**2)
         end do

         effective_distance = effective_distance / if_count

         do node_number = 1, node_count(positions_local)
             max_size_local%val (node_number) = distance_max
             min_size_local%val (node_number) = distance_min
             effe_size_local%val(node_number) = effective_distance
             !viscosity = viscosity_ref * (PT_local%val(node_number)-deltaPT_min)/deltaPT_ref * effective_distance/mesh_size_ref * velocity_max/velocity_ref
             !viscosity = max(viscosity_ref, viscosity)
             !effe_size_local%val(node_number) = viscosity
         end do

         call deallocate ( positions )

     end if

     ewrite(1,*) 'Finished calculate average adaptive mesh size'

  end subroutine average_mesh_size
  !2022-04-02-001E================= Max Min and Effective Mesh Sizes Computing !@#$% jxli Apr 2022

end module advection_diffusion_cg
