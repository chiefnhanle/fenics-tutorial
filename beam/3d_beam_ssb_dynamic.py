# -*- coding: utf-8 -*-
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter
import ufl

print("FEniCSx version:", dolfinx.__version__)
comm = MPI.COMM_WORLD

# ------------------------------------------------------------
# 0) Output directory
# ------------------------------------------------------------
WIN_DIR = r"C:\Users\Nhan Le\Downloads\deflection"
WSL_DIR = "/mnt/c/Users/Nhan Le/Downloads/deflection"
outdir = WSL_DIR if os.path.exists("/mnt/c") else WIN_DIR

if comm.rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

bp_vol = os.path.join(outdir, "ssb_sine_pointload_volume.bp")
bp_top = os.path.join(outdir, "ssb_sine_pointload_load_top.bp")

# ------------------------------------------------------------
# 1) Geometry
# ------------------------------------------------------------
L, H, W = 5.0, 0.3, 0.1
nx, ny, nz = 60, 20, 8

domain = mesh.create_box(
    comm,
    [np.array([0, 0, 0]), np.array([L, H, W])],
    [nx, ny, nz],
    cell_type=mesh.CellType.tetrahedron,
)

gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1

# ------------------------------------------------------------
# 2) Function space
# ------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

def locate_dofs_sub_geometrical(Vsub, Vsub_collapsed, marker):
    dofs = fem.locate_dofs_geometrical((Vsub, Vsub_collapsed), marker)
    if isinstance(dofs, (list, tuple)):
        dofs = dofs[0]
    return np.asarray(dofs, dtype=np.int32)

# ------------------------------------------------------------
# 3) Simply supported BCs
# ------------------------------------------------------------
def x0_face(x): return np.isclose(x[0], 0.0)
def xL_face(x): return np.isclose(x[0], L)

Vy_c, _ = V.sub(1).collapse()
Vz_c, _ = V.sub(2).collapse()

zero = fem.Constant(domain, PETSc.ScalarType(0.0))

dofs_uy0 = locate_dofs_sub_geometrical(V.sub(1), Vy_c, x0_face)
dofs_uz0 = locate_dofs_sub_geometrical(V.sub(2), Vz_c, x0_face)
dofs_uyL = locate_dofs_sub_geometrical(V.sub(1), Vy_c, xL_face)

bc_uy0 = fem.dirichletbc(zero, dofs_uy0, V.sub(1))
bc_uz0 = fem.dirichletbc(zero, dofs_uz0, V.sub(2))
bc_uyL = fem.dirichletbc(zero, dofs_uyL, V.sub(1))

def vtx_000(x):
    return np.isclose(x[0],0)&np.isclose(x[1],0)&np.isclose(x[2],0)

def vtx_L00(x):
    return np.isclose(x[0],L)&np.isclose(x[1],0)&np.isclose(x[2],0)

verts_000 = mesh.locate_entities_boundary(domain,0,vtx_000)
verts_L00 = mesh.locate_entities_boundary(domain,0,vtx_L00)

dofs_ux_000 = fem.locate_dofs_topological(V.sub(0),0,verts_000)
dofs_uz_000 = fem.locate_dofs_topological(V.sub(2),0,verts_000)
dofs_uz_L00 = fem.locate_dofs_topological(V.sub(2),0,verts_L00)

bcs = [bc_uy0, bc_uz0, bc_uyL]

if len(dofs_ux_000):
    bcs.append(fem.dirichletbc(zero,np.asarray(dofs_ux_000,np.int32),V.sub(0)))
if len(dofs_uz_000):
    bcs.append(fem.dirichletbc(zero,np.asarray(dofs_uz_000,np.int32),V.sub(2)))
if len(dofs_uz_L00):
    bcs.append(fem.dirichletbc(zero,np.asarray(dofs_uz_L00,np.int32),V.sub(2)))

# ------------------------------------------------------------
# 4) Sinusoidal patch load
# ------------------------------------------------------------
x_center, z_center = 0.5*L, 0.5*W
r_patch = 0.06

def top_face(x): return np.isclose(x[1],H)

top_facets_all = mesh.locate_entities_boundary(domain,fdim,top_face)
top_mid = mesh.compute_midpoints(domain,fdim,top_facets_all)

dx = top_mid[:,0]-x_center
dz = top_mid[:,2]-z_center
patch_mask = (dx*dx+dz*dz)<=r_patch*r_patch
patch_facets = top_facets_all[patch_mask]

patch_tag = 11
facet_tags = mesh.meshtags(
    domain,fdim,patch_facets,
    np.full(len(patch_facets),patch_tag,np.int32)
)

ds = ufl.Measure("ds",domain=domain,subdomain_data=facet_tags)

one = fem.Constant(domain,PETSc.ScalarType(1.0))
A_local = fem.assemble_scalar(fem.form(one*ds(patch_tag)))
A_patch = comm.allreduce(A_local,op=MPI.SUM)

if comm.rank==0:
    print("Patch facets:",len(patch_facets)," patch area =",A_patch)

F0, freq = 1000.0, 1.0
omega = 2*np.pi*freq

tload = fem.Constant(domain,PETSc.ScalarType((0.0,0.0,0.0)))

# ------------------------------------------------------------
# Create top patch submesh (ROBUST FIX)
# ------------------------------------------------------------
sub_out = mesh.create_submesh(domain,fdim,patch_facets)
if isinstance(sub_out,tuple):
    top_mesh = sub_out[0]
else:
    top_mesh = sub_out

V_top = fem.functionspace(top_mesh,("Lagrange",1,(top_mesh.geometry.dim,)))
load_top = fem.Function(V_top)
load_top.name="load_vector_top"

# ------------------------------------------------------------
# 5) Elasticity
# ------------------------------------------------------------
E,nu = 210e9,0.30
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def eps(w): return ufl.sym(ufl.grad(w))
def sigma(w):
    return 2*mu*eps(w)+lmbda*ufl.tr(eps(w))*ufl.Identity(gdim)

a = ufl.inner(sigma(u),eps(v))*ufl.dx
Lform = ufl.dot(tload,v)*ds(patch_tag)

# ------------------------------------------------------------
# 6) Time loop
# ------------------------------------------------------------
t_end,dt = 2.0,0.02
times = np.arange(0,t_end+0.5*dt,dt)

uh = fem.Function(V); uh.name="displacement"

V_vm = fem.functionspace(domain,("Lagrange",1))
vm = fem.Function(V_vm); vm.name="von_mises"

vm_trial = ufl.TrialFunction(V_vm)
vm_test = ufl.TestFunction(V_vm)

sig_uh = sigma(uh)
s_dev = sig_uh-(1/3)*ufl.tr(sig_uh)*ufl.Identity(gdim)
von_mises_expr = ufl.sqrt(3/2*ufl.inner(s_dev,s_dev))

with VTXWriter(domain.comm,bp_vol,[uh,vm],engine="BP4") as vtx_vol,\
     VTXWriter(top_mesh.comm,bp_top,[load_top],engine="BP4") as vtx_top:

    for ti in times:

        F_t = F0*np.sin(omega*ti)
        p_t = (F_t/A_patch) if A_patch>0 else 0.0
        tload.value[:] = (0.0,-p_t,0.0)

        prob = LinearProblem(
            a,Lform,bcs=bcs,
            petsc_options_prefix="elas3d_ss_sine_",
            petsc_options={
                "ksp_type":"cg",
                "pc_type":"hypre",
                "ksp_rtol":1e-10,
                "ksp_max_it":2000
            }
        )
        uh_sol = prob.solve()
        uh.x.array[:] = uh_sol.x.array
        uh.x.scatter_forward()

        prob_vm = LinearProblem(
            ufl.inner(vm_trial,vm_test)*ufl.dx,
            von_mises_expr*vm_test*ufl.dx,
            petsc_options={"ksp_type":"cg","pc_type":"jacobi"}
        )
        vm_sol = prob_vm.solve()
        vm.x.array[:] = vm_sol.x.array
        vm.x.scatter_forward()

        def load_interp(x):
            vals = np.zeros((top_mesh.geometry.dim,x.shape[1]))
            vals[1] = -p_t
            return vals

        load_top.interpolate(load_interp)
        load_top.x.scatter_forward()

        vtx_vol.write(ti)
        vtx_top.write(ti)

if comm.rank==0:
    print("Wrote:",bp_vol)
    print("Wrote:",bp_top)
    print("Open in ParaView and use the TIME slider.")
