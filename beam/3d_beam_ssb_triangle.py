# -*- coding: utf-8 -*-
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import VTXWriter
import ufl

print("FEniCSx version:", dolfinx.__version__)
comm = MPI.COMM_WORLD

# ------------------------------------------------------------
# 0) Output directory: Windows path via WSL mount
# ------------------------------------------------------------
WIN_DIR = r"C:\Users\Nhan Le\Downloads\deflection"
WSL_DIR = "/mnt/c/Users/Nhan Le/Downloads/deflection"
outdir = WSL_DIR if os.path.exists("/mnt/c") else WIN_DIR

if comm.rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

bp_vol = os.path.join(outdir, "triangle_loaded_3d.bp")
bp_top = os.path.join(outdir, "top_load_surface.bp")

# ------------------------------------------------------------
# 1) 3D Beam geometry + mesh
# ------------------------------------------------------------
L = 5.0
H = 0.3
W = 0.1

nx, ny, nz = 60, 20, 8

domain = mesh.create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
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
# 3) Simply supported BCs + remove rigid modes
# ------------------------------------------------------------
def x0_face(x):
    return np.isclose(x[0], 0.0)

def xL_face(x):
    return np.isclose(x[0], L)

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
    return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0)

def vtx_L00(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0)

verts_000 = mesh.locate_entities_boundary(domain, 0, vtx_000)
verts_L00 = mesh.locate_entities_boundary(domain, 0, vtx_L00)

dofs_ux_000 = fem.locate_dofs_topological(V.sub(0), 0, verts_000)
dofs_uz_000 = fem.locate_dofs_topological(V.sub(2), 0, verts_000)
dofs_uz_L00 = fem.locate_dofs_topological(V.sub(2), 0, verts_L00)

bcs = [bc_uy0, bc_uz0, bc_uyL]
if len(dofs_ux_000):
    bcs.append(fem.dirichletbc(zero, np.asarray(dofs_ux_000, dtype=np.int32), V.sub(0)))
if len(dofs_uz_000):
    bcs.append(fem.dirichletbc(zero, np.asarray(dofs_uz_000, dtype=np.int32), V.sub(2)))
if len(dofs_uz_L00):
    bcs.append(fem.dirichletbc(zero, np.asarray(dofs_uz_L00, dtype=np.int32), V.sub(2)))

# ------------------------------------------------------------
# 4) TRIANGULAR traction on TOP face y = H
#    q(x) = qmax * x/L, downward in -y
# ------------------------------------------------------------
def top_face(x):
    return np.isclose(x[1], H)

top_facets = mesh.locate_entities_boundary(domain, fdim, top_face)
top_tag = 1
facet_tags = mesh.meshtags(
    domain, fdim, top_facets, np.full(len(top_facets), top_tag, dtype=np.int32)
)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

qmax = 1.0e-2
X = ufl.SpatialCoordinate(domain)
q = qmax * (X[0] / L)
t = ufl.as_vector((0.0, -q, 0.0))

# ------------------------------------------------------------
# 4b) Volume load field (for debugging; arrows everywhere)
# ------------------------------------------------------------
V_load = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
load_vec = fem.Function(V_load)
load_vec.name = "load_vector_volume"

def load_interp(x):
    vals = np.zeros((gdim, x.shape[1]), dtype=np.float64)
    vals[1] = -qmax * (x[0] / L)
    return vals

load_vec.interpolate(load_interp)
load_vec.x.scatter_forward()

# ------------------------------------------------------------
# 4c) Surface-only load field (this is what looks "triangular")
#     Build a top-surface mesh and put vectors on it
# ------------------------------------------------------------
top_mesh, top_entity_map = mesh.create_submesh(domain, fdim, top_facets)

V_top = fem.functionspace(top_mesh, ("Lagrange", 1, (top_mesh.geometry.dim,)))
load_top = fem.Function(V_top)
load_top.name = "load_vector_top"

# Interpolate on the top mesh (its coordinates are in the same physical space)
def load_top_interp(x):
    vals = np.zeros((top_mesh.geometry.dim, x.shape[1]), dtype=np.float64)
    vals[1] = -qmax * (x[0] / L)   # still varies with x
    return vals

load_top.interpolate(load_top_interp)
load_top.x.scatter_forward()

# ------------------------------------------------------------
# 5) Linear elasticity
# ------------------------------------------------------------
E = 210e9
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def eps(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim)

a = ufl.inner(sigma(u), eps(v)) * ufl.dx
Lform = ufl.dot(t, v) * ds(top_tag)

# ------------------------------------------------------------
# 6) Solve
# ------------------------------------------------------------
problem = LinearProblem(
    a, Lform,
    bcs=bcs,
    petsc_options_prefix="elas3d_ss_tri_",
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
        "ksp_max_it": 2000,
        "ksp_monitor": None,
        "ksp_converged_reason": None,
    },
)
uh = problem.solve()
uh.name = "displacement"
uh.x.scatter_forward()

# ------------------------------------------------------------
# 7) Von Mises (project to CG1 scalar)
# ------------------------------------------------------------
sig_uh = sigma(uh)
s_dev = sig_uh - (1.0 / 3.0) * ufl.tr(sig_uh) * ufl.Identity(gdim)
von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))

V_vm = fem.functionspace(domain, ("Lagrange", 1))
vm_trial = ufl.TrialFunction(V_vm)
vm_test = ufl.TestFunction(V_vm)

problem_vm = LinearProblem(
    ufl.inner(vm_trial, vm_test) * ufl.dx,
    von_mises_expr * vm_test * ufl.dx,
    petsc_options_prefix="vm_",
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1.0e-10},
)
vm = problem_vm.solve()
vm.name = "von_mises"
vm.x.scatter_forward()

# ------------------------------------------------------------
# 8) Export
# ------------------------------------------------------------
# Volume (displacement + von Mises + volume load debug)
with VTXWriter(domain.comm, bp_vol, [uh, vm, load_vec], engine="BP4") as vtx:
    vtx.write(0.0)

# Top surface (load only) -> this is the one that visually looks like a triangular load
with VTXWriter(top_mesh.comm, bp_top, [load_top], engine="BP4") as vtx:
    vtx.write(0.0)

if comm.rank == 0:
    print("Wrote volume:", bp_vol)
    print("Wrote top surface load:", bp_top)
    print("ParaView: open top_load_surface.bp, then Glyph -> load_vector_top")

# ------------------------------------------------------------
# 9) Optional: midspan displacement
# ------------------------------------------------------------
points = np.array([[0.5 * L, 0.5 * H, 0.5 * W]], dtype=np.float64)
tree = bb_tree(domain, domain.topology.dim)
candidates = compute_collisions_points(tree, points)
cells = compute_colliding_cells(domain, candidates, points)

if len(cells.links(0)) > 0:
    cell = cells.links(0)[0]
    u_mid = uh.eval(points[0], cell)
    local_val = np.array([float(u_mid[0]), float(u_mid[1]), float(u_mid[2])], dtype=np.float64)
else:
    local_val = np.array([0.0, 0.0, 0.0], dtype=np.float64)

u_global = np.zeros(3, dtype=np.float64)
comm.Reduce(local_val, u_global, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("Midspan displacement (L/2,H/2,W/2): ux =", u_global[0],
          " uy =", u_global[1], " uz =", u_global[2])
