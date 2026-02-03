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

bp_path = os.path.join(outdir, "simply_supported_3d.bp")  # open this in ParaView

# ------------------------------------------------------------
# 1) 3D Beam geometry + mesh
# ------------------------------------------------------------
L = 5.0    # beam length (x)
H = 0.3    # depth / vertical (y)
W = 0.1    # width (z)

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
# 2) Vector function space for displacement
# ------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# ------------------------------------------------------------
# Helper: locate dofs on subspaces robustly (handles list-of-arrays)
# ------------------------------------------------------------
def locate_dofs_sub_geometrical(Vsub, Vsub_collapsed, marker):
    """
    Returns parent-space dofs for a subspace using geometric marker.
    Works across DOLFINx builds where locate_dofs_geometrical returns
    either ndarray[int32] or list[ndarray[int32]].
    """
    dofs = fem.locate_dofs_geometrical((Vsub, Vsub_collapsed), marker)
    if isinstance(dofs, (list, tuple)):
        dofs = dofs[0]
    return np.asarray(dofs, dtype=np.int32)

# ------------------------------------------------------------
# 3) Simply supported BCs (pin + roller + kill rigid modes)
#    Pin at x=0:    uy = 0, uz = 0
#    Roller at x=L: uy = 0
#    Kill rigid modes:
#       ux = 0 at ONE corner vertex (0,0,0)   (removes x-translation)
#       uz = 0 at ONE corner vertex (0,0,0)   (removes z-translation)
# ------------------------------------------------------------
def x0_face(x):
    return np.isclose(x[0], 0.0)

def xL_face(x):
    return np.isclose(x[0], L)

# collapse component subspaces (required for geometric dof location)
Vy_c, _ = V.sub(1).collapse()
Vz_c, _ = V.sub(2).collapse()

zero = fem.Constant(domain, PETSc.ScalarType(0.0))

# Face constraints
dofs_uy0 = locate_dofs_sub_geometrical(V.sub(1), Vy_c, x0_face)
dofs_uz0 = locate_dofs_sub_geometrical(V.sub(2), Vz_c, x0_face)
dofs_uyL = locate_dofs_sub_geometrical(V.sub(1), Vy_c, xL_face)

bc_uy0 = fem.dirichletbc(zero, dofs_uy0, V.sub(1))
bc_uz0 = fem.dirichletbc(zero, dofs_uz0, V.sub(2))
bc_uyL = fem.dirichletbc(zero, dofs_uyL, V.sub(1))

# Single-vertex constraints (robust, topological)
def corner_vertex(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0)

corner_verts = mesh.locate_entities_boundary(domain, 0, corner_vertex)

dofs_ux_corner = fem.locate_dofs_topological(V.sub(0), 0, corner_verts)
dofs_uz_corner = fem.locate_dofs_topological(V.sub(2), 0, corner_verts)

# Some MPI ranks may not own that vertex -> skip empty dof lists
bcs = [bc_uy0, bc_uz0, bc_uyL]
if len(dofs_ux_corner):
    bcs.append(fem.dirichletbc(zero, np.asarray(dofs_ux_corner, dtype=np.int32), V.sub(0)))
if len(dofs_uz_corner):
    bcs.append(fem.dirichletbc(zero, np.asarray(dofs_uz_corner, dtype=np.int32), V.sub(2)))

# ------------------------------------------------------------
# 4) Load: uniform traction on TOP face y = H (causes bending)
# ------------------------------------------------------------
def top_face(x):
    return np.isclose(x[1], H)

top_facets = mesh.locate_entities_boundary(domain, fdim, top_face)
top_tag = 1
facet_values = np.full(len(top_facets), top_tag, dtype=np.int32)
facet_tags = mesh.meshtags(domain, fdim, top_facets, facet_values)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

q = 1.0e-2  # traction magnitude
t = fem.Constant(domain, PETSc.ScalarType((0.0, -q, 0.0)))  # downward in -y

# ------------------------------------------------------------
# 5) 3D linear elasticity
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
    petsc_options_prefix="elas3d_ss_",
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
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
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
)
vm = problem_vm.solve()
vm.name = "von_mises"
vm.x.scatter_forward()

# ------------------------------------------------------------
# 8) Export to ParaView using VTX (ADIOS2)
# ------------------------------------------------------------
with VTXWriter(domain.comm, bp_path, [uh, vm], engine="BP4") as vtx:
    vtx.write(0.0)

if comm.rank == 0:
    print("Wrote:", bp_path)
    print("Fields: displacement (vector), von_mises (scalar)")

# ------------------------------------------------------------
# 9) Optional: midspan displacement at (L/2, H/2, W/2)
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
    print("Midspan displacement at (L/2, H/2, W/2): ux =", u_global[0],
          " uy =", u_global[1], " uz =", u_global[2])
