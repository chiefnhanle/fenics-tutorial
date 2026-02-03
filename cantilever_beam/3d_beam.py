import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import VTXWriter  # <-- KEY CHANGE
import ufl

print("FEniCSx version:", dolfinx.__version__)
comm = MPI.COMM_WORLD

# ------------------------------------------------------------
# 1) 3D Beam geometry + mesh
# ------------------------------------------------------------
L = 1.0
H = 0.2
W = 0.1
nx, ny, nz = 60, 12, 8

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
# 3) Clamped boundary at x=0
# ------------------------------------------------------------
u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0

def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
bc = fem.dirichletbc(u_bc, left_dofs)

# ------------------------------------------------------------
# 4) Traction load on x=L face
# ------------------------------------------------------------
def right_boundary(x):
    return np.isclose(x[0], L)

right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
right_tag = 1
facet_values = np.full(len(right_facets), right_tag, dtype=np.int32)
facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_values)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

T = 1.0e-2
t = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -T)))  # down in -z

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
Lform = ufl.dot(t, v) * ds(right_tag)

# ------------------------------------------------------------
# 6) Solve
# ------------------------------------------------------------
problem = LinearProblem(
    a, Lform,
    bcs=[bc],
    petsc_options_prefix="elas3d_",
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
# 7) Von Mises (compute + project to CG1 scalar)
# ------------------------------------------------------------
sig_uh = sigma(uh)
s_dev = sig_uh - (1.0 / 3.0) * ufl.tr(sig_uh) * ufl.Identity(gdim)
von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))

V_vm = fem.functionspace(domain, ("Lagrange", 1))
vm = fem.Function(V_vm)
vm.name = "von_mises"

vm_trial = ufl.TrialFunction(V_vm)
vm_test = ufl.TestFunction(V_vm)

problem_vm = LinearProblem(
    ufl.inner(vm_trial, vm_test) * ufl.dx,
    ufl.inner(von_mises_expr, vm_test) * ufl.dx,
    petsc_options_prefix="vm_",
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
)
vm = problem_vm.solve()
vm.name = "von_mises"
vm.x.scatter_forward()

# ------------------------------------------------------------
# 8) Export to ParaView using VTX (ADIOS2) -- fixes "partial"
# ------------------------------------------------------------
# Produces a folder/file: cantilever_3d.bp  (open directly in ParaView)
with VTXWriter(domain.comm, "cantilever_3d.bp", [uh, vm], engine="BP4") as vtx:
    vtx.write(0.0)

if comm.rank == 0:
    print("Wrote: cantilever_3d.bp (open this in ParaView)")
    print("Fields: displacement (vector), von_mises (scalar)")

# ------------------------------------------------------------
# 9) Optional: tip displacement at (L, H/2, W/2)
# ------------------------------------------------------------
points = np.array([[L, 0.5 * H, 0.5 * W]], dtype=np.float64)
tree = bb_tree(domain, domain.topology.dim)
candidates = compute_collisions_points(tree, points)
cells = compute_colliding_cells(domain, candidates, points)

local_val = None
if len(cells.links(0)) > 0:
    cell = cells.links(0)[0]
    u_tip = uh.eval(points[0], cell)
    local_val = np.array([float(u_tip[0]), float(u_tip[1]), float(u_tip[2])], dtype=np.float64)
else:
    local_val = np.array([0.0, 0.0, 0.0], dtype=np.float64)

u_tip_global = np.zeros(3, dtype=np.float64)
comm.Reduce(local_val, u_tip_global, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("Tip displacement at (L, H/2, W/2): ux =", u_tip_global[0],
          " uy =", u_tip_global[1], " uz =", u_tip_global[2])
