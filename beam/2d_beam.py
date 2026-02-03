import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import XDMFFile
import ufl

print("FEniCSx version:", dolfinx.__version__)
comm = MPI.COMM_WORLD

# ------------------------------------------------------------
# OUTPUT: write to Windows Downloads\deflection via WSL mount
# ------------------------------------------------------------
WIN_OUTDIR = r"C:\Users\Nhan Le\Downloads\deflection"
WSL_OUTDIR = "/mnt/c/Users/Nhan Le/Downloads/deflection"  # WSL-visible path

# If you might also run this natively on Windows (not WSL), pick path accordingly:
outdir = WSL_OUTDIR if os.path.exists("/mnt/c") else WIN_OUTDIR

if comm.rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

xdmf_path = os.path.join(outdir, "cantilever.xdmf")

# ------------------------------------------------------------
# 1) Beam geometry + mesh
# ------------------------------------------------------------
L = 0.3    # length
H = 1   # height
nx = 60
ny = 12

domain = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([L, H])],
    [nx, ny],
    cell_type=mesh.CellType.triangle,
)

gdim = domain.geometry.dim  # 2
tdim = domain.topology.dim
fdim = tdim - 1

# ------------------------------------------------------------
# 2) Vector function space for displacement u = (ux, uy)
# ------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# ------------------------------------------------------------
# 3) Clamped boundary at x = 0: u = (0, 0)
# ------------------------------------------------------------
u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0

def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
bc = fem.dirichletbc(u_bc, left_dofs)

# ------------------------------------------------------------
# 4) Traction load on right boundary x = L
# ------------------------------------------------------------
def right_boundary(x):
    return np.isclose(x[0], L)

right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)

right_tag = 1
facet_values = np.full(len(right_facets), right_tag, dtype=np.int32)
facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_values)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

T = 1.0e-2  # downward line traction magnitude (2D units)
t = fem.Constant(domain, PETSc.ScalarType((0.0, -T)))

# ------------------------------------------------------------
# 5) Linear elasticity (plane strain form)
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
    petsc_options_prefix="elas_",
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
    },
)
uh = problem.solve()
uh.name = "displacement"

# ------------------------------------------------------------
# 7) Export to ParaView (XDMF) in Windows folder via WSL mount
# ------------------------------------------------------------
with XDMFFile(domain.comm, xdmf_path, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

if comm.rank == 0:
    # Note: .h5 will be created next to the .xdmf with the same basename
    print("Wrote:", xdmf_path, "(open this in ParaView on Windows)")

# ------------------------------------------------------------
# 8) Optional: print tip displacement at (L, H/2)
# ------------------------------------------------------------
points = np.array([[L, 0.5 * H, 0.0]], dtype=np.float64)

tree = bb_tree(domain, domain.topology.dim)
candidates = compute_collisions_points(tree, points)
cells = compute_colliding_cells(domain, candidates, points)

local_val = None
if len(cells.links(0)) > 0:
    cell = cells.links(0)[0]
    u_tip = uh.eval(points[0], cell)  # [ux, uy]
    local_val = np.array([float(u_tip[0]), float(u_tip[1])], dtype=np.float64)

if local_val is None:
    local_val = np.array([0.0, 0.0], dtype=np.float64)

u_tip_global = np.zeros(2, dtype=np.float64)
comm.Reduce(local_val, u_tip_global, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("Tip displacement at (L, H/2): ux =", u_tip_global[0], " uy =", u_tip_global[1])
