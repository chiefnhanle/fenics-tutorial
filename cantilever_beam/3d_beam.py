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
# 1) 3D Beam geometry + mesh (block)
# ------------------------------------------------------------
L = 1.0     # length (x)
H = 0.2     # height (y)
W = 0.1     # thickness (z)  <-- NEW
nx = 60
ny = 12
nz = 8      # <-- NEW

domain = mesh.create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
    [nx, ny, nz],
    cell_type=mesh.CellType.tetrahedron,
)

gdim = domain.geometry.dim  # 3
tdim = domain.topology.dim
fdim = tdim - 1

# ------------------------------------------------------------
# 2) Vector function space for displacement u = (ux, uy, uz)
# ------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# ------------------------------------------------------------
# 3) Clamped boundary at x = 0: u = (0, 0, 0)
# ------------------------------------------------------------
u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0

def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
bc = fem.dirichletbc(u_bc, left_dofs)

# ------------------------------------------------------------
# 4) Traction load on right boundary x = L
#    Apply uniform traction on end face (x=L).
# ------------------------------------------------------------
def right_boundary(x):
    return np.isclose(x[0], L)

right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)

right_tag = 1
facet_values = np.full(len(right_facets), right_tag, dtype=np.int32)
facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_values)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

T = 1.0e-2  # traction magnitude (3D units: force/area)
# Choose direction:
#   - If you want "down" as negative y, use (0, -T, 0)
#   - If you want "down" as negative z, use (0, 0, -T)
t = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -T)))

# ------------------------------------------------------------
# 5) Linear elasticity (small strain, isotropic)
#    (This is the standard 3D Hooke law; your 2D plane strain
#     special-casing is no longer needed.)
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

# ------------------------------------------------------------
# 7) Export to ParaView (XDMF)
# ------------------------------------------------------------
with XDMFFile(domain.comm, "cantilever_3d.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

if comm.rank == 0:
    print("Wrote: cantilever_3d.xdmf (open in ParaView)")

# ------------------------------------------------------------
# 8) Optional: print tip displacement at (L, H/2, W/2)
# ------------------------------------------------------------
points = np.array([[L, 0.5 * H, 0.5 * W]], dtype=np.float64)

tree = bb_tree(domain, domain.topology.dim)
candidates = compute_collisions_points(tree, points)
cells = compute_colliding_cells(domain, candidates, points)

local_val = None
if len(cells.links(0)) > 0:
    cell = cells.links(0)[0]
    u_tip = uh.eval(points[0], cell)  # [ux, uy, uz]
    local_val = np.array([float(u_tip[0]), float(u_tip[1]), float(u_tip[2])], dtype=np.float64)

if local_val is None:
    local_val = np.array([0.0, 0.0, 0.0], dtype=np.float64)

u_tip_global = np.zeros(3, dtype=np.float64)
comm.Reduce(local_val, u_tip_global, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("Tip displacement at (L, H/2, W/2): ux =", u_tip_global[0],
          " uy =", u_tip_global[1], " uz =", u_tip_global[2])
