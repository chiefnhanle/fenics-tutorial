import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

print("FEniCSx version:", dolfinx.__version__)

comm = MPI.COMM_WORLD

# 1) Mesh
domain = mesh.create_unit_square(comm, 8, 8)

# 2) Function space
V = fem.functionspace(domain, ("Lagrange", 1))

# 3) Dirichlet BC u=0 on all boundaries
u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0

def on_boundary(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )

dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(u_bc, dofs)

# 4) Variational problem: -Δu = 1
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(1.0))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 5) Solve (your dolfinx requires petsc_options_prefix)
problem = LinearProblem(
    a, L,
    bcs=[bc],
    petsc_options_prefix="poisson_",
    petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10
    },
)
uh = problem.solve()

# 6) Evaluate at center (cell-aware + MPI-safe)

# IMPORTANT: dolfinx expects ndarray shape (N, 3), not a Python list
points = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)

tree = bb_tree(domain, domain.topology.dim)
candidates = compute_collisions_points(tree, points)
cells = compute_colliding_cells(domain, candidates, points)

local_value = None
if len(cells.links(0)) > 0:
    cell = cells.links(0)[0]
    # eval expects a single point of shape (3,)
    local_value = float(uh.eval(points[0], cell)[0])

value = comm.reduce(local_value if local_value is not None else 0.0, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("u(0.5, 0.5) =", value)
    assert np.isfinite(value)
    assert value > 0.0
    print("SANITY CHECK PASSED")
