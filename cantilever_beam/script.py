"""
ParaView-clean dolfinx writer (NO MultiBlock, NO "(partial)", NO duplicate undeformed beam)

Key idea:
- DO NOT use XDMF for ParaView if you want to avoid partitioned/multiblock artefacts.
- Use dolfinx.io.VTKFile (.pvd/.vtu). ParaView loads this as a single dataset and WarpByVector behaves normally.

This script:
- Solves 3D linear elasticity cantilever under self-weight
- Writes:
    1) displacement.pvd  (single dataset in ParaView)
    2) von_mises.pvd     (single dataset in ParaView)

Open displacement.pvd in ParaView → WarpByVector → choose "Displacement Vector" → no duplicates, no partial.
"""

from mpi4py import MPI
import numpy as np

import ufl
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


CANTILEVER_LENGTH = 1.0
CANTILEVER_WIDTH = 0.2

N_POINTS_LENGTH = 10
N_POINTS_WIDTH = 3

LAME_MU = 1.0
LAME_LAMBDA = 1.25
DENSITY = 1.0
ACCELERATION_DUE_TO_GRAVITY = 0.016


def main():
    comm = MPI.COMM_WORLD

    # ---- Mesh ----
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0], dtype=np.float64),
         np.array([CANTILEVER_LENGTH, CANTILEVER_WIDTH, CANTILEVER_WIDTH], dtype=np.float64)],
        [N_POINTS_LENGTH, N_POINTS_WIDTH, N_POINTS_WIDTH],
        cell_type=mesh.CellType.hexahedron,
    )
    gdim = domain.geometry.dim  # 3

    # ---- Function spaces ----
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))  # vector CG1
    Q = fem.functionspace(domain, ("Lagrange", 1))          # scalar CG1

    # ---- Clamp boundary x=0 ----
    fdim = domain.topology.dim - 1

    def clamp_plane(x):
        return np.isclose(x[0], 0.0)

    clamp_facets = mesh.locate_entities_boundary(domain, fdim, clamp_plane)
    clamp_dofs = fem.locate_dofs_topological(V, fdim, clamp_facets)

    u_D = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, clamp_dofs, V)

    # ---- Kinematics / constitutive ----
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return (LAME_LAMBDA * ufl.tr(epsilon(u)) * ufl.Identity(gdim)
                + 2.0 * LAME_MU * epsilon(u))

    # ---- Variational form ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(
        domain,
        np.array([0.0, 0.0, -DENSITY * ACCELERATION_DUE_TO_GRAVITY], dtype=default_scalar_type),
    )

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx

    # ---- Solve ----
    problem = LinearProblem(
        a, L,
        bcs=[bc],
        petsc_options_prefix="elas_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    u_sol = problem.solve()
    u_sol.name = "Displacement Vector"

    # ---- von Mises stress (interpolated to CG1 scalar) ----
    sig = sigma(u_sol)
    s = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(gdim)
    von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))

    von_mises = fem.Function(Q)
    von_mises.name = "von Mises stress"

    ips = Q.element.interpolation_points
    if callable(ips):
        ips = ips()

    expr = fem.Expression(von_mises_expr, ips)
    von_mises.interpolate(expr)

    # ---- ParaView-clean output: VTK (.pvd/.vtu) ----
    # These load as single datasets in ParaView (no multiblock/partial/duplicate geometry behaviour).
    with io.VTKFile(comm, "displacement.pvd", "w") as vtk:
        vtk.write_mesh(domain)
        vtk.write_function(u_sol, 0.0)

    with io.VTKFile(comm, "von_mises.pvd", "w") as vtk:
        vtk.write_mesh(domain)
        vtk.write_function(von_mises, 0.0)


if __name__ == "__main__":
    main()
