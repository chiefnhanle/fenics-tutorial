import dolfinx
# 1. Check version
print("FEniCS version:", dolfin.__version__)

# 2. Simple mesh
mesh = UnitSquareMesh(8, 8)

# 3. Function space
V = FunctionSpace(mesh, "Lagrange", 1)

# 4. Boundary condition u = 0
u_D = Constant(0.0)
def boundary(x):
    return True  # all boundaries
bc = DirichletBC(V, u_D, boundary)

# 5. Variational problem: -Δu = 1
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# 6. Solve
u_sol = Function(V)
solve(a == L, u_sol, bc)

# 7. Output a sample
print("u(0.5, 0.5) =", u_sol(0.5, 0.5))
print("Test completed successfully.")
