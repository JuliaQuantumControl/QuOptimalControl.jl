# defining some constants for use in the tests
const tol = 1e-6

const ρinit = [1.0 0.0]' * [1.0 0.0 + 0im]
const ρfin = [0.0 1.0]' * [0.0 1.0 + 0im]

const Sρinit = SArray{Tuple{2,2},ComplexF64}(ρinit)
const Sρfin = SArray{Tuple{2,2},ComplexF64}(ρfin)

const Sx = [0.0 1.0; 1.0 0.0+0.0im] / 2
const Sy = [0.0 -1.0im; 1.0im 0.0+0.0im] / 2
const Sz = [1.0 0.0; 0.0 -1.0+0.0im] / 2



const SSx = SArray{Tuple{2,2},ComplexF64}(Sx)
const SSy = SArray{Tuple{2,2},ComplexF64}(Sy)
const SSz = SArray{Tuple{2,2},ComplexF64}(Sz)


const Uinit = Array{ComplexF64,2}((I(2)))
const Ufin = 1 / sqrt(2) * [0 1; 1 -0.0+0.0im]

const ρinitS = SMatrix{2,2,ComplexF64}([1.0 0.0]' * [1.0 0.0 + 0im])
const ρfinS = SMatrix{2,2,ComplexF64}([0.0 1.0]' * [0.0 1.0 + 0im])

const UinitS = SMatrix{2,2,ComplexF64}(Array{ComplexF64,2}((I(2))))
const UfinS = SMatrix{2,2,ComplexF64}(1 / sqrt(2) * [0 1; 1 -0.0+0.0im])


A_gens = k -> (k - 2.5) / 2.5 * Sz * 5
B_gens = k -> [Sx, Sy]



A_gens_static = k -> (k - 2.5) / 2.5 * SSz * 5
B_gens_static = k -> [SSx, SSy]


function odd_switch(x)
    if Bool(x % 2)
        return ρfin
    else
        return ρinit
    end
end

X_init_gens = k -> ρinit
X_target_gens = k -> odd_switch(k)



X_init_gens_static = k -> ρinitS
X_target_gens_static = k -> SMatrix{2,2,ComplexF64}(odd_switch(k))
