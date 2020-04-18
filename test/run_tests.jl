import Pkg


Pkg.instantiate()

using Test
import LinearAlgebra
import Statistics
import Printf

import ForwardDiff
import RiemannianOptim TestCI

tests = [   "test1_with_Hessian";
            "test1_without_Hessian"]

for t in tests
    @testset "$t" begin
        include("$(t).jl")
    end
end
