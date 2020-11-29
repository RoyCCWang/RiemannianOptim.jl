import Pkg
#cmd_string = "./deps/my_private_repo/"
#Pkg.add(Pkg.PackageSpec(url=cmd_string))

Pkg.add("SpecialFunctions")
Pkg.add("BSON")
Pkg.add("Optim")
Pkg.build()



Pkg.activate(".")
Pkg.build()
