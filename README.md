# Funding

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

# Copyright

Copyright (c) 2024 Blue Brain Project/EPFL

# Project description

This is the detailed NGV metabolism model. Both unitary metabolism model and the Neurodamus-coupled model (running with the circuit) are here.
For any questions please contact polina.shichkova@epfl.ch

# Requirements

module neurodamus-neocortex
module julia/1.5.2 or 1.6.0
module py-neurodamus
module py-bluepy
module py-efel

bbpcode.epfl.ch/cells/BluePyEModel with git review -d 52878 with dependencies therein

diffeqpy
numpy
textwrap
collections
contextlib
pickle
Neurodamus
mpi4py


Julia pkgs:

  [0f109fa4] BifurcationKit v0.1.3
  [a134a8b2] BlackBoxOptim v0.5.0
  [8d3b24bd] CMAEvolutionStrategy v0.2.1
  [336ed68f] CSV v0.8.4
  [479239e8] Catalyst v6.12.1
  [a93c6f00] DataFrames v1.0.1
  [2b5f629d] DiffEqBase v6.60.0
  [aae7a2af] DiffEqFlux v1.36.1
  [c894b116] DiffEqJump v6.14.1
  [1130ab10] DiffEqParamEstim v1.20.1
  [0c46a032] DifferentialEquations v6.16.0
  [86b6b26d] Evolutionary v0.9.0
  [587475ba] Flux v0.12.1
  [a75be94c] GalacticOptim v1.1.0
  [7073ff75] IJulia v1.23.2
  [23fbe1c1] Latexify v0.15.5
  [961ee093] ModelingToolkit v5.16.0
  [3933049c] MultistartOptimization v0.1.2
  [76087f3c] NLopt v0.6.2
  [429524aa] Optim v1.3.0
  [1dea7af3] OrdinaryDiffEq v5.52.7
  [65888b18] ParameterizedFunctions v5.10.0
  [91a5bcdd] Plots v1.13.2
  [438e738f] PyCall v1.92.3
  [731186ca] RecursiveArrayTools v2.11.3
  [efcf1570] Setfield v0.7.0
  [276daf66] SpecialFunctions v0.10.3
  [9672c7b4] SteadyStateDiffEq v1.6.2
  [789caeaf] StochasticDiffEq v6.33.1

This is WIP. So that requirements will be updated in the future. 


# Installation procedure

only installation of dependencies is required

