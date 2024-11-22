: Intracellular ion accumulation
: Author: Braeden Benedict
: Date: 27-06-2017
: Does not include changing external concentrations or any diffusion

NEURON {
 SUFFIX internal_ions
 RANGE i_Tot
 USEION na READ ina WRITE nai
 USEION k READ ik WRITE ki
 USEION ca READ ica
 USEION cl READ icl WRITE cli VALENCE -1
 RANGE depth
}

DEFINE Nannuli 2

UNITS {
 (mA) = (milliamp)
 FARADAY = (faraday) (coulombs)
 (molar) = (1/liter)
 (mM) = (millimolar)
 PI = (pi) (1)
 (um) = (micron)
}

ASSIGNED {
  ina (mA/cm2)
  ik (mA/cm2)
  ica (mA/cm2)
  icl (mA/cm2)
  diam (micron)
  depth (micron)
  i_Tot (mA/cm2)
  conversionFactor (1/cm/coulombs)
  conversionFactorCa (1/cm/coulombs)
}

PARAMETER {
  DCa = 0.53 (um2/ms)
  DNa = 1.33 (um2/ms)
  DK = 1.96 (um2/ms)
  DCl = 2.03 (um2/ms)
}

STATE {
  nai (mM)
  ki (mM)
  cli (mM)
}

LOCAL volin, surf

INITIAL {
  depth = 0.1 :diam/4/(Nannuli-1)
  volin = PI*diam*diam/4 : Surface area (volume per unit length)
  surf = PI*diam : circumference (segment surface per unit length)
  conversionFactor = (1e4)*4/diam/FARADAY
}

BREAKPOINT {
  SOLVE state METHOD sparse
  if ( nai <= 0 ) {
    nai = 0.01
  }
  i_Tot = ina + ik + ica + icl :Sum of all ionic currents, for user's convenience
}

KINETIC state {
 COMPARTMENT volin {nai ki cli}
 LONGITUDINAL_DIFFUSION DNa*volin {nai}
 LONGITUDINAL_DIFFUSION DK*volin {ki}
 LONGITUDINAL_DIFFUSION DCl*volin {cli}
 ~ nai << (-ina * conversionFactor*volin)
 ~ ki << (-ik  * conversionFactor*volin)
 ~ cli << (icl * conversionFactor*volin)
}
