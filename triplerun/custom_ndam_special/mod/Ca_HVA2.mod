:Comment :
:Reference : :		Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993
: 								activation from Sayer, Schwindt and Crill (1990)
:									inactivation from Dichter and Zona 1989
: LJP: OK, inactivation corrected by 12.3 mV

NEURON	{
	SUFFIX Ca_HVA2
	USEION ca READ eca WRITE ica
	RANGE gCa_HVAbar, gCa_HVA, ica
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gCa_HVAbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)
	gCa	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
    i_init (mA/cm2)
    gCa0 (S/cm2)
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gCa = gCa_HVAbar*m*m*h
	ica = gCa*(v-eca) - i_init
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
    gCa0 = gCa_HVAbar*m*m*h
    i_init = gCa0*(v-eca)
}

PROCEDURE rates(){
	UNITSOFF
        if((v == -27) ){
            v = v+0.0001
        }
		mAlpha =  (0.055*(-27-v))/(exp((-27-v)/3.8) - 1)
		mBeta  =  (0.94*exp((-75-v)/17))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
		v = v + 12.3
		hAlpha =  (0.000457*exp((-13-v)/50))
		hBeta  =  (0.0065/(exp((-v-15)/28)+1))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = 1/(hAlpha + hBeta)
		v = v - 12.3
	UNITSON
}
