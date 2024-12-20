COMMENT
/**
 * @file ProbAMPANMDA_EMS.mod
 * @brief
 * @author king, muller, reimann, ramaswamy
 * @date 2011-08-17
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */
ENDCOMMENT

TITLE Probabilistic AMPA and NMDA receptor with presynaptic short-term plasticity


COMMENT
AMPA and NMDA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity as in Fuhrmann et al. 2002

_EMS (Eilif Michael Srikanth)
Modification of ProbAMPANMDA: 2-State model by Eilif Muller, Michael Reimann, Srikanth Ramaswamy, Blue Brain Project, August 2011
This new model was motivated by the following constraints:

1) No consumption on failure.
2) No release just after release until recovery.
3) Same ensemble averaged trace as deterministic/canonical Tsodyks-Markram
   using same parameters determined from experiment.
4) Same quantal size as present production probabilistic model.

To satisfy these constaints, the synapse is implemented as a
uni-vesicular (generalization to multi-vesicular should be
straight-forward) 2-state Markov process.  The states are
{1=recovered, 0=unrecovered}.

For a pre-synaptic spike or external spontaneous release trigger
event, the synapse will only release if it is in the recovered state,
and with probability u (which follows facilitation dynamics).  If it
releases, it will transition to the unrecovered state.  Recovery is as
a Poisson process with rate 1/Dep.

This model satisfies all of (1)-(4).

ENDCOMMENT


NEURON {
    THREADSAFE
        POINT_PROCESS ProbAMPANMDA_EMS
        GLOBAL tau_r_AMPA
        RANGE tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
        RANGE Use, Dep, Fac, u0, mg, Rstate, tsyn, tsyn_fac, u, release_accumulator
        RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA, NMDA_ratio, Nrrp
        RANGE A_AMPA_step, B_AMPA_step, A_NMDA_step, B_NMDA_step,ins
        RANGE ina, ik, ica, ina_NMDA, ik_NMDA, ica_NMDA, ina_AMPA, ik_AMPA, ica_AMPA, i_sum
        RANGE absPerm_na_NMDA, absPerm_na_AMPA
        GLOBAL e_NMDA, e_AMPA
        RANGE nai,nao,ki,ko,cai,cao
        RANGE occupied, unoccupied
        USEION na READ ena, nai, nao WRITE ina
        USEION k READ ek, ki, ko WRITE ik
        USEION ca READ eca, cai, cao WRITE ica
        BBCOREPOINTER rng
        RANGE synapseID, verboseLevel, selected_for_report

    GLOBAL slope_mg, scale_mg
    RANGE conductance
    RANGE next_delay
    BBCOREPOINTER delay_times, delay_weights
    GLOBAL nc_type_param
    GLOBAL minis_single_vesicle
    :RANGE sgid, tgid  : For debugging
}

UNITS	{
  	(uS) = (microsiemens)
  	(mV) = (millivolt)
  	(nA) = (nanoamp)
    (molar) = (1/liter)
    (mM) = (millimolar)
}

PARAMETER {
        tau_r_AMPA = 0.2   (ms)  : dual-exponential conductance profile
        tau_d_AMPA = 1.7    (ms)  : IMPORTANT: tau_r < tau_d
        tau_r_NMDA = 0.29   (ms) : dual-exponential conductance profile
        tau_d_NMDA = 43     (ms) : IMPORTANT: tau_r < tau_d
        Use = 1.0   (1)   : Utilization of synaptic efficacy (just initial values! Use, Dep and Fac are overwritten by BlueBuilder assigned values)
        Dep = 100   (ms)  : relaxation time constant from depression
        Fac = 10   (ms)  :  relaxation time constant from facilitation
        mg = 1   (mM)  : initial concentration of mg2+
        slope_mg = 0.062    (/mV) : default variables from Jahr & Stevens 1990
        scale_mg = 3.57     (mM)
        
        gmax = .001 (uS) : weight conversion factor (from nS to uS)
        u0 = 0 :initial value of u, which is the running value of release probability
        
        Nrrp = 1            (1)   : Number of total release sites for given contact

        synapseID = 0
        verboseLevel = 0

        :These are not actually relative permeabilities. Better to think of them as current ratios.
        :Values for calcium "permeability" were calculated by finding Ica/Ik at overall reversal potential using model implemented with GHK eqn, ProbAMPANMDA_EMS_ionic.mod
        :This gives a current ratio which is applicable here rather than using relative permeability, which results in orders of magnitude more Ca current.
        :Calculated using ionic concentrations of ko=5.6408,ki=140.0282,na=140.0,nai=9.315468,ca=2.0,cai=1e-4
        relP_na_NMDA = 1
        relP_k_NMDA = 1
        relP_ca_NMDA = 0.075
        relP_na_AMPA = 1
        relP_k_AMPA = 1
        relP_ca_AMPA = 0.00118
        e_NMDA = 0 (mV)  :NMDA reversal potential
        e_AMPA = 0 (mV)  :AMPA reversal potential
 
        selected_for_report = 0
        NMDA_ratio = 0.71   (1)   : The ratio of NMDA to AMPA

        conductance = 0.0
        nc_type_param = 4
        minis_single_vesicle = 0   :// 0 - no limit (old behavior)
}

COMMENT
The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1
for comparison with Pr to decide whether to activate the synapse or not
ENDCOMMENT

VERBATIM

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include "nrnran123.h"

#ifndef CORENEURON_BUILD
extern int ifarg(int iarg);

extern void* vector_arg(int iarg);
extern double* vector_vec(void* vv);
extern int vector_capacity(void* vv);
#endif

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

ENDVERBATIM


ASSIGNED {
        dt (ms) :modlunit wanted it to be assigned units of ms
        v (mV)
        i (nA)
        i_AMPA (nA)
        i_NMDA (nA)
        g_AMPA (uS)
        g_NMDA (uS)
        factor_AMPA
        factor_NMDA
        A_AMPA_step
        B_AMPA_step
        A_NMDA_step
        B_NMDA_step
        rng
        ina      (nA)
        ik       (nA)
        ica       (nA)
        ina_NMDA  (nA)
        ik_NMDA   (nA)
        ica_NMDA  (nA)
        ina_AMPA  (nA)
        ik_AMPA   (nA)
        ica_AMPA  (nA)
        i_sum     (nA)
        celsius   (degC) :Needs to be changed by user if user does not want the default value
        ena (mV)
        ek  (mV)
        eca (mV)
        nai (mM)
        nao (mM)
        ki  (mM)
        ko  (mM)
        cai (mM)
        cao (mM)
        nafrac_AMPA
        kfrac_AMPA
        cafrac_AMPA
        nafrac_NMDA
        kfrac_NMDA
        cafrac_NMDA

	      : Recording these three, you can observe full state of model
	      : tsyn_fac gives you presynaptic times, Rstate gives you
        : state transitions,
        : u gives you the "release probability" at transitions
        : (attention: u is event based based, so only valid at incoming events)

        Rstate (1) : recovered state {0=unrecovered, 1=recovered}
	      tsyn_fac (ms) : the time of the last spike
        tsyn (ms) : the time of the last spike
	      u (1) : running release probability
:        release_accumulator(0)
 
        mggate
        usingR123            : TEMPORARY until mcellran4 completely deprecated

        : MVR
        unoccupied (1) : no. of unoccupied sites following release event
        occupied   (1) : no. of occupied sites following one epoch of recovery

        : stuff for delayed connections
        delay_times
        delay_weights
        next_delay (ms)
}

STATE {
 release_accumulator
        A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA
        B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA
        A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA
        B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA
}

INITIAL {
        LOCAL tp_AMPA, tp_NMDA
        release_accumulator=0

        Rstate=1
        tsyn_fac=0
        tsyn = 0
        u=u0

        : MVR
        unoccupied = 0
        occupied = Nrrp

        A_AMPA = 0
        B_AMPA = 0

        A_NMDA = 0
        B_NMDA = 0

        tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance
        tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance

        factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak
        factor_AMPA = 1/factor_AMPA

        factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak
        factor_NMDA = 1/factor_NMDA

        A_AMPA_step = exp(dt*(( - 1.0 ) / tau_r_AMPA))
        B_AMPA_step = exp(dt*(( - 1.0 ) / tau_d_AMPA))
        A_NMDA_step = exp(dt*(( - 1.0 ) / tau_r_NMDA))
        B_NMDA_step = exp(dt*(( - 1.0 ) / tau_d_NMDA))

        nafrac_AMPA=relP_na_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
        kfrac_AMPA=relP_k_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
        cafrac_AMPA=relP_ca_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
        nafrac_NMDA=relP_na_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
        kfrac_NMDA=relP_k_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
        cafrac_NMDA=relP_ca_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
        VERBATIM
        if( usingR123 ) {
            nrnran123_setseq((nrnran123_State*)_p_rng, 0, 0);
        }
        ENDVERBATIM

        next_delay = -1

}

PROCEDURE setup_delay_vecs() {
VERBATIM
#ifndef CORENEURON_BUILD
    void** vv_delay_times = (void**)(&_p_delay_times);
    void** vv_delay_weights = (void**)(&_p_delay_weights);
    *vv_delay_times = (void*)NULL;
    *vv_delay_weights = (void*)NULL;
    if (ifarg(1)) {
        *vv_delay_times = vector_arg(1);
    }
    if (ifarg(2)) {
        *vv_delay_weights = vector_arg(2);
    }
#endif
ENDVERBATIM
}

BREAKPOINT {
        SOLVE state
        mggate = 1 / (1 + exp(slope_mg * -(v)) * (mg / scale_mg)) :mggate kinetics
        g_AMPA = gmax*(B_AMPA-A_AMPA) :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA
        g_NMDA = gmax*(B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics

        i_AMPA = g_AMPA*(v-e_AMPA) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal
        i_NMDA = g_NMDA*(v-e_NMDA) :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal
        i = i_AMPA + i_NMDA   :Calculated only for comparison to i_sum. They should be equal. Previous 3 lines can be removed.

        ina_NMDA = nafrac_NMDA*g_NMDA*(v-ena+findOffsetNa(relP_na_NMDA,relP_k_NMDA,relP_ca_NMDA,e_NMDA))
        ik_NMDA = kfrac_NMDA*g_NMDA*(v-ek)
        ica_NMDA = cafrac_NMDA*g_NMDA*(v-eca)

        ina_AMPA = nafrac_AMPA*g_AMPA*(v-ena+findOffsetNa(relP_na_AMPA,relP_k_AMPA,relP_ca_AMPA,e_AMPA))
        ik_AMPA = kfrac_AMPA*g_AMPA*(v-ek)
        ica_AMPA = cafrac_AMPA*g_AMPA*(v-eca)

        ina = ina_NMDA + ina_AMPA
        ik = ik_NMDA + ik_AMPA
        ica = ica_NMDA + ica_AMPA

        i_sum = ina+ik+ica
}

PROCEDURE state() {
        A_AMPA = A_AMPA*A_AMPA_step
        B_AMPA = B_AMPA*B_AMPA_step
        A_NMDA = A_NMDA*A_NMDA_step
        B_NMDA = B_NMDA*B_NMDA_step
}


NET_RECEIVE (weight,weight_AMPA, weight_NMDA, Psurv, nc_type) {
    : Psurv - survival probability of unrecovered state
    : nc_type:
    :   0 = presynaptic netcon
    :   1 = spontmini netcon
    :   2 = replay netcon

    LOCAL result, ves, occu
    weight_AMPA = weight
    weight_NMDA = weight * NMDA_ratio

    INITIAL {
        if (nc_type == 0) {  :// presynaptic netcon
    VERBATIM
            // setup self events for delayed connections to change weights
            void *vv_delay_times = *((void**)(&_p_delay_times));
            void *vv_delay_weights = *((void**)(&_p_delay_weights));
            if (vv_delay_times && vector_capacity(vv_delay_times)>=1) {
                double* deltm_el = vector_vec(vv_delay_times);
                int delay_times_idx;
                next_delay = 0;
                for (delay_times_idx = 0; delay_times_idx < vector_capacity(vv_delay_times); ++delay_times_idx) {
                    double next_delay_t = deltm_el[delay_times_idx];
    ENDVERBATIM
                    net_send(next_delay_t, 1)
    VERBATIM
                }
            }
    ENDVERBATIM
        }
    }

    if (flag == 1) {  :// self event to set next weight at
    VERBATIM
        void *vv_delay_weights = *((void**)(&_p_delay_weights));
        if (vv_delay_weights && vector_capacity(vv_delay_weights)>=next_delay) {
            double* weights_v = vector_vec(vv_delay_weights);
            double next_delay_weight = weights_v[(int)next_delay];
    ENDVERBATIM
            weight = conductance * next_delay_weight
            next_delay = next_delay + 1
    VERBATIM
        }
        return;
    ENDVERBATIM
    }

    : [flag == 0] Handle a spike which arrived
    :UNITSOFF
    :printf( "synapse %f (%f, %f) with weight %g at time %g\n", synapseID, sgid, tgid, weight, t)
    :UNITSON

    : Do not perform any calculations if the synapse (netcon) is deactivated. This avoids drawing from
    : random number stream. Also, disable in case of t < 0 (in case of ForwardSkip) which causes numerical
    : instability if synapses are activated.
    if ( weight <= 0 || t < 0 ) {
    VERBATIM
        return;
    ENDVERBATIM
    }

    : calc u at event-
    if (Fac > 0) {
        u = u * exp(-(t - tsyn)/Fac)  :// update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    } else {
        u = Use
    }
    if(Fac > 0){
        u = u + Use*(1-u)  :// update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    }

    : recovery
    FROM counter = 0 TO (unoccupied - 1) {
        : Iterate over all unoccupied sites and compute how many recover
        Psurv = exp(-(t-tsyn)/Dep)
        result = urand()
        if (result>Psurv) {
            occupied = occupied + 1     :// recover a previously unoccupied site
            if ( verboseLevel > 0 ) {
                UNITSOFF
                printf( "Recovered! %f at time %g: Psurv = %g, urand=%g\n", synapseID, t, Psurv, result )
                UNITSON
            }
        }
    }
   ves = 0                  :// Initialize the number of released vesicles to 0
   occu = occupied        :// Make a copy, so we can update occupied in the loop
    if (occu > 1 && minis_single_vesicle && nc_type == 1) {    : // if nc_type is spont_mini consider single vesicle
        occu = 1
    }
    FROM counter = 0 TO (occu - 1) {
        : iterate over all occupied sites and compute how many release
        result = urand()
        if (result < u) {
            : release a single site!
            occupied = occupied - 1  :// decrease the number of occupied sites by 1
            ves = ves + 1            :// increase number of relesed vesicles by 1
        }
    }

    : Update number of unoccupied sites
    unoccupied = Nrrp - occupied

    : Update tsyn
    : tsyn knows about all spikes, not only those that released
    : i.e. each spike can increase the u, regardless of recovered state.
    :      and each spike trigger an evaluation of recovery
    tsyn = t

    if (ves > 0) { :no need to evaluate unless we have vesicle release
        release_accumulator=release_accumulator+1

        A_AMPA = A_AMPA + ves/Nrrp*weight_AMPA*factor_AMPA
        B_AMPA = B_AMPA + ves/Nrrp*weight_AMPA*factor_AMPA
        A_NMDA = A_NMDA + ves/Nrrp*weight_NMDA*factor_NMDA
        B_NMDA = B_NMDA + ves/Nrrp*weight_NMDA*factor_NMDA

        if ( verboseLevel > 0 ) {
            UNITSOFF
            printf( "[Syn %f] Release! t = %g: vals %g %g %g %g\n", synapseID, t, A_AMPA, weight_AMPA, factor_AMPA, weight )
            UNITSON
        }

    } else {
        : total release failure
        if ( verboseLevel > 0 ) {
            UNITSOFF
            printf("[Syn %f] Failure! t = %g: urand = %g\n", synapseID, t, result)
            UNITSON
        }
    }
}


PROCEDURE setRNG() {
VERBATIM
    #ifndef CORENEURON_BUILD
    // For compatibility, allow for either MCellRan4 or Random123
    // Distinguish by the arg types
    // Object => MCellRan4, seeds (double) => Random123
    usingR123 = 0;
    if( ifarg(1) && hoc_is_double_arg(1) ) {
        nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
        uint32_t a2 = 0;
        uint32_t a3 = 0;

        if (*pv) {
            nrnran123_deletestream(*pv);
            *pv = (nrnran123_State*)0;
        }
        if (ifarg(2)) {
            a2 = (uint32_t)*getarg(2);
        }
        if (ifarg(3)) {
            a3 = (uint32_t)*getarg(3);
        }
        *pv = nrnran123_newstream3((uint32_t)*getarg(1), a2, a3);
        usingR123 = 1;
    } else if( ifarg(1) ) {   // not a double, so assume hoc object type
        void** pv = (void**)(&_p_rng);
        *pv = nrn_random_arg(1);
    } else {  // no arg, so clear pointer
        void** pv = (void**)(&_p_rng);
        *pv = (void*)0;
    }
    #endif
ENDVERBATIM
}


FUNCTION urand() {
VERBATIM
    double value = 0.0;
    if ( usingR123 ) {
        value = nrnran123_dblpick((nrnran123_State*)_p_rng);
    } else if (_p_rng) {
        #ifndef CORENEURON_BUILD
        value = nrn_random_pick(_p_rng);
        #endif
    } else {
        // Note: prior versions used scop_random(1), but since we never use this model without configuring the rng.  Maybe should throw error?
        value = 0.0;
    }
    _lurand = value;
ENDVERBATIM
}

FUNCTION bbsavestate() {
        bbsavestate = 0
VERBATIM
#ifndef CORENEURON_BUILD
        /* first arg is direction (0 save, 1 restore), second is array*/
        /* if first arg is -1, fill xdir with the size of the array */
        double *xdir, *xval, *hoc_pgetarg();
        long nrn_get_random_sequence(void* r);
        void nrn_set_random_sequence(void* r, int val);
        xdir = hoc_pgetarg(1);
        xval = hoc_pgetarg(2);
        if (_p_rng) {
            // tell how many items need saving
            if (*xdir == -1) {  // count items
                if( usingR123 ) {
                    *xdir = 2.0;
                } else {
                    *xdir = 1.0;
                }
                return 0.0;
            } else if(*xdir ==0 ) {  // save
                if( usingR123 ) {
                    uint32_t seq;
                    char which;
                    nrnran123_getseq( (nrnran123_State*)_p_rng, &seq, &which );
                    xval[0] = (double) seq;
                    xval[1] = (double) which;
                } else {
                    xval[0] = (double)nrn_get_random_sequence(_p_rng);
                }
            } else {  // restore
                if( usingR123 ) {
                    nrnran123_setseq( (nrnran123_State*)_p_rng, (uint32_t)xval[0], (char)xval[1] );
                } else {
                    nrn_set_random_sequence(_p_rng, (long)(xval[0]));
                }
            }
        }
#endif
ENDVERBATIM
}

FUNCTION toggleVerbose() {
    verboseLevel = 1 - verboseLevel
}



FUNCTION findOffsetNa(relP_na, relP_k, relP_ca, e (mV)) (mV) {
	findOffsetNa = (-relP_ca*(e-eca)-relP_k*(e-ek))/(relP_na) + ena - e
}


VERBATIM
static void bbcore_write(double* x, int* d, int* x_offset, int* d_offset, _threadargsproto_) {

  void *vv_delay_times = *((void**)(&_p_delay_times));
  void *vv_delay_weights = *((void**)(&_p_delay_weights));

  if (d) {
    uint32_t* di = ((uint32_t*)d) + *d_offset;
    nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
    nrnran123_getids3(*pv, di, di+1, di+2);

    char which;
    nrnran123_getseq(*pv, di+3, &which);
    di[4] = (int)which;
    //printf("SYN bbcore_write %d %d %d\n", di[0], di[1], di[2]);

  }
  // reserve random123 parameters on serialization buffer
  *d_offset += 5;

  // serialize connection delay vectors
  if (vv_delay_times && vv_delay_weights &&
     (vector_capacity(vv_delay_times) >= 1) && (vector_capacity(vv_delay_weights) >= 1)) {
    if (d) {
      uint32_t* di = ((uint32_t*)d) + *d_offset;
      // store vector sizes for deserialization
      di[0] = vector_capacity(vv_delay_times);
      di[1] = vector_capacity(vv_delay_weights);
    }
    if (x) {
      double* delay_times_el = vector_vec(vv_delay_times);
      double* delay_weights_el = vector_vec(vv_delay_weights);
      double* x_i = x + *x_offset;
      int delay_vecs_idx;
      int x_idx = 0;
      for(delay_vecs_idx = 0; delay_vecs_idx < vector_capacity(vv_delay_times); ++delay_vecs_idx) {
         x_i[x_idx++] = delay_times_el[delay_vecs_idx];
         x_i[x_idx++] = delay_weights_el[delay_vecs_idx];
      }
    }
    // reserve space for connection delay data on serialization buffer
    *x_offset += vector_capacity(vv_delay_times) + vector_capacity(vv_delay_weights);
  } else {
    if (d) {
      uint32_t* di = ((uint32_t*)d) + *d_offset;
      di[0] = 0;
      di[1] = 0;
    }

  }
  // reserve space for delay vectors (may be 0)
  *d_offset += 2;

}

static void bbcore_read(double* x, int* d, int* x_offset, int* d_offset, _threadargsproto_) {
  assert(!_p_rng && !_p_delay_times && !_p_delay_weights);

  // deserialize random123 data
  uint32_t* di = ((uint32_t*)d) + *d_offset;
  if (di[0] != 0 || di[1] != 0 || di[2] != 0) {
      nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
      *pv = nrnran123_newstream3(di[0], di[1], di[2]);
      char which = (char)di[4];
      nrnran123_setseq(*pv, di[3], which);
  }
  //printf("ProbAMPANMDA_EMS bbcore_read %d %d %d\n", di[0], di[1], di[2]);

  int delay_times_sz = di[5];
  int delay_weights_sz = di[6];
  *d_offset += 7;

  if ((delay_times_sz > 0) && (delay_weights_sz > 0)) {
    double* x_i = x + *x_offset;

    // allocate vectors
    _p_delay_times = vector_new1(delay_times_sz);
    _p_delay_weights = vector_new1(delay_weights_sz);

    double* delay_times_el = vector_vec(_p_delay_times);
    double* delay_weights_el = vector_vec(_p_delay_weights);

    // copy data
    int x_idx;
    int vec_idx = 0;
    for(x_idx = 0; x_idx < delay_times_sz + delay_weights_sz; x_idx += 2) {
      delay_times_el[vec_idx] = x_i[x_idx];
      delay_weights_el[vec_idx++] = x_i[x_idx+1];
    }
    *x_offset += delay_times_sz + delay_weights_sz;

  }
}
ENDVERBATIM
