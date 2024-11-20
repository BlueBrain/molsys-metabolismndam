#!/bin/env python
# inspired from http://steps.sourceforge.net/manual/diffusion.html
# to run: srun -Aproj40  special -mpi -python dualrun6.py

from __future__ import print_function
import logging
import numpy as np
import textwrap
import steps
import steps.model as smodel
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as meshio
import steps.utilities.geom_decompose as gd
from neurodamus import Neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from mpi4py import MPI
comm = MPI.COMM_WORLD
import dualrun.timer.mpi as mt


#import bluepy 
#from bluepy import Circuit

#import numpy as np

#from bluepy.enums import Synapse

#c.connectome.afferent_gids(10)
#circuit = Circuit("BlueConfig")
#connectome = circuit.connectome
#gid=9
#pre_pos = connectome.afferent_synapses(gid, properties=[Synapse.PRE_X_CENT\
#ER, Synapse.PRE_Y_CENTER, Synapse.PRE_Z_CENTER]).to_numpy()


np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = 1.602176634e-19

REPORT_FLAG = True


#################################################
# Model Specification
#################################################

dt_nrn2dt_steps: int = 4

class Geom:
    meshfile = './steps_meshes/cubeBig7'
    compname = 'extra'


class Na:
    name = 'Na'
    conc_0 = 140  # (mM/L)
    diffname = 'diff_Na'
    diffcst = 2e-9
    current_var = 'ina'
    charge = 1 * ELEM_CHARGE


class Volsys0:
    name = 'extraNa'
    specs = (Na,)


#################################################
# Model Build
#################################################

def gen_model():
    mdl = smodel.Model()
    vsys = smodel.Volsys(Volsys0.name, mdl)
    na_spec = smodel.Spec(Na.name, mdl)
    diff = smodel.Diff(Na.diffname, vsys, na_spec)
    diff.setDcst(Na.diffcst)
    return mdl


def gen_geom():
    mesh = meshio.loadMesh(Geom.meshfile)[0]
    ntets = mesh.countTets()
    comp = stetmesh.TmComp(Geom.compname, mesh, range(ntets))
    comp.addVolsys(Volsys0.name)
    return mesh, ntets


def init_solver(model, geom):
    rng = srng.create('mt19937', 512)
    rng.initialize(2903)

    import steps.mpi.solver as psolver
    tet2host = gd.linearPartition(geom, [1, 1, steps.mpi.nhosts])
    if steps.mpi.rank == 0:
        print("Number of tets: ", geom.ntets)
        # gd.validatePartition(geom, tet2host)
        # gd.printPartitionStat(tet2host)
    return psolver.TetOpSplit(model, geom, rng, psolver.EF_NONE, tet2host), tet2host



# returns a list of tet mappings for the neurons
# each segment mapping is a list of (tetnum, fraction of segment)
def get_sec_mapping(ndamus, mesh):
    neurSecmap = []

    micrometer2meter = 1e-6
    rank = comm.Get_rank()

    # Init bounding box
    inf: float = 1e15
    n_bbox_min = np.array([inf, inf, inf], dtype=float)
    n_bbox_max = np.array([-inf, -inf, -inf], dtype=float)

    count00=0
    
    for c in ndamus.circuits.base_cell_manager.cells:
        #print(dir(c.CCell.soma[0]))
        #print((c.CCell.soma[0].x3d(0)))
        print("Cell ",count00)
        count00=count00+1
        # Get local to global coordinates transform

        loc_2_glob_tsf = c.local_to_global_coord_mapping
        secs = [sec for sec in c.CCell.all if hasattr(sec, Na.current_var)]
        # Our returned struct is #neurons long list of
        # [(sec1, nparray([pt1, pt2, ...])), (sec2, npa...]
        secmap = []
        count0=0
        
        for sec in secs:
            npts0=0
            for seg in sec.allseg():
                #print("initarea: ",seg.area())
                npts0=npts0+1
            npts = sec.n3d()
            if not npts:
#                print("sec does not have 3d ",count0)
#                count0=count0+1
#                continue
                pts = np.empty((npts0, 3), dtype=float)
                for i in range(npts0):
                    pts[i] = np.array([c.CCell.soma[0].x3d(0), c.CCell.soma[0].y3d(0), c.CCell.soma[0].z3d(0)+i*0.00000001]) * micrometer2meter
                loc_pts=pts
            else:
                # Store points in section
                pts = np.empty((npts, 3), dtype=float)

                for i in range(npts):
                    pts[i] = np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]) * micrometer2meter
                loc_pts= loc_2_glob_tsf(pts)
   

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = np.array(loc_pts, dtype=float, order='C')

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)

            # get tet to fraction map, batch processing all points (segments)
            tet2fraction_map = mesh.intersect(pts_abs_coo)

            # store map for each section
            #print("Appending")
            secmap.append((sec, tet2fraction_map))

        #print("secmap length",len(secmap))    
#        for sec, tet2fraction_map in secmap:
#            for seg in sec.allseg():
#                print("initfaux",seg.area())

        neurSecmap.append(secmap)

#check
#        for secmap in neurSecmap:
#            for sec, tet2fraction_map in secmap:
#                for seg, tet2fraction in zip(sec.allseg(), tet2fraction_map):
#                    print("initfaux",seg.area())

    # Reduce Neuron bbox
    n_bbox_min_glo = np.empty((3), dtype=float)
    n_bbox_max_glo = np.empty((3), dtype=float)
    comm.Reduce([n_bbox_min, 3, MPI.DOUBLE], [n_bbox_min_glo, 3, MPI.DOUBLE], \
        op=MPI.MIN, root=0)
    comm.Reduce([n_bbox_max, 3, MPI.DOUBLE], [n_bbox_max_glo, 3, MPI.DOUBLE], \
        op=MPI.MAX, root=0)

    # Check bounding boxes
    if rank == 0:
        s_bbox_min = mesh.getBoundMin()
        s_bbox_max = mesh.getBoundMax()

        print("bounding box Neuron:", n_bbox_min_glo, n_bbox_max_glo)
        print("bounding box STEPS:", s_bbox_min, s_bbox_max)

        # Should add tolerance to check bounding box
        if np.less(n_bbox_min_glo, s_bbox_min).any() or np.greater(n_bbox_max_glo, s_bbox_max).any():
            logging.warning("STEPS mesh does not overlap with all neurons")

    return neurSecmap


##############################################
# Runtime
##############################################

def timesteps(end: float, step: float):
    return ((i+1) * step for i in range(int(end/step)))


def main():
    ndamus = Neurodamus("BlueConfig", enable_reports=False, logging_level=None)

    # Simulate one molecule each 10e9

    # Times are in ms
    DT = ndamus._run_conf['Dt']
    SIM_END = ndamus._run_conf['Duration']
    DT_s = DT * 1e3 * dt_nrn2dt_steps

    # In steps use M/L and apply the SIM_REAL ratio
    CONC_FACTOR = 1e-9

    AVOGADRO = 6.02e23
    COULOMB = 6.24e18
    CA = COULOMB/AVOGADRO*CONC_FACTOR*DT_s


    log_stage("Initializing steps model and geom...")
    model = gen_model()
    tmgeom, ntets = gen_geom()

    logging.info("Computing segments per tet...")
    neurSecmap = get_sec_mapping(ndamus, tmgeom)


    with mt.timer.region('init_sims'):
        logging.info("Initializing simulations...")
        ndamus.sim_init()
        steps_sim, tet2host = init_solver(model, tmgeom)
        steps_sim.reset()
        # there are 0.001 M/mM
        steps_sim.setCompConc(Geom.compname, Na.name, 0.001 * Na.conc_0 * CONC_FACTOR)
        tetVol = np.array([tmgeom.getTetVol(x) for x in range(ntets)], dtype=float)

    log_stage("===============================================")
    log_stage("Running both STEPS and Neuron simultaneously...")

    rank: int = comm.Get_rank()
    if rank == 0 and REPORT_FLAG:
        f = open("tetConcs.txt", "w+")

    def fract(neurSecmap):
        # compute the currents arising from each segment into each of the tets
        tet_currents = np.zeros((ntets,), dtype=float)
        count0=0
        sum1=0
        for secmap in neurSecmap:
            print("sum1seccountzero ",count0)
            count0=count0+1
            count1=0
            for sec, tet2fraction_map in secmap:
                print("sum1seccount ",count1)
                count1=count1+1
                for seg, tet2fraction in zip(sec.allseg(), tet2fraction_map):
                    if tet2fraction:
                        tet: int
                        fract: float
                        #sum1=0                
                        for tet, fract in tet2fraction:
                            # there are 1e8 µm2 in a cm2, final output in mA
                            #tet_currents[tet] += seg.ina * seg.area() * 1e-8 * fract
#                            tet_currents[tet] +=seg.ina * seg.area() * fract
                            tet_currents[tet] += seg.area() * fract
                            sum1=sum1+fract
                        print("sum1: ",sum1)    
                        continue
            fractmap = []

            # Store points in section
            pts = np.empty((npts, 3), dtype=float)

            for i in range(npts):
                pts[i] = np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]) * micrometer2meter

            # Transform to absolute coordinates (copy is necessary to get correct stride)
            pts_abs_coo = np.array(loc_2_glob_tsf(pts), dtype=float, order='C')

            # Update neuron bounding box
            n_bbox_min = np.minimum(np.amin(pts_abs_coo, axis=0), n_bbox_min)
            n_bbox_max = np.maximum(np.amax(pts_abs_coo, axis=0), n_bbox_max)

            # get tet to fraction map, batch processing all points (segments)
            tet2fraction_map = mesh.intersect(pts_abs_coo)

            # store map for each section
            secmap.append((sec, tet2fraction_map))

        neurSecmap.append(secmap)

    # Reduce Neuron bbox
    n_bbox_min_glo = np.empty((3), dtype=float)
    n_bbox_max_glo = np.empty((3), dtype=float)
    comm.Reduce([n_bbox_min, 3, MPI.DOUBLE], [n_bbox_min_glo, 3, MPI.DOUBLE], \
        op=MPI.MIN, root=0)
    comm.Reduce([n_bbox_max, 3, MPI.DOUBLE], [n_bbox_max_glo, 3, MPI.DOUBLE], \
        op=MPI.MAX, root=0)

    # Check bounding boxes
    if rank == 0:
        s_bbox_min = mesh.getBoundMin()
        s_bbox_max = mesh.getBoundMax()

        print("bounding box Neuron:", n_bbox_min_glo, n_bbox_max_glo)
        print("bounding box STEPS:", s_bbox_min, s_bbox_max)

        # Should add tolerance to check bounding box
        if np.less(n_bbox_min_glo, s_bbox_min).any() or np.greater(n_bbox_max_glo, s_bbox_max).any():
            logging.warning("STEPS mesh does not overlap with all neurons")

    return neurSecmap


##############################################
# Runtime
##############################################


if __name__ == "__main__":
    main()
    exit() # needed to avoid hanging

