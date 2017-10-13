import numpy as np
import cadnano
import functools

from cadnano.document import Document

from utils import vectortools
from origami import origami
from simulation import CGSimulation
from simulation import RBSimulation

def main():
    #Initialize cadnano
    app = cadnano.app()
    doc = app.document = Document()
    INPUT_FILENAME    = '../../cadnano-files/PFD_6hb.json'
    OUTPUT_FILENAME_1 = '../../cadnano-files/carbs_output/PFD_6hb_rigid.gsd'
    OUTPUT_FILENAME_2 = '../../cadnano-files/carbs_output/PFD_6hb_CG.gsd'
    PICKLE_FILE       = 'data/origami_relaxed.pckl'

    RELAX = False

    doc.readFile(INPUT_FILENAME);

    #Parse the structure for simulation
    new_origami      = origami.Origami()
    new_origami.part = doc.activePart()
    new_origami.list_oligos()
    new_origami.initialize_nucleotide_matrix()
    new_origami.find_skips()
    new_origami.create_oligos_list()
    new_origami.get_connections()
    new_origami.assign_nucleotide_types()
    new_origami.incorporate_skips()
    new_origami.assign_nucleotide_connections()
    new_origami.cluster_into_bodies()
    new_origami.parse_skip_connections()
    new_origami.calculate_quaternions_from_positions()

    #Start relaxation simulation
    if RELAX == True:
        relax_simulation         = RBSimulation.RigidBodySimulation()
        relax_simulation.origami = new_origami
        relax_simulation.initialize_relax_md()
        relax_simulation.initialize_particles()
        relax_simulation.create_rigid_bodies()
        relax_simulation.create_bonds()
        relax_simulation.set_initial_harmonic_bonds()
        relax_simulation.set_lj_potentials()
        relax_simulation.dump_settings(OUTPUT_FILENAME_1, 1)
        relax_simulation.run(1)
        relax_simulation.update_positions_and_quaternions()
        relax_simulation.save_to_pickle(PICKLE_FILE)

    #Start coarse-grained simulation
    elif RELAX == False:
        cg_simulation = CGSimulation.CGSimulation()
        cg_simulation.parse_origami_from_pickle(PICKLE_FILE)
        cg_simulation.initialize_cg_md()
        cg_simulation.initialize_particles()
        cg_simulation.initialize_system()
        cg_simulation.create_adjacent_bonds()
        cg_simulation.set_harmonic_bonds()
        cg_simulation.set_dihedral_bonds()
        cg_simulation.set_lj_potentials()
        cg_simulation.dump_settings(OUTPUT_FILENAME_2, 1)
        cg_simulation.run(2)

if __name__ == "__main__":
  main()
