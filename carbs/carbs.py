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
    FILENAME = 'PFD_tripod_2017_no_ssDNA'
    INPUT_FILENAME    = '../../cadnano-files/' + FILENAME +'.json'
    OUTPUT_FILENAME_1 = '../../cadnano-files/carbs_output/' + FILENAME +'_RB.gsd'
    OUTPUT_FILENAME_2 = '../../cadnano-files/carbs_output/' + FILENAME +'_CG.gsd'
    PICKLE_FILE       = 'data/' + FILENAME + '.pckl'

    RELAX = True

    doc.readFile(INPUT_FILENAME);

    #Parse the structure for simulation
    new_origami      = origami.Origami()
    new_origami.part = doc.activePart()
    new_origami.initialize_oligos()
    new_origami.initialize_nucleotide_matrix()
    new_origami.populate_skips_inserts_matrix()
    new_origami.populate_nucleotide_matrix()
    new_origami.get_connections()
    new_origami.assign_nucleotide_types()
    new_origami.assign_nucleotide_connections()
    new_origami.cluster_into_bodies()
    new_origami.parse_skip_connections()
    new_origami.calculate_next_nucleotide()
    new_origami.calculate_nucleotide_quaternions()
    new_origami.update_oligos_list()

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
      relax_simulation.dump_settings(OUTPUT_FILENAME_1, 1e3)
      relax_simulation.run(1e6)
      relax_simulation.update_positions_and_quaternions()
      relax_simulation.save_to_pickle(PICKLE_FILE)


    #Start coarse-grained simulation
    elif RELAX == False:
      cg_simulation = CGSimulation.CGSimulation()
      cg_simulation.parse_origami_from_pickle(PICKLE_FILE)
      cg_simulation.initialize_cg_md()
      cg_simulation.initialize_particles()
      cg_simulation.initialize_system()

      cg_simulation.create_rigid_bonds()
      cg_simulation.create_dihedral_bonds()
      cg_simulation.create_watson_crick_bonds()

      cg_simulation.create_adjacent_bonds()
      cg_simulation.set_harmonic_bonds()
      cg_simulation.set_dihedral_bonds()

      cg_simulation.set_wca_potentials()
      cg_simulation.fix_diameters()
      cg_simulation.dump_settings(OUTPUT_FILENAME_2, 10000)
      cg_simulation.integration()



if __name__ == "__main__":
  main()
