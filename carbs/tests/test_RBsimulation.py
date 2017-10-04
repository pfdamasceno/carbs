import sys
sys.path.append("..")

import unittest
import os
import cadnano
from cadnano.document import Document

from origami import origami
from simulation import RBSimulation

# unit tests for init.create_random
class rigid_body_simulation_test (unittest.TestCase):
    def test_pickle_creation(self):
        #Initialize cadnano
        app = cadnano.app()
        doc = app.document = Document()

        INPUT_FILENAME    = 'data/skip_2hb.test.json'
        PICKLE_FILE       = 'data/skip_2hb.test.pckl'

        assert os.path.exists(INPUT_FILENAME) == True

        if os.path.exists(PICKLE_FILE):
            os.remove(PICKLE_FILE)

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

        relax_simulation         = RBSimulation.RigidBodySimulation()
        relax_simulation.origami = new_origami
        relax_simulation.initialize_relax_md()
        relax_simulation.initialize_particles()
        relax_simulation.create_rigid_bodies()
        relax_simulation.create_bonds()
        relax_simulation.set_initial_harmonic_bonds()
        relax_simulation.set_lj_potentials()
        relax_simulation.run(1e1)
        relax_simulation.update_positions()
        relax_simulation.save_to_pickle(PICKLE_FILE)

        assert os.path.exists(PICKLE_FILE) == True

if __name__ == '__main__':
    unittest.main()
