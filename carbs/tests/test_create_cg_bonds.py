import sys
sys.path.append("..")

import unittest
import os
import cadnano
from cadnano.document import Document

from origami import origami
from simulation import CGSimulation

# unit tests for init.create_random
class create_CG_bonds_test (unittest.TestCase):
    def test_pickle_exists(self):
        PICKLE_FILE = 'data/skip_2hb.test.pckl'
        assert os.path.exists(PICKLE_FILE) == True

    def test_create_adjacent_bonds_with_skip(self):
        PICKLE_FILE = '/Users/damasceno/Documents/1_work/2_codes/carbs/carbs/tests/data/skip_2hb.test.pckl'
        app = cadnano.app()
        doc = app.document = Document()

        cg_simulation = CGSimulation.CGSimulation()
        cg_simulation.parse_origami_from_pickle(PICKLE_FILE)
        cg_simulation.initialize_cg_md()
        cg_simulation.initialize_particles()

        cg_simulation.initialize_system()

        cg_simulation.create_adjacent_bonds()
        assert len(cg_simulation.system.bonds) == 110

if __name__ == '__main__':
    unittest.main()
