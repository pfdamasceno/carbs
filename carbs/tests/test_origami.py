import sys
sys.path.append("..")

import cadnano
from origami import origami
from simulation import CGSimulation

# unit tests for init.create_random
class origami_test (unittest.TestCase):
    def setUp(self):
        #tbw
