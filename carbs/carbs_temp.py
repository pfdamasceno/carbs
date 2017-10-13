import numpy as np
import cadnano
import functools

from cadnano.document import Document

from utils import vectortools
from origami import origami
from simulation import CGSimulation
from simulation import RBSimulation

app = cadnano.app()
doc = app.document = Document()
INPUT_FILENAME    = '../../cadnano-files/PFD_6hb.json'
OUTPUT_FILENAME_1 = '../../cadnano-files/carbs_output/PFD_6hb_rigid.gsd'
OUTPUT_FILENAME_2 = '../../cadnano-files/carbs_output/PFD_6hb_CG.gsd'
PICKLE_FILE       = 'data/origami_relaxed.pckl'

doc.readFile(INPUT_FILENAME);

cg_simulation = CGSimulation.CGSimulation()
cg_simulation.parse_origami_from_pickle(PICKLE_FILE)
cg_simulation.initialize_cg_md()
cg_simulation.initialize_particles()
cg_simulation.initialize_system()

cg_simulation.create_adjacent_bonds()

cg_simulation.create_rigid_bonds()
cg_simulation.create_dihedral_bonds()
cg_simulation.create_base_bonds()
cg_simulation.set_harmonic_bonds()
cg_simulation.set_dihedral_bonds()

cg_simulation.set_wca_potentials()
cg_simulation.integration()
cg_simulation.fix_diameters()
cg_simulation.dump_settings(OUTPUT_FILENAME_2, 1)




cg_simulation.run(100000)
