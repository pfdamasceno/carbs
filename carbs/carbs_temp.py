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
INPUT_FILENAME    = '../../cadnano-files/PFD_6hb_skip.json'
OUTPUT_FILENAME_1 = '../../cadnano-files/carbs_output/PFD_6hb_skip_rigid.gsd'
OUTPUT_FILENAME_2 = '../../cadnano-files/carbs_output/PFD_6hb_skip_CG.gsd'
PICKLE_FILE       = 'data/origami_relaxed.pckl'

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

cg_simulation = CGSimulation.CGSimulation()
cg_simulation.parse_origami_from_pickle(PICKLE_FILE)
cg_simulation.initialize_cg_md()
cg_simulation.initialize_particles()
cg_simulation.initialize_system()

cg_simulation.create_adjacent_bonds()

cg_simulation.create_rigid_bonds()
cg_simulation.set_harmonic_bonds()
# cg_simulation.set_dihedral_bonds()
cg_simulation.set_wca_potentials()
cg_simulation.fix_diameters()
cg_simulation.dump_settings(OUTPUT_FILENAME_2, 1)
cg_simulation.run(2)
