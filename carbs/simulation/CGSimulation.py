from hoomd import *
from hoomd import md
import numpy as np

class CGSimulation:
    '''
    1-bead/bp CG simulation class for Cadnano designs
    '''

    def __init__(self):
        self.origami                 = None
        self.num_steps               = None
        self.ssDNA_harmonic_bond     = {'r0':None, 'k0':None}

        self.dsDNA_harmonic_bond     = {'r0':None, 'k0':None}

        self.dihedral1               = {'kappa':None, 'theta0':None}
        self.dihedral21              = {'kappa':None, 'theta0':None}
        self.dihedral22              = {'kappa':None, 'theta0':None}
        self.dihedral31              = {'kappa':None, 'theta0':None}
        self.dihedral32              = {'kappa':None, 'theta0':None}

        self.snapshot                = None

        self.particle_types          = ['backbone','sidechain','aux']
        self.bond_types              = ['backbone']
        self.dihedral_types          = ['dihedral1', \
                                        'dihedral21',\
                                        'dihedral22',\
                                        'dihedral31',\
                                        'dihedral32']
        #Rigid/soft bodies from Origami structure
        self.num_dsDNA_particles     = 0
        self.num_ssDNA_particles     = 0
        self.num_nucleotides         = self.num_dsDNA_particles + self.num_ssDNA_particles
        self.dsDNA_particles         = None
        self.ssDNA_particles         = None

    def parse_origami_from_pickle(self, pickle_file):
        '''
        Parse origami from pickle file
        '''
        import pickle
        f = open(pickle_file, 'rb')
        self.origami = pickle.load(f)
        f.close()

    def initialize_cg_md(self):
        '''
        Initialize relaxation protocol and switch to new simulation context
        '''
        context.initialize("");
        cg_sim = context.SimulationContext();
        cg_sim.set_current()

    def initialize_particles(self):
        '''
        Initialize particle positions, moment of inertia and velocities
        '''

        #Retrieve origami information
        oligos_list = self.origami.oligos_list

        bkbone_positions = [self.origami.get_nucleotide(pointer).position[1] \
             for chain in oligos_list for strand in chain for pointer in strand \
             if not self.origami.get_nucleotide(pointer).skip]

        nucl_quaternions = [self.origami.get_nucleotide(pointer).quaternion \
             for chain in oligos_list for strand in chain for pointer in strand \
             if not self.origami.get_nucleotide(pointer).skip]

        self.num_nucleotides = len(bkbone_positions)

        self.snapshot = data.make_snapshot(N             = self.num_nucleotides,
                                          box            = data.boxdim(Lx=120, Ly=120, Lz=300),
                                          particle_types =['backbone','sidechain','aux'],
                                          bond_types     = self.bond_types
                                          );

        self.snapshot.particles.position[:]       = bkbone_positions
        self.snapshot.particles.orientation[:]    = nucl_quaternions
        self.snapshot.particles.moment_inertia[:] = [[1., 1., 1.]]
        # self.snapshot.particles.typeid[:] = [0];

        #record particle types and update simulation_nucleotide_num
        i = 0
        for chain in oligos_list:
            for strand in chain:
                for pointer in strand:
                    if not self.origami.get_nucleotide_type(pointer).skip:
                        # self.snapshot.particles.typeid[i] = self.origami.get_nucleotide_type(pointer).type
                        self.origami.get_nucleotide(pointer).simulation_nucleotide_num = i
                        i += 1

        #random initial velocity
        self.snapshot.particles.velocity[:] = np.random.normal(0.0, np.sqrt(0.8 / 1.0), [self.snapshot.particles.N, 3]);

    def initialize_system(self):
        # Read the snapshot and create neighbor list
        self.system = init.read_snapshot(self.snapshot);
        self.nl     = md.nlist.cell();

    def create_adjacent_bonds(self):
        '''
        Create spring bonds between adjacent (neighboring) particles
        If neighbor is skip, find 'end' of skip region and add next bead as neighbor
        '''
        oligos_list = self.origami.oligos_list

        particle_sim_num = 0
        for chain in oligos_list:
            for strand in chain:
                strand_array = np.asarray(strand)

                this_bead = None
                next_bead = None
                for p in range(len(strand_array) - 1):
                    if not self.origami.get_nucleotide_type(strand_array[p]).skip:
                        this_bead = particle_sim_num
                        particle_sim_num += 1
                    if not self.origami.get_nucleotide_type(strand_array[p + 1]).skip:
                        next_bead = this_bead + 1
                    if this_bead != None and next_bead != None:
                        self.system.bonds.add(self.bond_types[0], this_bead, next_bead)
                        this_bead = None
                        next_bead = None
                # end of chain. next chain starts at another bead:
                particle_sim_num += 1

    def create_rigid_bonds(self):
        '''
        Create backbone-base and backbone-orthogonal rigid bonds in every particle
        '''
        oligos_list = self.origami.oligos_list
        pointer_0   = oligos_list[0][0][0]
        nucl_0      = self.origami.get_nucleotide(pointer_0)
        nucl_0_vecs = np.array(nucl_0.vectors_body_frame)

        rigid = md.constrain.rigid();

        rigid.set_param(self.particle_types[0], \
                        types=[self.particle_types[1], self.particle_types[2]], \
                        positions = [1.0*nucl_0_vecs[0], 1.0*nucl_0_vecs[2]]); #magic numbers. Check !!!

        rigid.create_bodies()

    def create_other_bonds(self):
        '''
        Create Watson-Crick spring as well as cross-spring to beads above and below
        '''
        #create bonds between watson-crick pairs
        watson_crick_connections = []
        cross_1_connections      = []
        cross_2_connections      = []
        dihedral_connections     = []

        for c, chain in enumerate(oligos_list):
            for s, strand in enumerate(chain):
                for p, pointer in enumerate(strand):
                    this_nucleotide         = self.origami.get_nucleotide(pointer)
                    this_nucleotide_sim_num = this_nucleotide.simulation_nucleotide_num
                    is_dsDNA                = self.origami.get_nucleotide_type(pointer).type
                    is_skip                 = self.origami.get_nucleotide_type(strand_array[p]).skip
                    if not is_dsDNA or is_skip : # all the following bonds are only relevant for dsDNA
                        continue

                    #1. calculate watson_crick pair and assign bonds for this nucleotide
                    [vh_0, index_0, is_fwd_0] = pointer
                    wc_pair_pointer           = [vh_0, index_0, 1 - is_fwd_0]
                    wc_pair_sim_num           = self.origami.get_nucleotide(wc_pair_pointer).simulation_nucleotide_num
                    conn_01 = [this_nucleotide_sim_num, wc_pair_sim_num]
                    conn_02 = [wc_pair_sim_num, this_nucleotide_sim_num]
                    if conn_01 and conn_02 not in watson_crick_connections:
                        watson_crick_connections.append(conn_01)

                    #2. calculate watson_crick for next nucleotide and assign bond
                    if p != len(strand) - 1:
                        next_nucleotide_pointer   = strand[p+1]
                        [vh_1, index_1, is_fwd_1] = next_nucleotide_pointer
                        next_nucleotide_sim_num = self.origami.get_nucleotide(next_nucleotide_pointer).simulation_nucleotide_num
                        is_dsDNA = self.origami.get_nucleotide_type(next_nucleotide_pointer).type
                        if not is_dsDNA: #TODO: handle skip !!
                            continue
                        next_pair_pointer         = [vh_1, index_1, 1 - is_fwd_1]
                        next_pair_sim_num         = self.origami.get_nucleotide(next_pair_pointer).simulation_nucleotide_num
                        conn_11 = [this_nucleotide_sim_num, next_pair_sim_num]
                        conn_12 = [next_pair_sim_num, this_nucleotide_sim_num]
                        if conn_11 and conn_12 not in cross_1_connections:
                            cross_1_connections.append(conn_11)

                        #3. Add dihedral bond
                        dihedral_particles = [wc_pair_sim_num, this_nucleotide_sim_num, next_nucleotide_sim_num, next_pair_sim_num]
                        dihedral_connections.append(dihedral_particles)

                    #4. calculate watson_crick for previous nucleotide and assign bond
                    if p > 0:
                        previous_nucleotide_pointer = strand[p-1]
                        [vh_2, index_2, is_fwd_2]   = previous_nucleotide_pointer
                        is_dsDNA = self.origami.get_nucleotide_type(previous_nucleotide_pointer).type
                        if not is_dsDNA: #TODO: handle skip !!
                            continue
                        previous_pair_pointer       = [vh_2, index_2, 1 - is_fwd_2]
                        previous_pair_sim_num       = self.origami.get_nucleotide(previous_pair_pointer).simulation_nucleotide_num
                        conn_21 = [this_nucleotide_sim_num, previous_pair_sim_num]
                        conn_22 = [previous_pair_sim_num, this_nucleotide_sim_num]
                        if conn_21 and conn_22 not in cross_2_connections:
                            cross_2_connections.append(conn_21)



        for wc_conn in watson_crick_connections:
            self.system.bonds.add('watson_crick', wc_conn[0], wc_conn[1])

        for x1_conn in cross_1_connections:
            self.system.bonds.add('cross_1', x1_conn[0], x1_conn[1])

        for x2_conn in cross_2_connections:
            self.system.bonds.add('cross_2', x2_conn[0], x2_conn[1])

        for dih in dihedral_connections:
            self.system.dihedrals.add('dihedral', dih[0], dih[1], dih[2], dih[3])

    def set_harmonic_bonds(self):
        '''
        Set harmonic bonds
        '''
        self.harmonic = md.bond.harmonic()
        self.harmonic.bond_coeff.set('backbone', k=150.0 , r0=0.75);

    def set_dihedral_bonds(self):
        '''
        set dihedral bonds
        '''
        #define harmonic angular bond
        def harmonic_angle(theta, kappa, theta0):
            V = 0.5 * kappa * (theta - theta0)**2
            F = - kappa * (theta - theta0)
            return(V, F)

        dtable = md.dihedral.table(width = 1000)
        dtable.dihedral_coeff.set('dihedral', func=harmonic_angle, coeff=dict(kappa=80, theta0=-0.28))

    def set_wca_potentials(self):
        '''
        Set WCA potentials
        '''
        wca = md.pair.lj(r_cut=2.0**(1/6), nlist=self.nl)
        wca.set_params(mode='shift')
        wca.pair_coeff.set(self.particle_types, self.particle_types, epsilon=1.0, sigma=0.75, r_cut=0.75*2**(1/6))

        ########## INTEGRATION ############
        md.integrate.mode_standard(dt=0.003);
        rigid = group.rigid_center();
        md.integrate.langevin(group=rigid, kT=0.2, seed=42);

    def fix_diameters(self):
        for i in range(0, self.num_nucleotides):
            self.system.particles[i].diameter = 0.75
        for i in range(self.num_nucleotides, len(self.system.particles), 2):
            self.system.particles[i].diameter = 0.5
            self.system.particles[i + 1].diameter = 0.1

    def dump_settings(self,output_fname,period):
        '''
        Dump settings
        '''
        dump.gsd(output_fname,
                       period=period,
                       group=group.all(),
                       static=[],
                       overwrite=True);

    def run(self,num_steps=1e6):
        run(num_steps)
