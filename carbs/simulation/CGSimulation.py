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
        self.bond_types              = ['backbone', 'watson_crick']
        self.angle_types             = ['axis']
        self.dihedral_types          = ['dihedral1', \
                                        'dihedral2',\
                                        'dihedral3',\
                                        'dihedral4',\
                                        'dihedral5']
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
                                          box            = data.boxdim(Lx=100, Ly=100, Lz=100),
                                          particle_types =['backbone','sidechain','aux'],
                                          bond_types     = self.bond_types,
                                          angle_types    = self.angle_types,
                                          dihedral_types = self.dihedral_types
                                          );

        self.snapshot.particles.position[:]       = bkbone_positions
        self.snapshot.particles.orientation[:]    = nucl_quaternions
        self.snapshot.particles.moment_inertia[:] = [[100., 100., 100.]]
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

    def create_dihedral_bonds(self):
        '''
        Create dihedral bonds between beads
        '''
        oligos_list              = self.origami.oligos_list
        num_backbones            = self.num_nucleotides
        index_1st_nucl_in_strand = 0

        for c, chain in enumerate(oligos_list):
            for s, strand in enumerate(oligos_list[c]):
                this_bead        = None
                next_bead        = None
                non_skip_counter = 0
                for n, nucl in enumerate(oligos_list[c][s]):

                    pointer_1 = self.origami.oligos_list_to_nucleotide_info(c,s,n)
                    # test whether 1st nucleotide is not skip
                    if not self.origami.get_nucleotide_type(pointer_1).skip:

                        bckb_1 = index_1st_nucl_in_strand + non_skip_counter
                        base_1 = num_backbones + 2*index_1st_nucl_in_strand + 2*non_skip_counter
                        orth_1 = base_1 + 1

                        nucl_1 = self.origami.get_nucleotide(pointer_1)
                        nucl_1.vectors_simulation_nums = [base_1, orth_1]

                        # test if end of chain
                        if n == len(oligos_list[c][s]) - 1:
                            non_skip_counter += 1
                            continue

                        this_bead = nucl_1.simulation_nucleotide_num
                        non_skip_counter += 1

                    # now try to find the nucleotide the first bead connects to
                    pointer_2 = self.origami.oligos_list_to_nucleotide_info(c,s,n+1)

                    # test whether 2nd nucleotide is not skip
                    if not self.origami.get_nucleotide_type(pointer_2).skip:
                        nucl_2    = self.origami.get_nucleotide(pointer_2)
                        next_bead = nucl_2.simulation_nucleotide_num

                        bckb_2 = index_1st_nucl_in_strand + non_skip_counter
                        base_2 = num_backbones + 2*index_1st_nucl_in_strand + 2*non_skip_counter
                        orth_2 = base_2 + 1

                    # test whether we found 1st and 2nd non-skip nucleotides
                    if this_bead != None and next_bead != None:
                        self.system.dihedrals.add('dihedral1', orth_1, bckb_1, base_1, base_2)
                        self.system.dihedrals.add('dihedral2', bckb_1, base_1, base_2, bckb_2)
                        self.system.dihedrals.add('dihedral3', base_1, bckb_1, orth_1, bckb_2)
                        self.system.dihedrals.add('dihedral4', orth_1, bckb_1, base_1, bckb_2)
                        self.system.dihedrals.add('dihedral5', base_1, bckb_1, orth_1, base_2)
                        this_bead = None
                        next_bead = None

                index_1st_nucl_in_strand += non_skip_counter

    def create_watson_crick_bonds(self):
        '''
        Create bonds between axial particles in dsDNA
        '''

        for vh in range(len(self.origami.nucleotide_matrix)):
            for idx in range(len(self.origami.nucleotide_matrix[vh])):
                for is_fwd in range(2):

                    # check if nucleotide exists
                    if self.origami.nucleotide_matrix[vh][idx][is_fwd] is None:
                        continue

                    # check if single stranded DNA
                    is_dsDNA = self.origami.nucleotide_type_matrix[vh][idx].type
                    if is_dsDNA == False:
                        continue

                    nucleotide = self.origami.nucleotide_matrix[vh][idx][is_fwd]
                    if nucleotide.skip == True:
                        continue

                    nucl_1 = self.origami.nucleotide_matrix[vh][idx][is_fwd]
                    nucl_2 = self.origami.nucleotide_matrix[vh][idx][1 - is_fwd]
                    base_1 = nucl_1.vectors_simulation_nums[0]
                    base_2 = nucl_2.vectors_simulation_nums[0]

                    self.system.bonds.add(self.bond_types[1], base_1, base_2)

    def create_adjacent_bonds(self):
        '''
        Create spring bonds between adjacent (neighboring) particles
        If neighbor is skip, find 'end' of skip region and add next bead as neighbor
        '''

        # wrong! axis particles dont make bonds across strands !

        oligos_list = self.origami.oligos_list
        for o, oligo in enumerate(oligos_list):
            for s, strand in enumerate(oligos_list[o]):
                for p, pointer in enumerate(oligos_list[o][s]):
                    this_nucleotide = self.origami.get_nucleotide(pointer)

                    #test for end of oligo
                    if this_nucleotide.oligo_end == True or \
                       this_nucleotide.skip == True:
                        continue

                    #else make bonds with next neighbor
                    next_nucleotide = this_nucleotide.next_nucleotide
                    this_sim_num    = this_nucleotide.simulation_nucleotide_num
                    next_sim_num    = next_nucleotide.simulation_nucleotide_num
                    self.system.bonds.add(self.bond_types[0], this_sim_num, next_sim_num)

                    # if this and next nucleotides are not end
                    if this_nucleotide.strand_end == False and \
                       next_nucleotide.strand_end == False:
                        third_nucleotide = next_nucleotide.next_nucleotide

                        this_axis_sim_num  = this_nucleotide.vectors_simulation_nums[0]
                        next_axis_sim_num  = next_nucleotide.vectors_simulation_nums[0]
                        third_axis_sim_num = third_nucleotide.vectors_simulation_nums[0]

                        self.system.angles.add(self.angle_types[0], this_axis_sim_num, next_axis_sim_num, third_axis_sim_num)

    def set_harmonic_bonds(self):
        '''
        Set harmonic bonds
        '''
        self.harmonic = md.bond.harmonic()
        self.harmonic.bond_coeff.set('backbone'    , k=20.0 , r0=0.75);
        self.harmonic.bond_coeff.set('watson_crick', k=500.0 , r0=0.0);

        self.angle_harmonic = md.angle.harmonic()
        self.angle_harmonic.angle_coeff.set('axis', k=20, t0=3.1415, r=0.25)

    def set_dihedral_bonds(self):
        '''
        set dihedral bonds
        '''
        #define harmonic angular bond
        def harmonic_angle(theta, kappa, theta0):
            V = 0.5 * kappa * (theta - theta0)**2
            F = - kappa * (theta - theta0)
            return(V, F)

        dtable = md.dihedral.table(width=1000)
        dtable.dihedral_coeff.set('dihedral1', func=harmonic_angle, coeff=dict(kappa=1, theta0=-1.571)) #+Pi: why?
        dtable.dihedral_coeff.set('dihedral2', func=harmonic_angle, coeff=dict(kappa=1, theta0=-0.598))
        dtable.dihedral_coeff.set('dihedral3', func=harmonic_angle, coeff=dict(kappa=10, theta0=+0.559))
        dtable.dihedral_coeff.set('dihedral4', func=harmonic_angle, coeff=dict(kappa=1, theta0=+0.317 - np.pi)) #-Pi: why?
        dtable.dihedral_coeff.set('dihedral5', func=harmonic_angle, coeff=dict(kappa=1, theta0=+0.280))


    def set_wca_potentials(self):
        '''
        Set WCA potentials
        '''
        wca = md.pair.lj(r_cut=2.0**(1/6), nlist=self.nl)
        wca.set_params(mode='shift')

        wca.pair_coeff.set('backbone', 'backbone',   epsilon=10.0, sigma=0.75, r_cut=0.75*2**(1/6))
        wca.pair_coeff.set('backbone', 'sidechain',  epsilon=0.0, sigma=0.375, r_cut=0.375*2**(1/6))
        wca.pair_coeff.set('sidechain', 'sidechain', epsilon=0.0, sigma=0.100, r_cut=0.4)

        wca.pair_coeff.set('backbone', 'aux',        epsilon=0.0, sigma=1.000, r_cut=1.000*2**(1/6))
        wca.pair_coeff.set('aux', 'sidechain',       epsilon=0.0, sigma=1.000, r_cut=1.000*2**(1/6))
        wca.pair_coeff.set('aux', 'aux',             epsilon=0.0, sigma=1.000, r_cut=1.000*2**(1/6))


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

    def integration(self):
        ########## INTEGRATION ############
        md.integrate.mode_standard(dt=0.001);
        rigid = group.rigid_center();
        md.integrate.langevin(group=rigid, kT=0.01, seed=42);

    def run(self,num_steps=1e6):
        run(num_steps)
