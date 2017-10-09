from hoomd import *
from hoomd import md
import numpy as np

class RigidBodySimulation:
    '''
    Rigid body simulation class for Cadnano designs
    '''

    def __init__(self):
        self.origami                 = None
        self.num_steps               = None
        self.ssDNA_harmonic_bond     = {'r0':None, 'k0':None}
        self.ssDNA_harmonic_angle    = {'a0':None, 'k0':None}

        self.dsDNA_harmonic_bond     = {'r0':None, 'k0':None}
        self.dsDNA_harmonic_angle    = {'a0':None, 'k0':None}

        self.bodies_comass_positions = []
        self.bodies_moment_inertia   = []

        self.snapshot                = None

        self.body_types              = []
        self.bond_types              = []

        #Rigid/soft bodies from Origami structure
        self.num_rigid_bodies        = 0
        self.num_soft_bodies         = 0
        self.rigid_bodies            = None
        self.soft_bodies             = None

    def initialize_relax_md(self):
        '''
        Initialize relaxation protocol
        '''
        context.initialize("");
        relax_sim = context.SimulationContext();
        relax_sim.set_current()

    def initialize_particles(self):
        '''
        Initialize particle positions, moment of inertia and velocities
        '''

        #Retrieve origami rigid body information
        self.rigid_bodies = self.origami.rigid_bodies
        self.soft_bodies  = self.origami.soft_bodies

        self.num_rigid_bodies = len(self.rigid_bodies)
        self.num_soft_bodies  = len(self.soft_bodies)

        self.rigid_bodies_comass_positions  = [body.comass_position for body in self.rigid_bodies]
        self.center_of_mass                 = np.average(np.asarray(self.rigid_bodies_comass_positions)[:,:3], axis=0)

        self.rigid_bodies_comass_positions -= self.center_of_mass
        self.rigid_bodies_moment_inertia    = [body.moment_inertia for body in self.rigid_bodies]

        if self.num_soft_bodies > 0:
            self.soft_bodies_comass_positions  = [body.comass_position for body in self.soft_bodies]
            self.soft_bodies_comass_positions -= self.center_of_mass
            self.soft_bodies_moment_inertia    = [body.moment_inertia for body in self.soft_bodies]

        self.body_types  = ["rigid_body"+"_"+str(i) for i in range(self.num_rigid_bodies)]
        self.body_types += ["nucleotides"]

        self.snapshot = data.make_snapshot(N = self.num_rigid_bodies + self.num_soft_bodies,
                                          box = data.boxdim(Lx=120, Ly=120, Lz=300),
                                          particle_types = self.body_types,
                                          bond_types = ['interbody']);

        if self.num_soft_bodies > 0:
            self.snapshot.particles.position[:]       = np.vstack((self.rigid_bodies_comass_positions, self.soft_bodies_comass_positions))
            self.snapshot.particles.moment_inertia[:] = np.vstack((self.rigid_bodies_moment_inertia  , self.soft_bodies_moment_inertia))
        else:
            self.snapshot.particles.position[:]       = np.vstack((self.rigid_bodies_comass_positions))
            self.snapshot.particles.moment_inertia[:] = np.vstack((self.rigid_bodies_moment_inertia))

        #particle types
        for i in range(self.num_rigid_bodies):
            self.snapshot.particles.typeid[i] = i

        #particle types
        for i in range(self.num_rigid_bodies,self.num_rigid_bodies+self.num_soft_bodies):
            self.snapshot.particles.typeid[i] = self.num_rigid_bodies

        self.snapshot.particles.velocity[:] = np.random.normal(0.0, np.sqrt(0.8 / 1.0), [self.snapshot.particles.N, 3]);

    def create_rigid_bodies(self):
        # Read the snapshot and create neighbor list
        self.system = init.read_snapshot(self.snapshot);
        self.nl     = md.nlist.stencil();

        # Create rigid particles
        self.rigid = md.constrain.rigid();
        for b, body in enumerate(self.rigid_bodies):
            body_type            = self.body_types[b]

            nucleotide_positions = [nucleotide.position[1] for nucleotide in body.nucleotides]
            #move particles to body reference frame
            nucleotide_positions -= body.comass_position
            self.rigid.set_param(body_type, \
                        types=['nucleotides']*len(nucleotide_positions), \
                        positions = nucleotide_positions);

        self.rigid.create_bodies()

    def create_bonds(self):
        '''
        Create interbody bonds
        '''
        self.nucleotide_bonds = self.origami.inter_nucleotide_connections

        for connection in self.nucleotide_bonds:
            # delta is needed because the 1st n bodies will be the com of the rigid blocks
            delta = self.num_rigid_bodies
            nucleotide_num_1, nucleotide_num_2 = connection
            self.system.bonds.add('interbody', delta + nucleotide_num_1, delta + nucleotide_num_2)

    def set_initial_harmonic_bonds(self):
        '''
        Set harmonic bonds
        '''
        self.harmonic = md.bond.harmonic()
        self.harmonic.bond_coeff.set('interbody', k=0.1 , r0=0.5);

        # fix diameters for vizualization
        for i in range(0, self.num_rigid_bodies):
            self.system.particles[i].diameter = 2.0
        for i in range(self.num_rigid_bodies, len(self.system.particles)):
            self.system.particles[i].diameter = 0.5

    def set_harmonic_bonds(self, k_spring):
        '''
        Set harmonic bonds
        '''
        self.harmonic.bond_coeff.set('interbody', k=k_spring , r0=0.5);

    def set_lj_potentials(self):
        '''
        Set LJ potentials
        '''
        wca = md.pair.lj(r_cut=2.0**(1/6), nlist=self.nl)
        wca.set_params(mode='shift')
        wca.pair_coeff.set(self.body_types, self.body_types, epsilon=1.0, sigma=1.0, r_cut=1.0*2**(1/6))

        ########## INTEGRATION ############
        md.integrate.mode_standard(dt=0.001, aniso=True);
        rigid     = group.rigid_center();
        non_rigid = group.nonrigid()
        combined  = group.union('combined',rigid,non_rigid)
        md.integrate.langevin(group=combined, kT=0.4, seed=42);

    def dump_settings(self,output_fname,period):
        '''
        Dump settings
        '''
        dump.gsd(output_fname,
                       period = period,
                       group = group.all(),
                       static = [],
                       overwrite = True);

    def run(self,num_steps):
        run(num_steps)
        # now run with stronger spring for another 10,000 steps
        self.set_harmonic_bonds(10.)
        run(100000)

    def update_positions_and_quaternions(self):
        '''
        update all particles positions (to be used after relaxation)
        '''
        # delta is needed because the 1st n bodies will be the com of the rigid blocks
        delta = self.num_rigid_bodies
        for vh in range(len(self.origami.nucleotide_matrix)):
            for idx in range(len(self.origami.nucleotide_matrix[vh])):
                for is_fwd in range(2):
                    nucleotide = self.origami.nucleotide_matrix[vh][idx][is_fwd]
                    if nucleotide != None:
                        simulation_num = delta + nucleotide.simulation_nucleotide_num
                        nucleotide.position[1] = self.system.particles[simulation_num].position
                        nucleotide.quaternion  = self.system.particles[simulation_num].orientation


    def save_to_pickle(self, filename):
        '''
        save origami to pickle file that can be reloaded later
        '''
        import pickle
        sys.setrecursionlimit(100000) #needed to export highly recursive object
        origami = self.origami
        with open(filename, 'wb') as f:
            pickle.dump(origami, f)
