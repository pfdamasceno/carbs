import numpy as np
import cadnano
import functools
import sys
import copy

from cadnano.document import Document

from utils import vectortools

class Origami:
    '''
    Parent DNA origami model class
    '''

    def __init__(self):
        #Body variables
        self.rigid_bodies                  = []
        self.soft_bodies                   = []

        self.rigid_body_nucleotide_types   = []
        self.rigid_body_nucleotides        = []

        self.soft_body_nucleotide_types    = []
        self.soft_body_nucleotides         = []

        self.num_rigid_bodies              = 0
        self.num_soft_bodies               = 0

        #Soft connections
        self.inter_rigid_body_connections  = []
        self.inter_nucleotide_connections  = []

        #Cadnano parameters
        self.part                          = None
        self.oligos_list                   = None
        self.num_vhs                       = None
        self.num_bases                     = None
        self.nucleotide_matrix             = None   # This is the 'global' information about every nucleotide
        self.insert_update_dict            = {}     # dict of old -> new pointers to be used when updating nucleotides in origami w/ inserts

        self.nucleotide_type_list          = None
        self.nucleotide_type_matrix        = None

        self.crossovers                    = None
        self.vh_vh_crossovers              = None   # Given vh1, vh2, vh_vh_crossovers[vh_1][vh_2] is the number of xovers between them
        self.long_range_connections        = {}     # Dict connecting pointer_1 (vh, index, is_fwd) to pointer_2
        self.short_range_connections       = {}     # Dict connecting pointer_1 (vh, index, is_fwd) to pointer_2
        self.skip_connections              = {}     # Dict of pointers referring to nucleotides separated by skip

        #Distance constraints
        self.crossover_distance            = 2.0   # Distance in Angstrom

    def initialize_oligos(self):
        '''
        Given a origami part
        Return an array with all oligos in part sorted by length
        '''
        self.oligos_list = []

        cadnano_oligos_list = self.part.oligos()
        cadnano_oligos_list = sorted(cadnano_oligos_list, key=lambda x: x.length(), reverse=True)
        for cadnano_oligo in cadnano_oligos_list:
            new_oligo               = Oligo()
            new_oligo.cadnano_oligo = cadnano_oligo
            new_oligo.is_circular   = cadnano_oligo.isCircular()

            generator               = cadnano_oligo.strand5p().generator3pStrand()
            # get 5' end and add to start_pointer in oligo
            first_strand             = next(generator)
            new_oligo.start_pointer  = [first_strand.idNum(), first_strand.idx5Prime(), first_strand.isForward()]
            # restart generator
            generator               = cadnano_oligo.strand5p().generator3pStrand()

            for cadnano_strand in generator:
                new_strand = Strand()

                new_strand.vh        = cadnano_strand.idNum()
                new_strand.index_5p  = cadnano_strand.idx5Prime()
                new_strand.index_3p  = cadnano_strand.idx3Prime()
                new_strand.is_fwd    = cadnano_strand.isForward()

                direction = (-1 + 2*cadnano_strand.isForward()) #-1 if backwards, 1 if fwd
                # add nucleotide pointers to strand
                new_strand.pointers_list = []
                for i in range(new_strand.index_5p, new_strand.index_3p + direction, direction):
                    new_strand.pointers_list.append([new_strand.vh, i, int(new_strand.is_fwd)])
                new_oligo.strands_list.append(new_strand)

            self.oligos_list.append(new_oligo)

    def initialize_nucleotide_matrix(self):
        '''
        Creates an empty matrix of len = vh_length x index_length
        to be populated with all nucleotides in part (fwd and rev)
        It has the form nucleotide_matrix[vh][index][rev or fwd]
        '''
        self.num_vhs                 = len(list(self.part.getIdNums()))
        self.num_bases               = self.part.getVirtualHelix(0).getSize()

        self.nucleotide_matrix       = [[[None, None]  for idx in range(self.num_bases)] for vh in range(self.num_vhs)]
        self.nucleotide_type_matrix  = [[ None  for idx in range(self.num_bases)] for vh in range(self.num_vhs)]
        self.vh_vh_crossovers        = [[0  for vh in range(self.num_vhs)] for vh in range(self.num_vhs)]
        self.skip_matrix             = [[[False, False]  for idx in range(self.num_bases)] for vh in range(self.num_vhs)]
        self.insert_matrix           = [[[None, None]  for idx in range(self.num_bases)] for vh in range(self.num_vhs)]

    def populate_skips_inserts_matrix(self):
        '''
        Identify all the skips in the structure
        '''
        for oligo in self.oligos_list:
            cadnano_oligo = oligo.cadnano_oligo
            generator  = cadnano_oligo.strand5p().generator3pStrand()
            for cadnano_strand in generator:
                vh = cadnano_strand.idNum()
                index_5p  = cadnano_strand.idx5Prime()
                index_3p  = cadnano_strand.idx3Prime()
                direction = (-1 + 2*cadnano_strand.isForward()) #-1 if backwards, 1 if fwd
                for idx in range(index_5p, index_3p + direction, direction):
                    if cadnano_strand.hasInsertionAt(idx):
                        insertion_length = cadnano_strand.insertionLengthBetweenIdxs(idx,idx)
                        if  insertion_length == -1:
                            self.skip_matrix[vh][idx][int(cadnano_strand.isForward())] = True
                        else:
                            self.insert_matrix[vh][idx][int(cadnano_strand.isForward())] = insertion_length

    def populate_nucleotide_matrix(self):
        '''
        Populate nucleotide_matrix following oligos,
        then populate it for non-used (vh, index) locations
        in case we need them for inserts
        '''

        for o, oligo in enumerate(self.oligos_list):
            for s, strand in enumerate(oligo.strands_list):
                for p, pointer in enumerate(strand.pointers_list):
                    if p < len(strand.pointers_list) - 1:
                        pointer_1 = pointer
                        pointer_2 = strand.pointers_list[p + 1]

                        #Assign short range connection
                        self.short_range_connections[tuple(pointer_1)] = tuple(pointer_2)

                    #Get the coordinates
                    [vh, index, is_fwd] = [pointer[0], pointer[1], pointer[2]]
                    direction = -(-1)**(is_fwd)
                    coordinates = self.get_coordinates(vh, index)

                    new_nucleotide = Nucleotide()
                    new_nucleotide.direction         = direction
                    new_nucleotide.index             = index
                    new_nucleotide.position          = [coordinates[2], coordinates[is_fwd]] #Sidechain(axis) and backbone coordinates
                    new_nucleotide.strand            = strand
                    new_nucleotide.vh                = vh
                    new_nucleotide.is_fwd            = is_fwd
                    new_nucleotide.skip              = self.skip_matrix[vh][index][is_fwd]
                    new_nucleotide.pointer           = [vh, index, is_fwd]
                    new_nucleotide.oligo             = oligo

                    if p == len(strand.pointers_list) - 1:
                        new_nucleotide.is_strand_end = True
                        if s == len(oligo.strands_list) - 1:
                            new_nucleotide.is_oligo_end = True

                    #check if end or beginning of canvas
                    if index == self.num_bases - 1:
                        new_nucleotide.is_canvas_end = True
                    if index == 0:
                        new_nucleotide.is_canvas_start  = True

                    #check if start of the oligo
                    if oligo.start_pointer == new_nucleotide.pointer:
                        new_nucleotide.is_oligo_start = True

                    #Assign the nucleotide
                    self.nucleotide_matrix[vh][index][is_fwd] = new_nucleotide

        # For unused nucleotides, assign their coordinates in case they are needed for insertions
        for vh in range(self.num_vhs):
            for index in range(self.num_bases):
                for is_fwd in range(2):
                    if self.nucleotide_matrix[vh][index][is_fwd] == None:
                        direction = -(-1)**(is_fwd)
                        coordinates = self.get_coordinates(vh, index)

                        new_nucleotide = Nucleotide()
                        new_nucleotide.direction           = direction
                        new_nucleotide.index               = index
                        new_nucleotide.position            = [coordinates[2], coordinates[is_fwd]] #Sidechain(axis) and backbone coordinates
                        new_nucleotide.strand              = None
                        new_nucleotide.vh                  = vh
                        new_nucleotide.is_fwd              = is_fwd
                        new_nucleotide.skip                = None
                        new_nucleotide.pointer             = [vh, index, is_fwd]

                        #check if end or beginning of canvas
                        new_nucleotide.is_canvas_end       = (index == self.num_bases - 1)
                        new_nucleotide.is_canvas_start     = (index == 0)

                        #Assign the nucleotide
                        self.nucleotide_matrix[vh][index][is_fwd] = new_nucleotide

    def parse_skip_connections(self):
        self.inter_rigid_body_connections = set()
        self.inter_nucleotide_connections = set()

        for pointer_1, pointer_2 in self.skip_connections.items():
            vh_1, index_1, is_fwd_1 = pointer_1
            vh_2, index_2, is_fwd_2 = pointer_2

            nucleotide_1 = self.nucleotide_matrix[vh_1][index_1][is_fwd_1]
            nucleotide_2 = self.nucleotide_matrix[vh_2][index_2][is_fwd_2]

            #Get body numbers
            body_num_1   = nucleotide_1.body_num
            body_num_2   = nucleotide_2.body_num

            #Skip the skip nucleotides
            if nucleotide_1.skip or nucleotide_2.skip:
                continue

            #If both nucleotides are part of the rigid bodies, add the connection to rigid body connections
            if nucleotide_1.body.type and nucleotide_2.body.type and body_num_1 != body_num_2 and not (body_num_2,body_num_1) in self.inter_rigid_body_connections:
                self.inter_rigid_body_connections.add((body_num_1,body_num_2))


            #Add nucleotide-nucleotide connections to list
            nucleotide_sim_num_1 = nucleotide_1.simulation_nucleotide_num
            nucleotide_sim_num_2 = nucleotide_2.simulation_nucleotide_num

            if not (nucleotide_sim_num_2,nucleotide_sim_num_1) in self.inter_nucleotide_connections:
                self.inter_nucleotide_connections.add((nucleotide_sim_num_1, nucleotide_sim_num_2))

    def connect_skip_boundaries(self):
        '''
        Incorporate skips in the model by creating 'short_range_connections'
        between nucleotides in the beginning and end of skip region
        '''
        #1. Determine the skip boundaries
        self.skip_boundaries  = []
        for vh in range(self.num_vhs):
            for is_fwd in [0,1]:

                skip_begin_found = False
                skip_begin = None
                skip_end   = None
                for idx in range(len(self.skip_matrix[vh])):
                    if self.skip_matrix[vh][idx][is_fwd] and not skip_begin_found:
                        skip_begin_found = True
                        skip_begin = idx
                    elif not self.skip_matrix[vh][idx][is_fwd] and skip_begin_found:
                        skip_begin_found = False
                        skip_end = idx - 1
                        self.skip_boundaries.append([vh, is_fwd, skip_begin, skip_end])

        #2. Make short range connection between beginning and end of skip
        for vh, is_fwd, idx_begin, idx_end in self.skip_boundaries:
            pointer_1 = (vh, idx_begin - 1, 1)
            pointer_2 = (vh, idx_end + 1, 1)

            if not is_fwd:
                pointer_1 = (vh, idx_end + 1, 0)
                pointer_2 = (vh, idx_begin - 1, 0)

            self.short_range_connections[pointer_1] = pointer_2

    def assign_nucleotide_types(self):
        '''
        Build the nucleotide network for an origami design
        '''
        self.nucleotide_type_list = []
        for vh in range(len(self.nucleotide_matrix)):
            for idx in range(len(self.nucleotide_matrix[vh])):

                #Determine the nucleotide type (ssNucleotide vs dsNucletide) and nucleotide connections
                current_nucleotide_rev = self.nucleotide_matrix[vh][idx][0]
                current_nucleotide_fwd = self.nucleotide_matrix[vh][idx][1]

                if current_nucleotide_fwd != None and current_nucleotide_rev != None:
                    ds_nucleotide = DSNucleotide()
                    ds_nucleotide.fwd_nucleotide   = current_nucleotide_fwd
                    ds_nucleotide.rev_nucleotide   = current_nucleotide_rev
                    ds_nucleotide.type             = 1
                    ds_nucleotide.skip             = current_nucleotide_fwd.skip
                    self.nucleotide_type_matrix[vh][idx] = ds_nucleotide
                    self.nucleotide_type_list.append(ds_nucleotide)

                elif current_nucleotide_fwd != None:
                    ss_nucleotide                  = SSNucleotide()
                    ss_nucleotide.nucleotide       = current_nucleotide_fwd
                    ss_nucleotide.type             = 0
                    ss_nucleotide.skip             = current_nucleotide_fwd.skip

                    self.nucleotide_type_matrix[vh][idx] = ss_nucleotide
                    self.nucleotide_type_list.append(ss_nucleotide)

                elif current_nucleotide_rev != None:
                    ss_nucleotide                  = SSNucleotide()
                    ss_nucleotide.nucleotide       = current_nucleotide_rev
                    ss_nucleotide.type             = 0
                    ss_nucleotide.skip             = current_nucleotide_rev.skip

                    self.nucleotide_type_matrix[vh][idx] = ss_nucleotide
                    self.nucleotide_type_list.append(ss_nucleotide)

    def assign_nucleotide_connections(self):
        '''
        Assign nucleotide connections by looping over nucleotides in nucleotide_type_matrix
        then appending to nucleotide_type_matrix[vh][idx].rigid_connections
        the nucleotide_type_matrix[vh][idx] for both next and previous idx, if existent.
        '''
        #1. Add the base-stacking interactions
        for vh in range(len(self.nucleotide_type_matrix)):
            for idx in range(len(self.nucleotide_type_matrix[vh])):

                if self.nucleotide_type_matrix[vh][idx] == None or self.nucleotide_type_matrix[vh][idx].skip:
                    continue

                self.nucleotide_type_matrix[vh][idx].rigid_connections = []
                self.nucleotide_type_matrix[vh][idx].skip_connections  = []

                #Get the type for nucleotide (ssDNA or dsDNA?)
                type_1 = self.nucleotide_type_matrix[vh][idx].type

                #Pointer 1
                pointer1_rev = (vh, idx, 0)
                pointer1_fwd = (vh, idx, 1)

                # Calculate connections between idx and next idx, if existent
                if idx+1 < len(self.nucleotide_type_matrix[vh]) \
                   and not self.nucleotide_type_matrix[vh][idx+1] == None \
                   and not self.nucleotide_type_matrix[vh][idx+1].skip:

                    type_2 = self.nucleotide_type_matrix[vh][idx+1].type

                    if type_1*type_2: # both types are DSNucleotide, make the connection RIGID
                        self.nucleotide_type_matrix[vh][idx].rigid_connections.append(self.nucleotide_type_matrix[vh][idx+1])
                    else: # at least one is SSNucleotide, make a soft connection either in the fwd or rev direction
                        self.nucleotide_type_matrix[vh][idx].skip_connections.append(self.nucleotide_type_matrix[vh][idx+1])

                        #Pointer 2
                        pointer2_rev = (vh, idx+1, 0)
                        pointer2_fwd = (vh, idx+1, 1)

                        if pointer1_fwd in self.short_range_connections.keys() and self.short_range_connections[pointer1_fwd] == pointer2_fwd:
                            self.skip_connections[pointer1_fwd] = pointer2_fwd

                        elif self.short_range_connections[pointer2_rev] == pointer1_rev: #ssDNA connection is in the reverse direction
                            self.skip_connections[pointer2_rev] = pointer1_rev

                # Calculate connections between idx and previous idx, if existent
                if idx-1 >= 0 \
                   and not self.nucleotide_type_matrix[vh][idx-1] == None \
                   and not self.nucleotide_type_matrix[vh][idx-1].skip:

                    type_2 = self.nucleotide_type_matrix[vh][idx-1].type

                    if type_1*type_2:
                        self.nucleotide_type_matrix[vh][idx].rigid_connections.append(self.nucleotide_type_matrix[vh][idx-1])
                    else:
                        self.nucleotide_type_matrix[vh][idx].skip_connections.append(self.nucleotide_type_matrix[vh][idx-1])

                        pointer2_fwd = (vh, idx-1, 1)
                        pointer2_rev = (vh, idx-1, 0)

                        if pointer2_fwd in self.short_range_connections.keys() and self.short_range_connections[pointer2_fwd] == pointer1_fwd:
                            self.skip_connections[pointer2_fwd] = pointer1_fwd

                        elif self.short_range_connections[pointer1_rev] == pointer2_rev:
                            self.skip_connections[pointer1_rev] = pointer2_rev

        #2. Add short range connections that are not adjacent in sequence due to skips
        for pointer1, pointer2 in self.short_range_connections.items():
            vh1, idx1, is_fwd1 = pointer1
            vh2, idx2, is_fwd2 = pointer2

            #If the bases are not adjacent in sequence, add the connections to soft connections
            if abs(idx1-idx2) > 1:
                #Add the connections first in nucleotide type matrix
                self.nucleotide_type_matrix[vh1][idx1].skip_connections.append(self.nucleotide_type_matrix[vh2][idx2])
                self.nucleotide_type_matrix[vh1][idx1].skip_connections.append(self.nucleotide_type_matrix[vh2][idx2])
                self.skip_connections[pointer1] = pointer2

        #3. Add the crossover connections
        for pointer_1, pointer_2 in self.crossovers.items():
            (vh_1, index_1, is_fwd_1) = pointer_1
            (vh_2, index_2, is_fwd_2) = pointer_2

            type_1 = self.nucleotide_type_matrix[vh_1][index_1].type
            type_2 = self.nucleotide_type_matrix[vh_2][index_2].type

            if self.vh_vh_crossovers[vh_1][vh_2] > 1 and type_1*type_2: #make rigid if more than 1 xover and both nucleotides are dsDNA
                self.nucleotide_type_matrix[vh_1][index_1].rigid_connections.append(self.nucleotide_type_matrix[vh_2][index_2])
                self.nucleotide_type_matrix[vh_2][index_2].rigid_connections.append(self.nucleotide_type_matrix[vh_1][index_1])
            else: # make soft otherwise
                self.nucleotide_type_matrix[vh_1][index_1].skip_connections.append(self.nucleotide_type_matrix[vh_2][index_2])
                self.nucleotide_type_matrix[vh_2][index_2].skip_connections.append(self.nucleotide_type_matrix[vh_1][index_1])

                #Add the connection to soft connection list
                self.skip_connections[pointer_1] = pointer_2

        #4. Add long-range connections (always soft!)
        for pointer_1, pointer_2 in self.long_range_connections.items():
            (vh_1, index_1, is_fwd_1) = pointer_1
            (vh_2, index_2, is_fwd_2) = pointer_2

            self.nucleotide_type_matrix[vh_1][index_1].skip_connections.append(self.nucleotide_type_matrix[vh_2][index_2])
            self.nucleotide_type_matrix[vh_2][index_2].skip_connections.append(self.nucleotide_type_matrix[vh_1][index_1])

            #Add the connection to soft connection list
            self.skip_connections[pointer_1] = pointer_2

    def get_connections(self):
        '''
        Populate 3' connections for each (staple / scaffold) strand
        '''
        self.crossovers             = {}
        self.long_range_connections = {}
        for vh in range(self.num_vhs):
            staple_strandSet   = self.part.getStrandSets(vh)[not(vh % 2)]
            scaffold_strandSet = self.part.getStrandSets(vh)[(vh % 2)]

            for cadnano_strand in staple_strandSet:
                self.connection3p(cadnano_strand)
            for strand in scaffold_strandSet:
                self.connection3p(cadnano_strand)

    def dfs(self, start_nucleotide_type):
        '''
        Depth-first-search graph traverse algorithm to find connected components
        '''
        visited, stack = set(), [start_nucleotide_type]
        while stack:
            new_nucleotide_type = stack.pop()
            if new_nucleotide_type not in visited:
                visited.add(new_nucleotide_type)
                new_nucleotide_type.visited = True
                stack.extend(set(new_nucleotide_type.rigid_connections) - visited)
        return visited

    def cluster_into_bodies(self):
        '''
        Cluster the DNA origami structure into body clusters
        '''

        self.num_bodies            = 0
        self.body_nucleotide_types = []
        self.body_nucleotides      = []
        self.rigid_bodies          = []
        self.soft_bodies           = []

        #1. Identify the clusters using depth-first-search
        for nucleotide_type in self.nucleotide_type_list:

            if nucleotide_type.visited or nucleotide_type.skip:
                continue

            #Get the nucleotide types for each cluster
            self.body_nucleotide_types.append([])
            self.body_nucleotide_types[self.num_bodies] = list(self.dfs(nucleotide_type))

            #Check if the cluster is a rigid body
            rigid_body = functools.reduce(lambda x,y:x*y,[nucleotide_type.type for nucleotide_type in self.body_nucleotide_types[self.num_bodies]])

            #Create a new Body object
            new_body = Body()
            new_body.nucleotide_types = self.body_nucleotide_types[self.num_bodies]
            new_body.type             = rigid_body

            #If the body is rigid add to rigid body collection
            if rigid_body:
                self.rigid_bodies.append(new_body)
            else:
                self.soft_bodies.append(new_body)

            self.num_bodies += 1

        #2. Update soft body nucleotide body position numbers and body numbers
        nucleotide_number = 0
        for i in range(len(self.soft_bodies)):
            soft_body               = self.soft_bodies[i]
            soft_body.nucleotides   = []
            nucleotide_type         = soft_body.nucleotide_types[0]

            #Update the nucleotide number
            nucleotide_type.nucleotide.simulation_nucleotide_num = nucleotide_number

            #Assign soft body to nucleotide
            nucleotide_type.nucleotide.body = soft_body

            soft_body.nucleotides  += [nucleotide_type.nucleotide]
            self.body_nucleotides  += [nucleotide_type.nucleotide]

            nucleotide_number      += 1

            #Initialize soft bodies
            soft_body.initialize()

        #3. Update rigid body nucleotide body position numbers and body numbers
        for i in range(len(self.rigid_bodies)):
            rigid_body             = self.rigid_bodies[i]
            rigid_body.nucleotides = []
            for nucleotide_type in rigid_body.nucleotide_types:
                fwd_nucleotide = nucleotide_type.fwd_nucleotide
                rev_nucleotide = nucleotide_type.rev_nucleotide

                fwd_nucleotide.simulation_nucleotide_num = nucleotide_number
                rev_nucleotide.simulation_nucleotide_num = nucleotide_number+1

                #Assign the rigid bodies to nucleotides
                fwd_nucleotide.body = rigid_body
                rev_nucleotide.body = rigid_body

                nucleotide_number    += 2

                rigid_body.nucleotides+= [fwd_nucleotide,rev_nucleotide]
                self.body_nucleotides += [fwd_nucleotide,rev_nucleotide]

            #Initialize rigid bodies
            rigid_body.initialize()

    def get_coordinates(self, vh, index):
        '''
        Given a vh and a index, returns (x,y,z)
        for the sidechain pts and backbones fwd and rev
        '''
        axis_pts = self.part.getCoordinates(vh)[0][index]
        fwd_pts  = self.part.getCoordinates(vh)[1][index]
        rev_pts  = self.part.getCoordinates(vh)[2][index]

        return [rev_pts, fwd_pts, axis_pts]

    def get_nucleotide(self, pointer):
        '''
        Given a tuple of pointers in the form [vh, index, is_fwd],
        Returns the global nucleotide referent to the pointers
        '''
        [vh, index, is_fwd] = pointer
        return self.nucleotide_matrix[vh][index][is_fwd]

    def get_nucleotide_type(self, pointer):
        '''
        Given a tuple of pointers in the form [vh, index, is_fwd],
        Returns the global nucleotide type referent to the pointers
        '''
        [vh, index, is_fwd] = pointer
        return self.nucleotide_type_matrix[vh][index]

    def connection3p(self, cadnano_strand):
        '''
        Given a strand, returns the vhelix to which the 3p end
        connects to, if the distance is not too far
        '''
        if cadnano_strand.connection3p() != None:
            vh_1 = cadnano_strand.idNum()
            index_1 = cadnano_strand.idx3Prime()
            is_fwd_1 = int(cadnano_strand.isForward())

            vh_2 = cadnano_strand.connection3p().idNum()
            index_2 = cadnano_strand.connection3p().idx5Prime()
            is_fwd_2 = int(cadnano_strand.connection3p().isForward())

            conn_pointer_1 = (vh_1, index_1, is_fwd_1)
            conn_pointer_2 = (vh_2, index_2, is_fwd_2)

            distance = self.distance_between_vhs(vh_1, index_1, is_fwd_1, vh_2, index_2, is_fwd_2)

            if distance < self.crossover_distance:
                self.crossovers[conn_pointer_1]    =  conn_pointer_2
                self.vh_vh_crossovers[vh_1][vh_2] += 1
                self.vh_vh_crossovers[vh_2][vh_1] += 1
            else:
                self.long_range_connections[conn_pointer_1] = conn_pointer_2

    def distance_between_vhs(self, vh1, index1, is_fwd1, vh2, index2, is_fwd2):
        '''
        Given 2 points(vh, index), calculates the
        Euclian distance between them
        '''
        pos1 = self.get_coordinates(vh1, index1)[is_fwd1]
        pos2 = self.get_coordinates(vh2, index2)[is_fwd2]
        distance = np.linalg.norm(pos1 - pos2)
        return distance

    def calculate_nucleotide_quaternions(self):
        '''
        Calculate the quaternion for every nucleotide wrt
        the first nucleotide in oligos_list
        '''

        # 1. Calculate list of vectors for very 1st nucleotide in oligo. It will have quaternion (1,0,0,0)
        strand_0 = self.oligos_list[0].strands_list[0]

        [vh_0, index_0, is_fwd_0] = strand_0.pointers_list[0]  # very first nucleotide in origami
        [vh_1, index_1, is_fwd_1] = strand_0.pointers_list[1]  # second nucleotide in origami

        [axis_0, backbone_0] = self.nucleotide_matrix[vh_0][index_0][is_fwd_0].position
        [axis_1, backbone_1] = self.nucleotide_matrix[vh_1][index_1][is_fwd_1].position

        if self.get_nucleotide([vh_0, index_0, is_fwd_0]).skip == True:
            raise Exception('First nucleotide in design is a skip!!')

        if self.get_nucleotide([vh_1, index_1, is_fwd_1]).skip == True:
            raise Exception('Second nucleotide in design is a skip!!')

        #Calculate the vectors to be used for quaternion calculation
        base_vector_0  = axis_0 - backbone_0 # along watson_crick direction
        axial_vector_0 = axis_1 - axis_0     # along double-helix axis direction
        orth_vector_0  = np.cross(base_vector_0, axial_vector_0)

        # return 3 orthogonal vectors in nucleotide, for quaternion
        vect_list_0 = (base_vector_0/np.linalg.norm(base_vector_0), \
                       axial_vector_0/np.linalg.norm(axial_vector_0), \
                       orth_vector_0/np.linalg.norm(orth_vector_0))

        # 2. Calculate quaternions for other nucleotides wrt vect_list_0
        for vh in range(self.num_vhs):
            for index in range(self.num_bases):
                for is_fwd in range(2):
                    nucleotide = self.nucleotide_matrix[vh][index][is_fwd]

                    direction = -(-1)**(is_fwd)

                    [vh_1, index_1, is_fwd_1] = [vh, index, is_fwd]
                    [axis_1, backbone_1]      = nucleotide.position

                    is_rev = 1 - is_fwd
                    end_of_canvas = False
                    if nucleotide.is_canvas_end * is_fwd  or nucleotide.is_canvas_start * is_rev:
                        end_of_canvas = True

                    if nucleotide.is_strand_end == False and end_of_canvas == False:
                        [vh_2, index_2, is_fwd_2] = [vh, index + direction, is_fwd]
                        [axis_2, backbone_2]      = self.nucleotide_matrix[vh_2][index_2][is_fwd_2].position

                        base_vector_1  = axis_1 - backbone_1
                        axial_vector_1 = axis_2 - axis_1
                        orth_vector_1  = np.cross(base_vector_1, axial_vector_1)

                    #last oligo in strand chain or canvas, calculate vectors wrt to previous nucleotide
                    elif nucleotide.is_strand_end == True or end_of_canvas == True:
                        [vh_2, index_2, is_fwd_2] = [vh, index - direction, is_fwd]
                        [axis_2, backbone_2]      = self.nucleotide_matrix[vh_2][index_2][is_fwd_2].position

                        base_vector_1 = axis_1 - backbone_1
                        axial_vector_1 = - (axis_2 - axis_1)
                        orth_vector_1 = np.cross(base_vector_1, axial_vector_1)

                    vect_list_1 = (base_vector_1+np.array([0.00001,0,0])/np.linalg.norm(base_vector_1+np.array([0.00001,0,0])), \
                                   axial_vector_1+np.array([0.00001,0,0])/np.linalg.norm(axial_vector_1+np.array([0.00001,0,0])), \
                                   orth_vector_1+np.array([0.00001,0,0])/np.linalg.norm(orth_vector_1+np.array([0.00001,0,0])))

                    nucl                      = self.nucleotide_matrix[vh_1][index_1][is_fwd_1]
                    nucl.vectors_body_frame   = vect_list_1
                    nucl.vectors_global_frame = [axis_1, backbone_1, orth_vector_1 + backbone_1]
                    nucl_quaternion           = vectortools.find_quaternion_from_2_axes(vect_list_0, vect_list_1)
                    nucl.quaternion           = [nucl_quaternion.w, \
                                                 nucl_quaternion.x, \
                                                 nucl_quaternion.y, \
                                                 nucl_quaternion.z]

    def update_shifted_nucleotide(self, old_nucleotide, index_shift, insert_num):
        '''
        copy the attributes from old_nucleotide that don't change with shift,
        then update the ones that do
        '''
        [vh, index, is_fwd] = old_nucleotide.pointer
        shifted_index       = index + index_shift

        try:
            shifted_nucleotide  = self.nucleotide_matrix[vh][shifted_index][is_fwd]
        except IndexError:
            sys.exit('not enough bases in canvas to add all inserts')

        #explicit is better than implicit (ZEN)
        #copy old nucleotide values that do not change with shift
        shifted_nucleotide.direction                 = old_nucleotide.direction
        shifted_nucleotide.is_fwd                    = old_nucleotide.is_fwd
        shifted_nucleotide.strand                    = old_nucleotide.strand
        shifted_nucleotide.vh                        = old_nucleotide.vh
        shifted_nucleotide.skip                      = old_nucleotide.skip
        shifted_nucleotide.inserts                   = old_nucleotide.inserts
        shifted_nucleotide.is_oligo_end              = old_nucleotide.is_oligo_end
        shifted_nucleotide.is_strand_end             = old_nucleotide.is_strand_end
        shifted_nucleotide.vectors_body_frame        = old_nucleotide.vectors_body_frame
        shifted_nucleotide.body                      = old_nucleotide.body
        shifted_nucleotide.body_num                  = old_nucleotide.body_num

        shifted_nucleotide.oligo                     = old_nucleotide.oligo

        #now update the values that do change with shift
        shifted_index                                = old_nucleotide.index + index_shift

        shifted_nucleotide.index                     = shifted_index
        shifted_nucleotide.is_canvas_end             = shifted_index == self.num_bases - 1
        shifted_nucleotide.is_canvas_start           = shifted_index == 0
        shifted_nucleotide.pointer                   = [shifted_nucleotide.vh,   \
                                                           shifted_nucleotide.index, \
                                                           shifted_nucleotide.is_fwd]

        shifted_nucleotide.is_oligo_start            = old_nucleotide.is_oligo_start
        #update oligo's starting point
        if shifted_nucleotide.is_oligo_start:
            shifted_nucleotide.oligo.start_pointer   = shifted_nucleotide.pointer

        #update the dictionary containint old->new correspondence (to be used for inserts)
        old_pointer                                  = tuple(old_nucleotide.pointer)
        old_pointer                                 += (insert_num,)
        new_pointer                                  = shifted_nucleotide.pointer

        self.insert_update_dict[old_pointer]         = new_pointer

        # keep the nucleotide's next pointing to the original one, this will be updated later in calculate_next_nucleotide
        shifted_nucleotide.next                      = old_nucleotide.next

        return(shifted_nucleotide)

    def calculate_next_nucleotide(self):
        '''
        '''
        #1. populate who's next by following strands along oligo (ignore skips for now)
        for o, oligo in enumerate(self.oligos_list):
            for s, strand in enumerate(oligo.strands_list):
                for p, pointer in enumerate(strand.pointers_list):
                    this_nucleotide = self.get_nucleotide(pointer)

                    # test for end of strand, and end of oligo
                    if this_nucleotide.is_strand_end:
                        if this_nucleotide.is_oligo_end:
                            # if oligo is circular, connect last and 1st nucleotides and exit
                            if oligo.is_circular:
                                next_pointer    = self.oligos_list[o].strands_list[0].pointers_list[0]
                                next_nucleotide = self.get_nucleotide(next_pointer)

                        # if end of strand but not of oligo, next is the 1st in next strand
                        elif this_nucleotide.is_oligo_end == False:
                            next_pointer    = self.oligos_list[o].strands_list[s+1].pointers_list[0]
                            next_nucleotide = self.get_nucleotide(next_pointer)

                    elif this_nucleotide.is_strand_end == False:
                        # test for skips
                        next_pointer    = self.oligos_list[o].strands_list[s].pointers_list[p + 1]
                        next_nucleotide = self.get_nucleotide(next_pointer)
                        while next_nucleotide.skip == True:
                            p += 1
                            next_pointer    = self.oligos_list[o].strands_list[s].pointers_list[p + 1]
                            next_nucleotide = self.get_nucleotide(next_pointer)

                    this_nucleotide.next = next_nucleotide

        #2. Shift nucleotides according to the number of inserts in the structure
        for vh in range(self.num_vhs):
            strand_sets = self.part.getStrandSets(vh)
            [fwd_strand_set, rvs_strand_set] = strand_sets

            #Identify low and high bases in origami
            high_base = fwd_strand_set.indexOfRightmostNonemptyBase()

            try:
                low_base  = fwd_strand_set.indexOfLeftmostNonemptyBase()
            except AttributeError:
                sys.exit('indexOfLeftmostNonemptyBase is only available at v2.5+ of cadnano. Please update.')

            for strand_set in [fwd_strand_set, rvs_strand_set]:
                inserts_in_vh  = 0
                strand0        = strand_set.strands()[0]
                is_fwd         = strand0.isForward()
                inserts        = strand0.insertionLengthBetweenIdxs(low_base, high_base)
                inserts_in_vh += inserts
                num_inserted   = 0
                index_shift        = inserts_in_vh - num_inserted

                #start from rightmost base in strandset and walk to the left
                for index in range(high_base, low_base - 1, -1):

                    shifted_index      = index + index_shift
                    old_nucleotide     = self.nucleotide_matrix[vh][index][is_fwd]

                    has_inserts        = self.insert_matrix[old_nucleotide.vh][old_nucleotide.index][old_nucleotide.is_fwd]

                    if is_fwd:
                        local_inserts = 0
                        if has_inserts is not None:
                            num_inserts   = has_inserts
                            # print(old_nucleotide.pointer)
                            for inserts in range(num_inserts):
                                shifted_nucleotide = self.update_shifted_nucleotide(old_nucleotide, index_shift, local_inserts)
                                # print(["1", old_nucleotide.pointer,old_nucleotide.next.pointer,shifted_nucleotide.pointer, shifted_nucleotide.next.pointer])
                                local_inserts     += 1
                                num_inserted      += 1
                                index_shift        = inserts_in_vh - num_inserted

                        shifted_nucleotide = self.update_shifted_nucleotide(old_nucleotide, index_shift, local_inserts)
                    else:
                        local_inserts = 0
                        if has_inserts is not None:
                            num_inserts   = has_inserts
                            # print(old_nucleotide.pointer)
                            for inserts in reversed(range(num_inserts)):
                                shifted_nucleotide = self.update_shifted_nucleotide(old_nucleotide, index_shift, inserts + 1)
                                # print(["1", old_nucleotide.pointer,old_nucleotide.next.pointer,shifted_nucleotide.pointer, shifted_nucleotide.next.pointer])
                                local_inserts     += 1
                                num_inserted      += 1
                                index_shift        = inserts_in_vh - num_inserted

                        shifted_nucleotide = self.update_shifted_nucleotide(old_nucleotide, index_shift, 0)

                    # print(["2", old_nucleotide.pointer,old_nucleotide.next.pointer,shifted_nucleotide.pointer, shifted_nucleotide.next.pointer])

        #3. Now that all nucleotides have been shifted, update 'next' based on dictionary of tuples
        #3.1 go to a new pointer (new_pointer) and calculate the shift between old and new (shift)
        #3.2 find who new_pointer used to point to (old_next)
        #3.3 update the old_next to new_next based on the shift
        #3.4 update who this shifted is now pointing to after neighbors have shifted

        def find_leftmost_insert_boundary(lookup_pointer):
            '''
            given a pointer part of a insert region,
            return pointer to the lowest index in the insert region
            '''
            dict = self.insert_update_dict
            boundary = 0
            for extended_pointer in dict:
                pointer = extended_pointer[:3]
                if list(pointer) == lookup_pointer:
                    insert_num = extended_pointer[3]
                    if insert_num > boundary:
                        boundary = insert_num
            new_extended_pointer = tuple(lookup_pointer) + (boundary,)
            # print([lookup_pointer, new_extended_pointer])
            return(new_extended_pointer)

        for old_pointer in self.insert_update_dict:

            insert_shift            = old_pointer[3]
            shifted_pointer         = self.insert_update_dict[old_pointer]
            shifted_nucleotide      = self.get_nucleotide(shifted_pointer)

            old_next_pointer        = shifted_nucleotide.next.pointer
            is_fwd                  = old_pointer[2]

            old_next_pointer = find_leftmost_insert_boundary(old_next_pointer)

            # print([shifted_nucleotide.next.pointer, old_next_pointer])
            shifted_next_pointer    = self.insert_update_dict[old_next_pointer][:] #shallow copy, dont overwrite it
            if is_fwd:
                shifted_next_pointer[1]-= insert_shift
            else:
                shifted_next_pointer[1]+= insert_shift

            shifted_nucleotide.next = self.get_nucleotide(shifted_next_pointer)

            # print([old_pointer, shifted_nucleotide.pointer, shifted_nucleotide.next.pointer])


    def update_oligos_list(self):

        new_oligo_list = []
        for oligo in self.oligos_list:
            start_pointer    = oligo.start_pointer
            this_nucleotide = self.get_nucleotide(start_pointer)
            end_of_oligo    = False
            strand_counter  = 0 #strand number
            while not end_of_oligo:
                new_pointers_list    = []

                end_of_strand        = False
                strand               = oligo.strands_list[strand_counter]

                new_pointers_list.append(this_nucleotide.pointer)

                while not end_of_strand:
                    next_nucleotide   = this_nucleotide.next
                    new_pointers_list.append(next_nucleotide.pointer)

                    this_nucleotide   = next_nucleotide
                    end_of_strand     = this_nucleotide.is_strand_end
                strand.pointers_list  = new_pointers_list

                end_of_oligo     = this_nucleotide.is_oligo_end

                # if strand_end but not oligo_end, move nucleotide pointer to next strand
                if not end_of_oligo:
                    this_nucleotide = this_nucleotide.next

                strand_counter  += 1

class Oligo:
    '''
    An oligo is a collection of strands
    '''
    def __init__(self):
        self.cadnano_oligo     = None                 # Oligo class from cadnano, inherenting all its attributes
        self.is_circular       = False
        self.strands_list      = []                   # Ordered list of strands making up this oligo
        self.start_pointer     = None

class Strand:
    '''
    A strand is a collection of connected nucleotides,
    bounded between xovers or starting / ending points
    '''
    def __init__(self):
        #cadnano properties
        self.vh                = None
        self.index_5p          = None
        self.index_3p          = None
        self.is_fwd            = None

        #for carbs usage
        self.pointers_list     = []                   # Ordered list of pointers in the form [vh, index, is_fwd] for each nucleotide in strand

class DSNucleotide:
    '''
    Fwd and Rev (sense/antisense) nucleotides making up a double strand nucleotide
    '''
    def __init__(self):
        self.fwd_nucleotide    = None                 # Nucleotide in forward direction (reference frame)
        self.rev_nucleotide    = None                 # Nucleotide in reverse direction
        self.type              = 1                    # 1: double strand (dsDNA)
        self.visited           = False                # To be used by depth-first-search
        self.rigid             = False
        self.skip              = None                 # Whether this nucleotide is a skip in cadnano

        #Connections
        self.rigid_connections = []                   # List of rigid connections
        self.skip_connections  = []                   # List of soft connections

class SSNucleotide:
    '''
    Nucleotide that is part of ssDNA strand
    '''
    def __init__(self):
        self.nucleotide        = None
        self.type              = 0                    # 0: single strand (ssDNA)
        self.visited           = False                # To be used by depth-first-search
        self.rigid             = False
        self.skip              = None                 # Whether this nucleotide is a skip in cadnano

        #Connections
        self.rigid_connections = []                   # List of rigid connections
        self.skip_connections  = []                   # List of soft connections

class Nucleotide:
    '''
    Fixed attributes of a nucleotide
    '''
    def __init__(self):
        self.direction                    = None      # 1 is fwd, 0 is reverse
        self.is_fwd                       = None      # 0: reverse, 1:forward
        self.index                        = None      # z position in cadnano's unit
        self.oligo                        = None      # Nucleotide's oligo number
        self.strand                       = None      # Nucleotide's strand number
        self.vh                           = None      # Nucleotide's virtual helix
        self.skip                         = False     # Whether the nucleotide is a skip
        self.inserts                      = None      # How many insertions in this nucleotide site
        self.pointer                      = [self.vh, self.index, self.is_fwd]

        self.next                         = None      # Next nucleotide in a strand, if existent
        self.is_strand_end                = False     # Whether this nucleotide is in the end of a strand
        self.is_oligo_end                 = False     # Whether this nucleotide is in the end of an oligo
        self.is_oligo_start               = False     # Whether this nucleotide is in the beginning of an oligo
        self.is_canvas_end                = False     # Whether this nucleotide is in the end of the canvas (high index)
        self.is_canvas_start              = False     # Whether this nucleotide is in the beginning of the canvas (low index)

        self.quaternion                   = None      # quaternion orientation for this nucleotide
        self.vectors_body_frame           = None      # normalized orthogonal vectors in the body reference frame (bases = along WC pairing, axial, orthogonal)
        self.position                     = None      # Nucleotide positions for axis particle [0] and backbone [1]
        self.vectors_simulation_nums      = []        # hoomd simulation number for base and aux (orthogonal) beads

        # Body / simulation variables
        self.body                         = None      # body (class) this nucleotide belongs to
        self.body_num                     = 0         # body number
        self.simulation_nucleotide_num    = 0         # nucleotide number wrt the hoomd simulation

class Body:
    '''
    Fixed attributes of a body.
    A body is a combination of neighboring vhs that move together during
    relaxation as one rigid body. HOOMD will need its center of mass position,
    orientation (quaternion), and moment of inertia.
    '''
    def __init__(self):
        self.comass_position   = None                 # position of body's center of mass
        self.comass_quaternion = None                 # quaternion of body's center of mass
        self.moment_inertia    = None                 # body's moment of intertia (calculated via vectortools)
        self.mass              = None                 # body's mass (sum of number of nucleotides)
        self.nucleotide_types  = []                   # list of nucleotide types belonging to this body
        self.nucleotides       = []                   # list of nucleotides belonging to this body
        self.vhs               = []                   # list of vhs belonging to this body
        self.type              = None                 # 0: soft, 1:rigid

    def add_nucleotide_type(self, nucleotide_type):
        self.nucleotide_types.append(nucleotide_type)

    def add_vh(self, vh):
        self.vhs.add(vh)

    def initialize(self):
        '''
        Given a collection of nucleotides making up a body, initialize the body
        by calculating the following properties: comass_position, comass_quaternion, and moment_inertia
        '''

        # extract the position of backbone bead (1) acquired from cadnano
        positions = [nucleotide.position[1] for nucleotide in self.nucleotides]
        self.comass_position    = vectortools.calculateCoM(positions)
        self.mass               = float(len(positions))
        self.moment_inertia     = vectortools.calculateMomentInertia(positions) * self.mass
        self.comass_quaternion  = [1., 0., 0., 0.]
