import numpy as np
import cadnano
import functools

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
        self.part                   = None
        self.oligos_list            = None
        self.num_vhs                = None
        self.nucleotide_matrix      = None   # This is the 'global' information about every nucleotide

        self.nucleotide_type_list   = None
        self.nucleotide_type_matrix = None

        self.crossovers             = None
        self.vh_vh_crossovers       = None   # Given vh1, vh2, vh_vh_crossovers[vh_1][vh_2] is the number of xovers between them
        self.long_range_connections = {}     # Dict connecting pointer_1 (vh, index, is_fwd) to pointer_2
        self.short_range_connections= {}     # Dict connecting pointer_1 (vh, index, is_fwd) to pointer_2
        self.skip_connections       = {}     # Dict of pointers referring to nucleotides separated by skip

        #Distance constraints
        self.crossover_distance      = 2.0   # Distance in Angstrom

    def parse_oligos_from_cadnano(self):
        '''
        Given a origami part
        Return an array with all oligos in part sorted by length
        '''
        self.oligos_list = []

        cadnano_oligos_list = self.part.oligos()
        cadnano_oligos_list = sorted(cadnano_oligos_list, key=lambda x: x.length(), reverse=True)
        for cadnano_oligo in cadnano_oligos_list:
            new_oligo = Oligo()
            new_oligo.cadnano_oligo = cadnano_oligo
            new_oligo.is_circular = cadnano_oligo.isCircular()
            self.oligos_list.append(new_oligo)

    def parse_oligo(self, oligo):
        '''
        Given a initialized carbs oligo,
        populates its strands_list with strands
        '''
        cadnano_oligo = oligo.cadnano_oligo
        generator     = cadnano_oligo.strand5p().generator3pStrand()

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
                new_strand.pointers_list.append([new_strand.vh, new_strand.is_fwd, i])

            oligo.strands_list.append(new_strand)
        return oligo_helper_list


    def initialize_nucleotide_matrix(self):
        '''
        Creates an empty matrix of len = vh_length x index_length
        to be populated with all nucleotides in part (fwd and rev)
        It has the form nucleotide_matrix[vh][index][rev or fwd]
        '''
        self.num_vhs = len(list(self.part.getIdNums()))
        num_bases    = self.part.getVirtualHelix(0).getSize()

        self.nucleotide_matrix       = [[[None,None]  for idx in range(num_bases)] for vh in range(self.num_vhs)]
        self.nucleotide_type_matrix  = [[ None  for idx in range(num_bases)] for vh in range(self.num_vhs)]
        self.vh_vh_crossovers        = [[0  for vh in range(self.num_vhs)] for vh in range(self.num_vhs)]
        self.skip_matrix             = [[[False,False]  for idx in range(num_bases)] for vh in range(self.num_vhs)]
        self.insert_matrix           = [[[False,False]  for idx in range(num_bases)] for vh in range(self.num_vhs)]

    def populate_skips_inserts_matrix(self):
        '''
        Identify all the skips in the structure
        '''
        for oligo in self.oligos_list:
            cadnano_oligo = oligo.cadnano_oligo
            generator  = cadnano_oligo.strand5p().generator3pStrand()
            for strand in generator:
                vh = strand.idNum()
                index_5p  = strand.idx5Prime()
                index_3p  = strand.idx3Prime()
                direction = (-1 + 2*strand.isForward()) #-1 if backwards, 1 if fwd
                for idx in range(index_5p, index_3p + direction, direction):
                    if strand.hasInsertionAt(idx):
                        if strand.insertionLengthBetweenIdxs(idx,idx) == -1:
                            self.skip_matrix[vh][idx][int(strand.isForward())] = True
                        elif strand.insertionLengthBetweenIdxs(idx,idx) == 0:
                            self.insert_matrix[vh][idx][int(strand.isForward())] = True

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
                self.inter_nucleotide_connections.add((nucleotide_sim_num_1,nucleotide_sim_num_2))

    def incorporate_skips(self):
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

    def parse_oligo(self, oligo):
        '''
        Given a initialized carbs oligo,
        populates its strands_list with strands
        '''
        cadnano_oligo = oligo.cadnano_oligo
        generator     = cadnano_oligo.strand5p().generator3pStrand()

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
                new_strand.pointers_list.append([new_strand.vh, new_strand.is_fwd, i])

            oligo.strands_list.append(new_strand)
        return oligo_helper_list

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

    def create_strand_list_and_populate_nucleotide_matrix(self, oligo):
        '''
        Given an oligo, returns a list of strands,
        each containing the pointers ([vh][index][is_fwd]) to the
        nucleotides making up such strand and *populate nucleotides matrix*
        with attributes for this oligo
        '''
        if self.nucleotide_matrix == None:
            self.initialize_nucleotide_matrix()

        strand_list = []
        for helper_strands in self.parse_oligo(oligo):
            nucleotides_list = []
            for i in range(len(helper_strands)):
                #Current nucleotide
                strand, direction, vh, index = helper_strands[i]

                if i+1 < len(helper_strands):
                    strand_next, direction_next,vh_next,index_next = helper_strands[i+1]
                    conn_pointer_1 = (vh     ,index     , int(direction > 0     ))
                    conn_pointer_2 = (vh_next,index_next, int(direction_next > 0))

                    #Assign short range connection
                    self.short_range_connections[conn_pointer_1] = conn_pointer_2

                #Get the coordinates
                coordinates = self.get_coordinates(vh, index)

                #Get direction
                is_fwd = int(direction > 0)

                new_nucleotide = Nucleotide()
                new_nucleotide.direction         = direction
                new_nucleotide.index             = index
                new_nucleotide.position          = [coordinates[2], coordinates[is_fwd]] #Sidechain(axis) and backbone coordinates
                new_nucleotide.strand            = strand
                new_nucleotide.vh                = vh
                new_nucleotide.is_fwd            = is_fwd
                new_nucleotide.skip              = self.skip_matrix[vh][index][is_fwd]

                #Assign the nucleotide
                self.nucleotide_matrix[vh][index][is_fwd] = new_nucleotide

                nucleotides_list.append([vh, index, is_fwd])
            strand_list.append(nucleotides_list)
        return strand_list

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

    def oligos_list_to_nucleotide_info(self, i, j, k):
        '''
        Returns a tuple list (aka pointer) [vh, index, is_fwd]
        for the nucleotide oligos_list[i][j][k] where
        i,j,k are the indices for the oligo, strand, and nucleotide.
        '''
        [vh, index, is_fwd] = self.oligos_list[i][j][k]
        return [vh, index, is_fwd]

    def create_oligos_list(self):
        '''
        Given an array of oligos in part, returns a list of oligos,
        each containing a list of strands, each containing a
        list of nucleotides making up the part.
        In the process, also populate the nucleotide_matrix w/ nucleotides
        '''
        self.oligos_list     = []
        for oligo in self.oligos:
            strand_list = self.create_strand_list_and_populate_nucleotide_matrix(oligo)
            self.oligos_list.append(strand_list)

    def distance_between_vhs(self, vh1, index1, is_fwd1, vh2, index2, is_fwd2):
        '''
        Given 2 points(vh, index), calculates the
        Euclian distance between them
        '''
        pos1 = self.get_coordinates(vh1, index1)[is_fwd1]
        pos2 = self.get_coordinates(vh2, index2)[is_fwd2]
        distance = np.linalg.norm(pos1 - pos2)
        return distance

    def calculate_quaternions_from_positions(self):
        '''
        Given an list of list of strands, calculates the quaternion
        for each nucleotide wrt the very first nucleotide in oligos_list
        for o, oligo in enumerate(self.oligos_list):
        '''
        for o, oligo in enumerate(self.oligos_list):
            for s, strand in enumerate(oligo):

                [vh_0, index_0, is_fwd_0] = self.oligos_list_to_nucleotide_info(0, 0, 0) # very first nucleotide
                [vh_1, index_1, is_fwd_1] = self.oligos_list_to_nucleotide_info(0, 0, 1) # second nucleotide

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

                for p, pointer in enumerate(self.oligos_list[o][s]):
                    [vh_1, index_1, is_fwd_1] = self.oligos_list_to_nucleotide_info( o, s, p)
                    [axis_1, backbone_1]      = self.nucleotide_matrix[vh_1][index_1][is_fwd_1].position

                    if p < len(self.oligos_list[o][s]) - 1:
                        [vh_2, index_2, is_fwd_2] = self.oligos_list_to_nucleotide_info(o, s, p + 1)
                        [axis_2, backbone_2]      = self.nucleotide_matrix[vh_2][index_2][is_fwd_2].position

                        base_vector_1  = axis_1 - backbone_1
                        axial_vector_1 = axis_2 - axis_1
                        orth_vector_1  = np.cross(base_vector_1, axial_vector_1)

                    #last oligo in strand chain, calculate vectors wrt to previous nucleotide
                    elif p == len(self.oligos_list[o][s]) - 1:
                        [vh_2, index_2, is_fwd_2] = self.oligos_list_to_nucleotide_info(o, s, p - 1)
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

    def calculate_next_nucleotide(self):
        '''
        Given an list of list of strands, calculates the next nucleotide for
        each nucleotide in strand. If is the end of strand or oligo,
        assign that to the proper variable
        '''
        for o, oligo in enumerate(self.oligos_list):
            oligo_is_circular = self.oligos_type_list[o]
            for s, strand in enumerate(self.oligos_list[o]):
                for p, pointer in enumerate(self.oligos_list[o][s]):
                    this_nucleotide = self.get_nucleotide(pointer)

                    # test for end of strand
                    if p == len(strand) - 1:
                        this_nucleotide.strand_end = True

                        # test for end of oligo
                        if s == len(oligo) - 1:
                            this_nucleotide.oligo_end = True
                            # if oligo is circular, connect last and 1st nucleotides and exit
                            if oligo_is_circular == True:
                                next_pointer    = self.oligos_list[o][0][0]
                                next_nucleotide = self.get_nucleotide(next_pointer)
                                this_nucleotide.next_nucleotide = next_nucleotide
                            continue
                        elif s < len(oligo) - 1:
                            next_pointer    = self.oligos_list[o][s+1][0]
                            next_nucleotide = self.get_nucleotide(next_pointer)
                    elif p < len(strand) - 1:
                        # test for skips
                        next_pointer    = self.oligos_list[o][s][p + 1]
                        next_nucleotide = self.get_nucleotide(next_pointer)
                        while next_nucleotide.skip == True:
                            p += 1
                            next_pointer    = self.oligos_list[o][s][p + 1]
                            next_nucleotide = self.get_nucleotide(next_pointer)
                    this_nucleotide.next_nucleotide = next_nucleotide

class Oligo:
    '''
    An oligo is a collection of strands
    '''
    def __init__(self):
        self.cadnano_oligo     = None                 # Oligo class from cadnano, inherenting all its attributes
        self.is_circular       = False
        self.strands_list      = None                   # Ordered list of strands making up this oligo

class Strand:
    '''
    A strand is a collection of connected nucleotides,
    bounded between xovers or starting / ending points
    '''
    def __init__(self):
        #cadnano properties
        self.vh
        self.index_5p
        self.index_3p
        self.is_fwd

        #for carbs usage
        self.pointers_list      = []                   # Ordered list of pointers in the form [vh, index, is_fwd] for each nucleotide in strand

class DSNucleotide:
    '''
    Fwd and Rev (sense/antisense) nucleotides making up a double strand nucleotide
    '''
    def __init__(self):
        self.fwd_nucleotide     = None                 # Nucleotide in forward direction (reference frame)
        self.rev_nucleotide     = None                 # Nucleotide in reverse direction
        self.type               = 1                    # 1: double strand (dsDNA)
        self.visited            = False                # To be used by depth-first-search
        self.rigid              = False
        self.skip               = None                 # Whether this nucleotide is a skip in cadnano

        #Connections
        self.rigid_connections  = []                   # List of rigid connections
        self.skip_connections   = []                   # List of soft connections

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
        self.strand                       = None      # Nucleotide's strand number
        self.vh                           = None      # Nucleotide's virtual helix
        self.skip                         = False     # Whether the nucleotide is a skip
        self.insertion                    = False     # Whether the nucleotide is a insertion

        self.next                         = None      # Next nucleotide in a strand, if existent
        self.previous                     = None      # Previous nucleotide in strand, if existent
        self.strand_end                   = False     # Whether this nucleotide is in the end of a strand
        self.oligo_end                    = False     # Whether this nucleotide is in the end of a oligo

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
