from hoomd import *
from hoomd import md
import numpy as np
import particlesFromPDB as fromPDB

##########################
###### VECTOR MATH #######
# magnitude of a vector
def magvect(v):
  magnitude = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
  return magnitude

# angle between vectors
def angle(v1,v2,degrees=False):
  ang = np.arccos(np.dot(v1,v2)/(magvect(v1)*magvect(v2)))
  if degrees == True:
      ang = ang*180/np.pi
  return ang

def projection(v1,v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    v1_hat = v1/v1_norm
    v2_hat = v2/v2_norm
    cosTheta = np.dot(v1,v2)/(v1_norm * v2_norm)
    projection = v1_norm * cosTheta * v2_hat
    return(projection)

def rejection(v1,v2):
    rejection = v1 - projection(v1,v2)
    return(rejection)
##########################

chains = fromPDB.getChainsFromPDB("/Users/damasceno/Desktop/2017-07-31/oxDNA.pdb")

chains[0].nucleotides[0].beads[0].position

c = []
b = []
a = []
for i in range(0, len(chains[0].nucleotides)):
    nucleotide = chains[0].nucleotides[i]
    c.append(nucleotide.beads[0].position[0])
    b.append(nucleotide.beads[1].position[0])

for i in range(len(c)):
    b = np.asarray(b)
    c = np.asarray(c)
    if i != len(c) - 1:
        b1_ref = b[i] - c[i]
        c2_ref = c[i+1] - c[i]
        aa = np.cross(b1_ref, c2_ref)
    else:
        b1_ref = b[i] - c[i]
        c2_ref = c[i-1] - c[i]
        aa = np.cross(c2_ref, b1_ref)
        aa = aa / np.linalg.norm(aa)
    a.append(list(aa))
a = np.asarray(a)

for i in range(len(c)-1):
    angle1 = angle(c[i+1]-c[0], a[i+1]-c[i+1],False)
    angle2 = angle(c[i+1]-c[0], b[i+1]-c[i+1],False)
    print(angle1,angle2)

def dihedral(v1,v2,v3,v4,degrees=False):
    normal = np.cross(v1-v2, v3-v2)
    dihedral = angle(normal, v4-v3,degrees)
    return(dihedral)

for i in range(len(c)-1):
    dihedral1 = dihedral(b[i],c[i],c[i+1],b[i+1],False)
    dihedral2 = dihedral(b[i],c[i],a[i],c[i+1],False)
    dihedral3 = dihedral(a[i],c[i],b[i],c[i+1],False)
    print(dihedral3)
