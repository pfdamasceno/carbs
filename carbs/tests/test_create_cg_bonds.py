import carbs
import unittest
import os

# unit tests for init.create_random
class create_CG_bonds (unittest.TestCase):
    def setUp(self):
        print

    # tests the creation of adjacent bonds for files with skip 
    def test_create_adjacent_bonds_with_skip(self):
        option.set_notice_level(1);
        self.assert_(hoomd.context.options.notice_level == 1);

        option.set_notice_level(10);
        self.assert_(hoomd.context.options.notice_level == 10);

        self.assertRaises(RuntimeError, option.set_notice_level, 'foo');

    # checks for an error if initialized twice
    def test_inittwice(self):
        init.create_random(N=100, phi_p=0.05);
        self.assertRaises(RuntimeError, init.create_random, N=100, phi_p=0.05);

    # test that angle,dihedral, and improper types are initialized correctly
    def test_angleA(self):
        s = init.create_random(N=100, phi_p=0.05);
        snap = s.take_snapshot(all=True);
        self.assertEqual(len(snap.bonds.types), 0);
        self.assertEqual(len(snap.impropers.types), 0);
        self.assertEqual(len(snap.angles.types), 0);
