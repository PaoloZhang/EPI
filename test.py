import unittest
import code

class TestCase(unittest.TestCase):
  def setUp(self):
    return super().setUp()

  def tearDown(self):
    return super().tearDown()
  
  def testZero(self):
    s1 = [-1,9,3]
    s2 = [0]
    self.assertEqual( code.multiply(s1,s2),[0])
  
  def testPositiveNegative(self):
    s1 = [-1,9,3]
    s2 = [2,5,6]
    self.assertEqual( code.multiply(s1,s2),[-4,9,4,0,8])

  def testCanReach(self):
    A = [3,3,7,0,2,0,1]
    B =  [3,2,0,0,2,1]
   # self.assertEqual(code.can_reach_end(A),True)
    self.assertTrue(code.can_reach_end(A)) and self.assertFalse(code.can_reach_end(A))
  
  def test_remove_key(self):
    num1,key1,num2,key2 = [],1,[1,2,1,3,1],1
    len1 = code.remove_key(num1,key1)
    len2 = code.remove_key(num2,key2)
    self.assertTrue(len1 == 0 and num1==[] and len2 ==2 and num2[0:2]==[2,3])

  def test_get_single(self):
    A = [0,1,1]
    ret1 = code.get_single(A)
    B = [2,2,1,1,0]
    ret2 = code.get_single(B)
    C = [2,2,0,1,1]
    ret3 = code.get_single(C)
    D = [2,2,0,1,1,3,3]
    ret4 = code.get_single(D)
    E = [2,2,1,1,0,3,3]
    ret5 = code.get_single(E)
    self.assertTrue(ret1 == 0 and ret2 == 4 and ret3 == 2 and ret4 == 2 and
     ret5 == 4,str(ret1)+":"+str(ret2)+":"+str(ret3)+":"+str(ret4)+":"+str(ret5))


def multiplySuite():
  suite = unittest.TestSuite()
  #suite.addTest(TestCase('testPositiveNegative'))
  #suite.addTest(TestCase('testZero'))
  suite.addTests([TestCase('test_get_single')])
  return suite

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  runner.run(multiplySuite())