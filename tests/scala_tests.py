from nose.tools import *
from math import sqrt
from scala import scala

edo12 = """
! edo12.scl
!
12 tone equal temperament
 12
!
 100.0
 200.0
 300.0
 400.0
 500.0
 600.0
 700.0
 800.0
 900.0
 1000.0
 1100.0
 2
"""

testscale = """
! test.scl
!
This is a test
 5
!
 9/8 ! Pythagorean major second
 5/4 ! Just major third
 3/2 ! Perfect fifth
 5/3 ! Just major sixth
 2   ! Octave
"""

tolerance = 0.01 #permissible error in cents

def test_cents():
    assert abs(scala.cents("2")-1200)<tolerance
    assert abs(scala.cents("1200.0")-1200)<tolerance
    assert abs(scala.cents("4/2")-1200)<tolerance
    assert abs(scala.cents(str(sqrt(2))+"/1")-600)<tolerance
    assert abs(scala.cents("3/2")-701.9550008653874)<tolerance

def test_ratio():
    assert_equal(scala.ratio("1200.0"),2)
    assert_equal(scala.ratio("701.9550008653874"),1.5)

def test_cps():
    pass

def test_savarts():
    pass

def test_parse_expression():
    assert_equal(scala.parse_expression("3/2*2"),"3")
    assert_equal(scala.parse_expression("3/(2*2)"),"3/4")
    assert_equal(scala.parse_expression("1/(5*4/(3*7))"),"21/20")
    assert_equal(scala.parse_expression("3**4/(5*2**4)"),"81/80")
    assert_equal(scala.parse_expression("100.0+50"), "150.0")

def test_scale_str():
    scale = scala.get_scale(testscale)
    assert_equal(testscale.strip(), str(scale))

def test_scale_cents():
    scale = scala.get_scale_string(edo12)
    assert_equal(scale.cents(1),100)
    assert_equal(scale.cents(1.5),150)
    assert_equal(scale.cents(12),1200)
    assert_equal(scale.cents(11.5),1150)
    assert_equal(scale.cents(12.5),1250)
    assert_equal(scale.cents(-1), -100)
    assert_equal(scale.cents(-0.5),-50)

