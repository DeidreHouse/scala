'''
-----------------------------------------------------------------
This is my Python module for working with Scala files.
Initial design is oriented towards extending Csound and CsoundAC.
------------------------------------------------------------------
'''
# Copyright 2016 Dillon Ethier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
from math import log, floor, ceil
from fractions import Fraction, gcd
from numbers import Number
from functools import reduce  # because they removed it in Python 3
import os
import re


# For debugging purposes:
testscale = """
! test.scl
!
This is a test scale
 12
!
 18/17 ! 17-limit semitone
 9/8 ! Pythagorean major second
 7/6 ! Septimal minor third
 5/4 ! Major third
 4/3 ! Perfect fourth
 7/5 ! Septimal tritone
 3/2 ! Perfect fifth
 8/5 ! Minor sixth
 5/3 ! Major sixth
 7/4 ! Harmonic (half-flat minor) seventh
 15/8  ! Major seventh
 2 ! Octave
 """

_defaultpitch = 261.6255653  # (Hz) Equal-tempered Middle C for A4=440 Hz


class ScaleError(Exception):
    pass


def prod(ls):
    if isinstance(ls, Number):
        return ls
    elif ls:
        product = reduce(lambda x, y: x * y, ls, 1)
        if bool(product) == 0:
            return 0
        elif str(float(product)).endswith(".0"):
            return int(product)
        return product
    return 1


def ordinal(n):
    num = str(int(n))
    suffixes = {'1': 'st', '2': 'nd', '3': 'rd', '4': 'th', '5': 'th',
                '6': 'th', '7': 'th', '8': 'th', '9': 'th', '0': 'th'}
    if len(num) > 1 and num[-2] == "1":
        return num + "th"
    return num + suffixes[num[-1]]


def cents(expr):
    ''' Returns the cent value expressed by expr.'''
    if "/" in expr:
        nums = expr.split("/")
        num = float(nums[0])
        den = prod(map(float, nums[1:]))
        return 1200 * log(num/den, 2)
    elif "." in expr:
        return float(expr)
    else:
        return 1200 * log(int(expr), 2)


def ratio(expr):
    '''Returns the ratio expressed by expr.'''
    if "/" in expr:
        nums = expr.split("/")
        num = float(nums[0])
        den = prod(map(float, nums[1:]))
        return num / den
    elif "." in expr:
        return 2**(float(expr)/1200)
    else:
        return float(expr)

# why not?
def savarts(expr):
    '''Returns the number of savarts expressed by expr.'''
    return cents(expr)*log(2, 10)/1.2


def cps(expr, basepitch=_defaultpitch):
    '''
    Returns the Hz value of the frequency expressed by expr,
    relative to basepitch (default = 261.625567, Middle C)
    '''
    return ratio(expr)*basepitch

# some useful regexes

_int_form = re.compile(r"^[1-9]\d*$")

_frac_form = re.compile(r"^[1-9]\d*/[1-9]\d*$")

_centsum = re.compile(r"^\d+\.\d*([+-][1-9]\d*\.?\d*)*$")

# wrapper for fraction parsing function to handle parentheses
def _simplify_fraction(func):
    def remove_parentheses(args):
        def invert(string):
            if _int_form.match(string):
                return string
            elif _frac_form.match(string):
                halves = string.split("/")
                return halves[0]+"*"+halves[1]
            else:
                raise SyntaxError("Must be valid fraction: "+str(string))
        if "(" not in args and ")" not in args:
            return func(args)
        parens = []
        numopen = 0
        # Check to see if parentheses are balanced:
        for i, s in enumerate(args):
            if s == "(":
                if numopen == 0:
                    parens.append([i])
                numopen += 1
            elif s == ")":
                if numopen < 1:
                    raise SyntaxError("Unbalanced parentheses: " + str(args))
                elif numopen == 1:
                    parens[-1].append(i)
                numopen -= 1
        if numopen:
            raise SyntaxError("Unbalanced parentheses: parentheses are still"
                              + "open at the end of the expression: "
                              + str(args))
        else:
            ans = args[:parens[0][0]]
            for i, s in enumerate(parens):
                if args[s[1]+1:s[1]+3] != "**":
                    inner = args[s[0]+1:s[1]]
                    if s[0] > 1 and args[s[0]-2:s[0]] == "**" and "/" in inner:
                        raise SyntaxError("Can't have fractional exponents")
                    if s[0] > 0 and args[s[0]-1] == "/":
                        ans += invert(remove_parentheses(inner))
                    else:
                        ans += remove_parentheses(inner)
                else:
                    ind = s[1]+3
                    power = args[ind]
                    if power.isdigit():
                        ind += 1
                        while ind < len(args) and args[ind].isdigit():
                            power += args[ind]
                            ind += 1
                    elif power == "(":
                        exponent = args[parens[i+1][0]+1:parens[i+1][1]]
                        if "/" in exponent:
                            raise SyntaxError("Can't have "
                                              "fractional exponents.")
                        else:
                            power = remove_parentheses(exponent)
                    else:
                        raise SyntaxError("Expected a number or (, got "
                                          + power)
                    subans = remove_parentheses(args[s[0]+1:s[1]]).split("/")
                    if len(subans) == 2:
                        if s[0] > 0 and args[s[0]-1] == "/":
                            ans += (str(int(subans[0])**int(power))
                                    + "*" + subans[1])
                        else:
                            ans += (str(int(subans[0])**int(power))
                                    + "/" + subans[1])
                    elif len(subans) == 1:
                        ans += str(int(subans[0]))
                    else:
                        err = ("Something broke: "
                               + str(remove_parentheses(args[s[0]+1:s[1]]))
                               + " should be an integer or fraction.")
                        raise SyntaxError(err)
                if i == len(parens)-1:
                    ans += args[s[1]+1:]
                else:
                    ans += args[s[1]+1:parens[i+1][0]]
            return func(ans)
    return remove_parentheses

# more complicated regexs

_numfactorform = re.compile(r"""
^[1-9]\d*  # an integer at the beginning
(?=[*/])   # followed by * or /
(?!\*\*)   # but not followed by **
|          # or
^[1-9]\d*$ # the whole string is an integer
|          # or
(?<=\*)    # preceded by *
(?<!\*\*)  # but not preceded by **
[1-9]\d*$  # followed by an integer at the end
|          # or
(?<=\*)    # preceded by *
(?<!\*\*)  # but not preceded by **
[1-9]\d*   # an integer
(?=[*/])   # followed by * or /
(?!\*\*)   # but not followed by **
""", re.X)

_numpowerform = re.compile(r"""
^[1-9]\d* # an integer at the beginning
\*\*      # **
[1-9]\d*  # an integer
(?=[*/])  # followed by * or /
(?!\*\*)  # but not followed by **
|         # or
^[1-9]\d* # an integer at the beginning
\*\*      # **
[1-9]\d*$ # an integer at the end
|         # or
(?<=\*)   # preceded by *
(?<!\*\*) # but not preceded by **
[1-9]\d*  # an integer
\*\*      # **
[1-9]\d*$ # an integer at the end
|         # or
(?<=\*)   # preceded by *
(?<!\*\*) # but not preceded by **
[1-9]\d*  # an integer
\*\*      # **
[1-9]\d*  # an integer
(?=[*/])  # followed by * or /
(?!\*\*)  # but not followed by **
""", re.X)

_denomfactorform = re.compile(r"""
(?<=/)    # preceded by /
[1-9]\d*$ # an integer at the end
|         # or
(?<=/)    # preceeded by /
[1-9]\d*  # an integer
(?=[*/])  # followed by * or /
(?!\*\*)  # but not followed by **
""", re.X)

_denompowerform = re.compile(r"""
(?<=/)    # preceded by /
[1-9]\d*  # an integer
\*\*      # **
[1-9]\d*$ # an integer at the end
|         # or
(?<=/)    # preceded by /
[1-9]\d*  # an integer
\*\*      # **
[1-9]\d*  # an integer
(?=[*/])  # followed by * or /
(?!\*\*)  # but not followed by **
""", re.X)

_validinput = re.compile(r"""
^[1-9]\d*        # an integer at the beginning
(\*\*[1-9]\d*)?  # possibly raised to an integer power
([*/][1-9]\d*    # * or / an integer
(\*\*[1-9]\d*)?) # possibly raised to an integer power
*$               # repeated any number of times before the end
""", re.X)

# Only use eval() after checking against this
_arithsymbols = re.compile(r"^[0-9+\-*/.)(]+$")

_int_float = re.compile(r"^[1-9]\d*\.0$")

_imperfect_frac_form = re.compile(r"^[1-9]\d*\.\d*/[1-9]\d*$")


@_simplify_fraction
def parse_fraction(input):
    """Simplifies the expression into a fraction."""

    if not _validinput.match(input):
        raise SyntaxError("Invalid input: can't parse "
                          + str(input) + " as a fraction.")

    def pow_(string):
        parts = string.split("**")
        return str(int(parts[0])**int(parts[1]))

    numfactors = (_numfactorform.findall(input)
                  + list(map(pow_, _numpowerform.findall(input))))
    denomfactors = (_denomfactorform.findall(input)
                    + list(map(pow_, _denompowerform.findall(input))))
    if len(numfactors) == 1:
        numerator = int(numfactors[0])
    else:
        numerator = prod(map(int, numfactors))
    if denomfactors == []:
        return str(numerator)
    else:
        if len(denomfactors) == 1:
            denominator = int(denomfactors[0])
        else:
            denominator = prod(map(int, denomfactors))
        d = gcd(numerator, denominator)
        if d == denominator:
            return str(numerator//d)
        else:
            return str(numerator//d) + "/" + str(denominator//d)


def parse_expression(input):
    '''Simplify an expression into a valid Scala scale entry.'''
    input = re.sub(r"\s+", "", input, flags=re.U)
    if _centsum.match(input):
        return str(eval(input, {'__builtins__': {}}))
    elif _frac_form.match(input) or _imperfect_frac_form.match(input):
        return input
    else:
        try:
            return parse_fraction(input)
        except (SyntaxError):
            if _arithsymbols.match(input):
                try:
                    calc = "{:.6f}".format(eval(input))
                    if _int_float.match(calc):
                        return calc[0:-2]
                    elif calc.isdigit():
                        return calc
                    else:
                        return calc + "/1"
                except (SyntaxError, ZeroDivisionError):
                    raise SyntaxError("Invalid input: " + input)
            else:
                raise SyntaxError("Invalid input: " + input)


def parse_line(input):
    """Parses a line of a scala file.

    Returns a pair consisting of the parsed expression and
    the corresponding comment (empty string if no comment)"""
    if "\n" in input:
        raise SyntaxError("Input must be a single line")
    broken = input.split("!", 1)
    expr = parse_expression(broken[0])
    if len(broken) == 2:
        return expr, broken[1].strip()
    return expr, ""


def combine_intervals(*args):
    args = map(parse_expression, args)
    if any(["." in x and "/" not in x for x in args]):
        return str(sum([cents(x) for x in args]))
    elif any(["." in x for x in args]):
        numerator = str(prod(map(ratio, args)))
        if numerator.endswith(".0"):
            return numerator[:-2]
        else:
            return numerator + "/1"
    else:
        return parse_fraction("*".join(args))


def invert_interval(interval, period):
    if (("." in interval and "/" not in interval)
        or ("." in period and "/" not in period)):
        return str(cents(period)-cents(interval))
    elif "." in interval or "." in period:
        return str(ratio(period)/ratio(interval)) + "/1"
    else:
        return parse_fraction(period + "*1/({0})".format(interval))


class Scale(object):

    def __init__(self, name, exprs, description, comments={}):
        self.name = name
        self.entries = ["1"] + exprs
        self.desc = description
        self.comments = {foo: "" for foo in self.entries}
        self.comments.update({k: comments[k] for k in comments if k in exprs})

    def __cmp__(self, other):
        if cmp(len(self), len(other)):
            return cmp(len(self), len(other))
        elif cmp(cents(self.period()), cents(other.period())):
            return cmp(cents(self.period()), cents(other.period()))
        else:
            def _roundcents(x):
                return round(cents(x), 6)
            return cmp(map(_roundcents, self.entries),
                       map(_roundcents, other.entries))

    def __eq__(self, other):
        '''
        Compares scale entries only, name and description are ignored.
        '''
        def _roundcents(x):
            return round(cents(x), 6)
        return (map(_roundcents, self.entries)
                == map(_roundcents, other.entries))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        combination = self.entries[1:] \
                     + map(lambda x: combine_intervals(x, self.period()),
                           other.entries[1:])
        return Scale(self.name + "+" + other.name,
                     combination, "Sum of " + self.name + " and " + other.name)

    def __mul__(self, num):
        if int(num) == 0:
            return Scale("empty", [], "empty")
        result = reduce(lambda x, y: x + y, [self]*int(num))
        return result

    def __str__(self):
        lines = []
        maxlen = 0
        if any([self.comments[entry] for entry in self.entries[1:]]):
            maxlen = max([len(entry) for entry in self.entries[1:]])
        for entry in self.entries[1:]:
            if self.comments[entry]:
                lines.append(entry.ljust(maxlen)
                             + " ! " + self.comments[entry])
            else:
                lines.append(entry)
        return ("! {0}\n!\n{1}\n {2}\n!\n {3}"
                ).format(self.name + ".scl",
                         self.desc, len(self),
                         "\n ".join(lines))

    def __repr__(self):
        return ("Scale(name=\"{0}\", exprs={1},"
                + " description=\"{2}\", comments={3})").format(
                self.name, self.entries[1:], self.desc, self.comments)

    def __len__(self):
        return len(self.entries)-1

    def __getitem__(self, key):
        expr = self.entries[key]
        comment = self.comments[expr]
        if comment:
            return expr + " ! " + comment
        return self.entries[key]

    def __getslice__(self, start, stop):
        exprs = [invert_interval(self.entries[start], entry)
                 for entry in self.entries[start+1:stop+1]]
        return Scale(self.name, exprs, self.desc)

    def __setitem__(self, key, val):
        del self.comments[self.entries[key]]
        expr, comment = parse_line(val)
        self.entries[key] = expr
        self.comments[self.entries[key]] = comment

    def __delitem__(self, key):
        del self.comments[self.entries[key]]
        del self.entries[key]

    def __iter__(self):
        return iter(self.entries[1:])

    def __reversed__(self):
        '''Returns an inverted version of self'''
        inversion = map(lambda x: invert_interval(x, self.period()),
                        self.entries[len(self)-1:0:-1])
        return Scale(self.name + "_inverted",
                     inversion + [self.period()], self.desc + "_inverted")

    def __contains__(self, expr):
        octave = floor(cents(expr)/cents(self.period()))
        new = cents(expr)-octave*cents(self.period())
        for entry in self.entries[:-1]:
            if abs(cents(entry)-new) < 10**-6:
                return True
        return False

    def period(self):
        return self.entries[-1]

    def append(self, val):
        expr, comment = parse_line(val)
        self.entries.append(expr)
        self.comments[expr] = comment

    def insert(self, i, val):
        expr, comment = parse_line(val)
        self.entries.insert(i, expr)
        self.comments[expr] = comment

    def ascending(self):
        '''
        Returns an ascending version of self.
        '''
        return Scale(self.name,
                     sorted(self.entries[1:], key=lambda x: ratio(x)),
                     self.desc, self.comments)

    def mode(self, n=0):
        '''
        Returns a scale representing the nth mode of self.
        '''
        n %= len(self)
        pre = self + self
        result = pre[n:len(self)+n]
        result.name = self.name
        result.desc = ordinal(n) + " mode of " + self.desc
        return result

    def cents(self, degree):
        '''
        Returns the cent value of the given scale degree.

        Will interpolate for non-integer input.
        '''
        deg = int(floor(degree))
        octave = deg//len(self)
        freq1 = self.entries[deg % len(self)]
        freq2 = self.entries[(deg % len(self)) + 1]
        return (cents(freq1) * (1-(degree % 1)) + cents(freq2) * (degree % 1)
                + octave * cents(self.period()))

    def ratio(self, degree):
        '''
        Returns the frequency ratio corresponding to the given scale degree.

        Will interpolate for non-integer input.
        '''
        deg = int(floor(degree))
        octave = deg//len(self)
        freq1 = self.entries[deg % len(self)]
        freq2 = self.entries[(deg % len(self))+1]
        return (ratio(freq1)**(1-(degree % 1)) * ratio(freq2)**(degree % 1)
                * ratio(self.period())**octave)

    def cps(self, degree, basepitch=_defaultpitch):
        '''
        Returns the frequency in Hz corresponding to the given scale degree.

        basepitch default is 261.625567, Middle C.
        '''
        return round(self.ratio(degree)*basepitch, 6)

    def cpspch(self, input, basepitch=_defaultpitch):
        '''
        Converts csound's pch notation to a frequency in Hz.
        '''
        oct = floor(int(input)) - 8
        numdigits = int(log(len(self), 10)) + 1
        degree = int(round(input % 1, numdigits)*10**(numdigits))
        return ratio(self.period())**oct * self.cps(degree, basepitch)

    def cpsmidinn(self, input, basekey=60, basepitch=_defaultpitch):
        '''Converts the midi note number to a frequency in Hz.'''
        return self.cps(int(input)-int(basekey), basepitch)

    # def normalize(self, period = "2"):

    def standardize(self):
        '''Converts all imperfect fractions to cents'''
        for n, i in enumerate(self.entries):
            if "." in i and "/" in i:
                self.entries[n] = str(cents(i))

    def standardized(self):
        '''
        Returns a standardized copy of self.

        All imperfect fractions are converted to cents.
        '''
        standard_list = []
        for i in self.entries[1:]:
            if "." in i and "/" in i:
                standard_list.append(str(cents(i)))
            else:
                standard_list.append(i)
        return Scale(self.name, standard_list, self.desc)

    def csound_table(self, basepitch=_defaultpitch, basekey=60):
        '''
        Writes self to a .txt file readable by csound as a table object.

        Use with the cpstun function in Csound.
        Optional arguments:
        basepitch (default = 261.625567 Hz middle C)
        basekey (default = 60, corresponds to basepitch)
        '''
        ratios = map(ratio, self.entries)
        target = self.name + ".txt"
        with open(target, "w+") as table:
            table.write("="*7
                        + " TABLE X SIZE: {0} values".format(len(self) + 4)
                        + "="*int(8-log(len(self) + 4, 10)-1) + "\n")
            table.write("flen: {0}\n".format(len(self)+4))
            table.write("lenmask: 0\n")
            table.write("lobits: 0\n")
            table.write("lomask: 0\n")
            table.write("lodiv: 0\n")
            table.write("cvtbas: 0\n")
            table.write("cpscvt: 0\n")
            table.write("loopmode1: 0\n")
            table.write("loopmode2: 0\n")
            table.write("begin1: 0\n")
            table.write("end1: 0\n")
            table.write("begin2: 0\n")
            table.write("end2: 0\n")
            table.write("soundend: 0\n")
            table.write("flenfrms: 0\n")
            table.write("nchnls: 1\n")
            table.write("fno: \n")
            table.write("gen01args.gen01: 0\n")
            table.write("gen01args.ifilno: 0\n")
            table.write("gen01args.iskptim: 0\n")
            table.write("gen01args.iformat: 0\n")
            table.write("gen01args.channel: 0\n")
            table.write("gen01args.sample_rate: 0\n")
            table.write("-"*9 + "END OF HEADER" + "-"*14 + "\n")
            table.write(str(len(self)) + "\n")
            table.write(str(ratios.pop()) + "\n")
            table.write(str(basepitch) + "\n")
            table.write(str(int(basekey)) + "\n")
            for r in ratios:
                table.write("{0:.6f}\n".format(r))
            table.write("0.000000" + "\n")
            table.write("-"*9 + "END OF TABLE" + "-"*15)

# works for me but is terribly written- NEEDS a re-write
    def write_scale(self, path=None):
        '''
        Writes a scale object to a .scl file in a location specified by path.
        '''
        if path is not None:
            if len(os.path.dirname(path)):
                directory = os.path.dirname(path)
                title, ext = os.path.splitext(os.path.basename(path))
                if len(ext) and ext != ".scl":
                    raise ScaleError("Output file must be a .scl file")
                elif not len(ext):
                    directory += "/" + title + "/"
                    title = self.name
                else:
                    directory += "/"
            else:
                title, ext = os.path.splitext(os.path.basename(path))
                if len(ext) and ext != ".scl":
                    raise ScaleError("Output file must be a .scl file")
                elif not len(ext) and len(title):
                    directory = title + "/"
                    title = self.name
                elif not len(title):
                    directory = "./"
                    title = self.name
        else:
            directory = "./"
            title = self.name
        if os.path.isfile(directory + title + ".scl"):
            it = 1
            while os.path.isfile(directory + title + "_" + str(it) + ".scl"):
                it += 1
            outname = directory + title + "_" + str(it) + ".scl"
        else:
            outname = directory + title + ".scl"

        scl = open(outname, 'w')
        scl.write(str(self))
        scl.close()


def get_scale_string(str_):
    '''
    Returns a Scale object.

    str_ must be a string consisting of the contents of a Scala file
    '''
    size = 1
    ratios = []
    sizeindex = 0
    desc = ""
    lines = str_.strip().split("\n")
    name = lines[0].split(".")[0].split("!")[1].strip()
    for i, line in enumerate(lines):
        try:
            size = int(line)
            sizeindex = i
            break
        except (ValueError):
            desc = line.strip()
    comments = {}
    for i, line in [(i, l) for i, l in enumerate(lines)
                    if i > sizeindex and l[0] != "!"]:
        try:
            expr, comment = parse_line(line)
            ratios.append(expr)
            comments[expr] = comment
        except (SyntaxError, ZeroDivisionError):
            raise ScaleError("Invalid input: Failed to parse "
                             + first + "on line " + str(i))
    if size != len(ratios):
        raise ScaleError("Invalid input: size must match number of entries."
                         "(size = {0}, no. entries = {1})").format(size,
                                                                   len(ratios))
    return Scale(name, ratios, desc, comments)


def get_scale_file(path):
    '''
    Reads a scale from a .scl file and returns a Scale object.
    '''
    filename = os.path.normpath(path)
    suffix = os.path.splitext(filename)[1]
    if suffix == ".scl":
        with open(filename) as fp:
            return get_scale_string(fp.read())
    else:
        raise ScaleError("Expected a .scl file.")


def get_scale(str_):
    '''
    Returns a scale object.

    str_ can either be the name of a .scl file
    or a string consisting of the contents of a .scl file
    '''
    if "\n" in str_:
        return get_scale_string(str_)
    else:
        return get_scale_file(str_)


def interval_list(*args, **kwargs):
    '''
    Returns a scale object constructed from a list of intervals.

    Automatically reorders the scale unless you specify ascending=False
    '''
    name = kwargs.get('name', 'intervals')
    desc = kwargs.get('description', "Generated from a list")
    ascend = kwargs.get('ascending', True)
    data = map(parse_line, args)
    result = Scale(name, [x[0] for x in data],
                   desc, {x[0]: x[1] for x in data})
    if ascend:
        return result.ascending()
    else:
        return result


def equal_temp(nosteps=12, interval="2"):
    '''
    Returns a scale object corresponding to an equal temperament.

    Arguments:
    nosteps = Number of notes in one period of the scale.
              Input coerced to integer.
    interval (default="2") = the period of the scale,
                 as a frequency ratio expressed in Scala notation.
    '''
    if nosteps < 1:
        raise ScaleError("Invalid step size.")
    period = ratio(parse_expression(interval))
    if period <= 1:
        raise ScaleError("Dividend interval must be ascending.")
    steps = int(nosteps)
    name = "{0}_equal_{1}".format(steps, interval)
    desc = ""
    if period == 2:
        desc = "{0} equal divisions of the octave".format(steps)
    else:
        desc = "{0} equal divisions of {1}".format(steps,
                                                   parse_expression(interval))
    ratios = []
    for i in range(1, steps+1):
        ratios.append(str(cents(interval)*i/steps))
    ratios[-1] = interval
    return Scale(name, ratios, desc)

# def mos(*args):
    # '''Not yet implemented- need to learn what it means, first'''
