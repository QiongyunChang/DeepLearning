# -*- coding: utf-8 -*-
"""assignment_1.ipynb



This tutorial was originally written by [Justin Johnson](https://web.eecs.umich.edu/~justincj/) for cs231n. It was adapted as a Jupyter notebook for cs228 by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335).

This version has been adapted for Colab by Kevin Zakka for the Spring 2020 edition of [cs231n](https://cs231n.github.io/). It runs Python3 by default.

##Introduction

Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib) it becomes a powerful environment for scientific computing.

We expect that many of you will have some experience with Python and numpy; for the rest of you, this section will serve as a quick crash course both on the Python programming language and on the use of Python for scientific computing.

Some of you may have previous knowledge in Matlab, in which case we also recommend the numpy for Matlab users page (https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html).


This notebook will walk you through many of the important features of Python that you will need to use throughout the semester. In some cells you will see code blocks that look like this:

```python
##############################################################################
# TODO: Write the equation for a line
##############################################################################
pass
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
```

You should replace the `pass` statement with your own code and leave the blocks intact, like this:

```python
##############################################################################
# TODO: Instructions for what you need to do
##############################################################################
y = m * x + b
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
```

When completing the notebook, please adhere to the following rules:
- Do not write or modify any code outside of code blocks
- Do not add or delete any cells from the notebook. You may add new cells to perform scatch work, but delete them before submitting.
- Run all cells before submitting. You will only get credit for code that has been run.

In this tutorial, we will cover:

* Basic Python: Basic data types (Containers, Lists, Dictionaries, Sets, Tuples), Functions, Classes
* Numpy: Arrays, Array indexing, Datatypes, Array math, Broadcasting
* Matplotlib: Plotting, Subplots, Images
* IPython: Creating notebooks, Typical workflows

## A Brief Note on Python Versions

As of Janurary 1, 2020, Python has [officially dropped support](https://www.python.org/doc/sunset-python-2/) for `python2`. We'll be using Python 3.7 for this iteration of the course. You can check your Python version at the command line by running `python --version`. In Colab, we can enforce the Python version by clicking `Runtime -> Change Runtime Type` and selecting `python3`. Note that as of April 2020, Colab uses Python 3.6.9 which should run everything without any errors.
"""

!python --version

"""##Basics of Python

Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable. As an example, here is an implementation of the classic quicksort algorithm in Python:
"""

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

"""###Basic data types

####Numbers

Integers and floats work as you would expect from other languages:
"""

x = 3
print(x, type(x))

print(x + 1)   # Addition
print(x - 1)   # Subtraction
print(x * 2)   # Multiplication
print(x ** 2)  # Exponentiation

x += 1
print(x)
x *= 2
print(x)

y = 2.5
print(type(y))
print(y, y + 1, y * 2, y ** 2)

"""Note that unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.

Python also has built-in types for long integers and complex numbers; you can find all of the details in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#numeric-types-int-float-long-complex).

####Booleans

Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (`&&`, `||`, etc.):
"""

t, f = True, False
print(type(t))

"""Now we let's look at the operations:"""

print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;

"""####Strings"""

hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter
print(hello, len(hello))

hw = hello + ' ' + world  # String concatenation
print(hw)

hw12 = '{} {} {}'.format(hello, world, 12)  # string formatting
print(hw12)

"""String objects have a bunch of useful methods; for example:"""

s = "hello"
print(s.capitalize())  # Capitalize a string
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces
print(s.center(7))     # Center a string, padding with spaces
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
print('  world '.strip())  # Strip leading and trailing whitespace

"""You can find a list of all string methods in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#string-methods).

###Containers

Python includes several built-in container types: lists, dictionaries, sets, and tuples.

####Lists

A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:
"""

xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"

xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)

xs.append('bar') # Add a new element to the end of the list
print(xs)

x = xs.pop()     # Remove and return the last element of the list
print(x, xs)

"""As usual, you can find all the gory details about lists in the [documentation](https://docs.python.org/3.7/tutorial/datastructures.html#more-on-lists).

####Slicing

In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:
"""

nums = list(range(5))    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9]  # Assign a new sublist to a slice
print(nums)         # Prints "[0, 1, 8, 9, 4]"

"""####Loops

You can loop over the elements of a list like this:
"""

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

"""If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:"""

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))

"""####List comprehensions:

When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:
"""

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)

"""You can make this code simpler using a list comprehension:"""

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)

"""**List** comprehensions can also contain conditions:"""

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)

"""####Dictionaries

A dictionary stores (key, value) pairs, similar to a `Map` in Java or an object in Javascript. You can use it like this:
"""

d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"

d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"

print(d['monkey'])  # KeyError: 'monkey' not a key of d

print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"

del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"

"""You can find all you need to know about dictionaries in the [documentation](https://docs.python.org/2/library/stdtypes.html#dict).

It is easy to iterate over the keys in a dictionary:
"""

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))

"""Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:"""

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

"""####Sets

A set is an unordered collection of distinct elements. As a simple example, consider the following:
"""

animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"

animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;

animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))

"""_Loops_: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:"""

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))

"""Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:"""

from math import sqrt
print({int(sqrt(x)) for x in range(30)})

"""####Tuples

A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:
"""

d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d[t])       
print(d[(1, 2)])

t[0] = 1

"""###Functions

Python functions are defined using the `def` keyword. For example:
"""

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

"""We will often define functions to take optional keyword arguments, like this:"""

def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')
hello('Fred', loud=True)

"""###Your turn

#### Fizz, Buzz, FizzBuzz! (10 points)
If we list all of the natural numbers under 41 that are a multiple of 3 or 5, we get

```
 3,  5,  6,  9, 10, 12, 15,
18, 20, 21, 24, 25, 27, 30,
33, 35, 36, 39, 40
```

The sum of these numbers is 408.

Find the sum of all the multiples of 3 or 5 below 1001. As a sanity check, the last two digits of the sum should be `68`.
"""

def fizzbuzz(n):
  """Returns the sum of all numbers less than n divisible by 3 or 5."""
  res = None
  ##############################################################################
  # TODO: Write code to return the sum of all numbers less than 3 divisible by #
  # 3 or 5.                                                                    #
  ##############################################################################
  # Replace "pass" statement with your code
  res = 0
  for x in range(n):
    if (x % 3 == 0 or x % 5 == 0) :
      res =  res + x   
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return res

print(fizzbuzz(41))  # => 408
print(fizzbuzz(1001))

"""#### SUPER tic-tae-toe board (10 points)

Write a program that prints out a SUPER tic-tac-toe board.

```
  |  |  H  |  |  H  |  |  
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
========+========+========
  |  |  H  |  |  H  |  |  
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
========+========+========
  |  |  H  |  |  H  |  | 
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
--+--+--H--+--+--H--+--+--
  |  |  H  |  |  H  |  |  
```

You'll find that there might be many ways to solve this problem. Which do you think is the most 'pythonic?' Talk to someone next to you about your approach to this problem. Remember the Zen of Python!
"""

def print_super_tictactoe():
  """Print an empty SUPER tic-tac-toe board."""
  ##############################################################################
  # TODO: Print an empty SUPER tic-tac-toe board depicted above.               #
  ##############################################################################
  # Replace "pass" statement with your code
  list = 0
  for i in range(17):
    if ( i % 2 == 0 or i == 0 ):
      print("  |  |  H  |  |  H  |  |  ")
    elif ( i == 5 or i % 11 == 0):
      print("========+========+=======")
    elif ( i % 2 == 1 or i % 5 == 0 or i % 11 != 0 ):
      print("--+--+--H--+--+--H--+--+--")

      

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################

print_super_tictactoe()

"""# #### Pascal's Triangle (10 points)
Write a function that generates the next level of [Pascal's triangle](https://en.wikipedia.org/wiki/Pascal%27s_triangle) given a list that represents a row of Pascal’s triangle.

```
generate_pascal_row([1, 2, 1]) -> [1, 3, 3, 1]
generate_pascal_row([1, 4, 6, 4, 1]) -> [1, 5, 10, 10, 5, 1]
generate_pascal_row([]) -> [1]
```

As a reminder, each element in a row of Pascal's triangle is formed by summing the two elements in the previous row directly above (to the left and right) that elements. If there is only one element directly above, we only add that one. For example, the first 5 rows of Pascal's triangle look like:

```
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

*Hint: Check out the diagram below. How could you use this insight to help complete this problem?*

```
  0 1 3 3 1
+ 1 3 3 1 0
-----------
  1 4 6 4 1
```
"""

def generate_pascal_row(row):
  """Generate the next row of Pascal's triangle."""
  ##############################################################################
  # TODO: Write code to generate a row of Pascal's triangle.                  #
  ##############################################################################
  # Replace "pass" statement with your code
  list1= [] # 新增一個list
  for i in range(len(row)):
    if( i == 0 or i == len(row) ):
      list1.append(1)
    else : 
      sum = row[i] + row[i-1]
      list1.append(sum)
  list1.append(1) # 給最後一個值
  return list1
    
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################

print(generate_pascal_row([1, 2, 1]))  # => [1, 3, 3, 1]
print(generate_pascal_row([1, 4, 6, 4, 1]))  # => [1, 5, 10, 10, 5, 1]
print(generate_pascal_row([]))  # => [1]

"""#### Greatest common divisor (10 points)

Write a function to compute the [GCD](https://en.wikipedia.org/wiki/Greatest_common_divisor) of two positive integers. You can freely use the fact that `gcd(a, b)` is mathematically equal to `gcd(b, a % b)`, and that `gcd(a, 0) == a`.

You can assume that `a >= b` if you'd like.

It is possible to accomplish this in three lines of Python code (or with extra cleverness, even fewer!). Consider exploiting tuple packing and unpacking!

*Note: The standard library has a `gcd` function. Avoid simply importing that function and using it here - the goal is to practice with tuple packing and unpacking!*
"""

def gcd(a, b):
  """Compute the GCD of two positive integers."""
  ##############################################################################
  # TODO: Write code to compute GCD of two positive integers.                  #
  ##############################################################################
  # Replace "pass" statement with your code
  while b != 0 :
    a, b = b, a % b
  return a


  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
    
print(gcd(10, 25)) # => 5
print(gcd(14, 15)) # => 1
print(gcd(3, 9)) # => 3
print(gcd(1, 1)) # => 1

"""#### Comprehensions practices (10 points)

Write comprehensions to transform the input data structure into the output data structure:

```
[0, 1, 2, 3] -> [1, 3, 5, 7]  # Double and add one
['apple', 'orange', 'pear'] -> ['A', 'O', 'P']  # Capitalize first letter
['apple', 'orange', 'pear'] -> ['apple', 'pear']  # Contains a 'p'

["TA_sam", "student_poohbear", "TA_guido", "student_htiek"] -> ["sam", "guido"]
['apple', 'orange', 'pear'] -> [('apple', 5), ('orange', 6), ('pear', 4)]

['apple', 'orange', 'pear'] -> {'apple': 5, 'orange': 6, 'pear': 4}
```
"""

nums = [0, 1, 2, 3]
fruits = ['apple', 'orange', 'pear']
people = ["TA_sam", "student_poohbear", "TA_guido", "student_htiek"]

##############################################################################
# TODO: Write comprehensions to obtain the desired results. Be sure to print #
# out the results.                                                           #
##############################################################################
# Replace "pass" statement with your code

# 第一題
for i in range(len(nums)):
  nums[i] = nums[i] * 2 + 1 
print(nums)

# 第二題
show = []
# print(type(list))
for i in fruits:
    show.append(i[0].upper())
print(show)

# 第三題
has_p =[] 
for letter in range(len(fruits)):
  if( 'p' in fruits[letter]): 
    has_p.append(fruits[letter])
print(has_p)

# 第四題
name = []
matchers = ['TA']
matching = [i for i in people if any(j in i for j in matchers)]
for i in matching :
  name.append(i[3:])
print(name)


# 第五題
b = []
for i in range(len(fruits)):
  b.append(len(fruits[i]))
d = zip(fruits,b)
print(list(d))


# 第六題
a = []
lengh = 0
for i in range(len(fruits)):
  a.append(len(fruits[i]))
d = zip(fruits,a)
print(dict(d))
 


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

"""#### Dictionaries (10 points)

Write a function that properly reverses the keys and values of a dictionary - each key (originally a value) should map to a collection of values (originally keys) that mapped to it. For example,

```
flip_dict({"CA": "US", "NY": "US", "ON": "CA"})
# => {"US": ["CA", "NY"], "CA": ["ON"]}
```

Note: there is a data structure in the `collections` module from the standard library called `defaultdict` which provides exactly this sort of functionality. You provide it a factory method for creating default values in the dictionary (in this case, a list.) You can read more about `defaultdict` and other `collections` data structures [here](https://docs.python.org/3/library/collections.html).
"""

def flip_dict(input_dict):
  """Reverse the keys and values of a dictionary."""
  ##############################################################################
  # TODO: Write code to reverse the keys and values of a dictionary.           #
  ##############################################################################
  # Replace "pass" statement with your code
  
  from collections import defaultdict
  new_d = defaultdict(list)
  for key, value in input_dict.items():
      new_d[value].append(key)

  return new_d
  
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################

print(flip_dict({"CA": "US", "NY": "US", "ON": "CA"}))

"""###Classes

The syntax for defining classes in Python is straightforward:
"""

class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

"""### Your turn (10 points)

Let’s create a class to represent courses at NSYSU! A course will have three attributes to start: a department (like `"MIS"`), a course code (like `"583"` or `"205"`), and a title (like `"Deep Learning"`).
"""

class Course:
  def __init__(self, department, code, title):
    ##############################################################################
    # TODO: Create three instance variables.                  #
    ##############################################################################
    # Replace "pass" statement with your code
    self.department = department
    self.code = code
    self.title = title
    
  def info (self):
    print("Department:",self.department)
    print("Code:",self.code)
    print("Title:",self.title)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################


dl = Course("MIS", "583", "Deep Learning")
##############################################################################
# TODO: Print out the department, the course code, and the title of the      #
# course.                                                                    #
##############################################################################
# Replace "pass" statement with your code
dl.info() 
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

"""##Numpy

Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. If you are already familiar with MATLAB, you might find this [tutorial](http://wiki.scipy.org/NumPy_for_Matlab_Users) useful to get started with Numpy.

To use Numpy, we first need to import the `numpy` package:
"""

import numpy as np

"""###Arrays

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:
"""

a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)

print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])

"""Numpy also provides many functions to create arrays:"""

a = np.zeros((2,2))  # Create an array of all zeros
print(a)

b = np.ones((1,2))   # Create an array of all ones
print(b)

c = np.full((2,2), 7) # Create a constant array
print(c)

d = np.eye(2)        # Create a 2x2 identity matrix
print(d)

e = np.random.random((2,2)) # Create an array filled with random values
print(e)

"""###Array indexing

Numpy offers several ways to index into arrays.

Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:
"""

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)

"""A slice of an array is a view into the same data, so modifying it will modify the original array."""

print(a[0, 1])
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])

"""You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. Note that this is quite different from the way that MATLAB handles array slicing:"""

# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

"""Two ways of accessing the data in the middle row of the array.
Mixing integer indexing with slices yields an array of lower rank,
while using only slices yields an array of the same rank as the
original array:
"""

row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)

"""Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:"""

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))

"""One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:"""

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)

"""Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:"""

import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])

# We can do all of the above in a single concise statement:
print(a[a > 2])

"""For brevity we have left out a lot of details about numpy array indexing; if you want to know more you should read the documentation.

###Datatypes

Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:
"""

x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)

"""You can read all about numpy datatypes in the [documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

###Array math

Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:
"""

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

"""Note that unlike MATLAB, `*` is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:"""

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

"""You can also use the `@` operator which is equivalent to numpy's `dot` operator."""

print(v @ w)

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)

"""Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:"""

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

"""You can find the full list of mathematical functions provided by numpy in the [documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).

Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:
"""

print(x)
print("transpose\n", x.T)

v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)

"""###Broadcasting

Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:
"""

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

"""This works; however when the matrix `x` is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix `x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically, then performing elementwise summation of `x` and `vv`. We could implement this approach like this:"""

vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"

y = x + vv  # Add x and vv elementwise
print(y)

"""Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:"""

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

"""The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape `(3,)` due to broadcasting; this line works as if v actually had shape `(4, 3)`, where each row was a copy of `v`, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
3. The arrays can be broadcast together if they are compatible in all dimensions.
4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

If this explanation does not make sense, try reading the explanation from the [documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) or this [explanation](http://wiki.scipy.org/EricsBroadcastingDoc).

Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the [documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

Here are some applications of broadcasting:
"""

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:

print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:

print((x.T + w).T)

# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
print(x * 2)

"""Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.

This brief overview has touched on many of the important things that you need to know about numpy, but is far from complete. Check out the [numpy reference](http://docs.scipy.org/doc/numpy/reference/) to find out much more about numpy.

### Your turn

#### Sigmoid (10 points)

x could now be either a real number, a vector, or a matrix. The data structures we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays.
$$ \text{For } x \in \mathbb{R}^n \text{,     } sigmoid(x) = sigmoid\begin{pmatrix}
    x_1  \\
    x_2  \\
    ...  \\
    x_n  \\
\end{pmatrix} = \begin{pmatrix}
    \frac{1}{1+e^{-x_1}}  \\
    \frac{1}{1+e^{-x_2}}  \\
    ...  \\
    \frac{1}{1+e^{-x_n}}  \\
\end{pmatrix}\tag{1} $$
"""

def sigmoid(x):
  """ Compute the sigmoid of x
  Arguments:
  x -- A scalar or numpy array of any size

  Return:
  s -- sigmoid(x)
  """

  s = None
  ##############################################################################
  # TODO: compute the sigmoid of x.                                            #
  ##############################################################################
  # Replace "pass" statement with your code
  import math
  s = []
  for i in range(len(x)):
    output = 1 / (1 + math.exp(-x[i]))
    s.append(output)

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return s
import numpy as np
x = np.array([1, 2, 3])
print(sigmoid(x))

"""#### Normalizing rows (10 points)

A common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to $ \frac{x}{\| x\|} $ (dividing each row vector of x by its norm).

For example, if $$x = 
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}\tag{3}$$ then $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix}\tag{4} $$and        $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}\tag{5}$$ Note that you can divide matrices of different sizes and it works fine: this is called broadcasting.


Implement normalize_rows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
"""

def normalize_rows(x):
  """
  Implement a function that normalizes each row of the matrix x (to have unit length).
    
  Argument:
  x -- A numpy matrix of shape (n, m)
    
  Returns:
  x -- The normalized (by row) numpy matrix. You are allowed to modify x.
  """

  ##############################################################################
  # TODO:                                                                      #
  # Step 1: Compute x_norm as the norm 2 of x.                                 #
  #         Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)      #
  # Step 2: Divide x by its norm
  ##############################################################################
  # Replace "pass" statement with your code

  norm = np.linalg.norm( x, ord=2, axis=1, keepdims=True )

  x = np.divide(x,norm)

  ### END CODE HERE ###
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ############################################################################## 
  return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = \n" + str(normalize_rows(x)))

"""#### Moving avarage (10 points)

Moving average is a calculation to analyze data points by creating a series of averages of different subsets of the full data set. It is often used in technical analysis of financial data, like stock prices.

Write a function to calculate the moving average by given window size.

For example, 
$$ x = \begin{bmatrix} 1,3,5,7,9 \end{bmatrix} $$

$$ n = 3 $$

$$ moving\_average(x,n) = \begin{bmatrix} \frac{1+3+5}{3} , \frac{3+5+7}{3} , \frac{5+7+9}{3}  \end{bmatrix}  = \begin{bmatrix} 3,5,7 \end{bmatrix}$$

It is possible to accomplish this in three lines of Python code (or with extra cleverness, even fewer!).

Note: The numpy library has a `convolve` [function](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html) to calculate moving average. Avoid simply importing that function and using it here.
"""

def moving_average(x, n):
  """
  Calculate moving averages of an array with a given window size
    
  Argument:
  x -- A one-dimensional array
  n -- Window size
    
  Returns:
  m -- the moving averages of array x with a specified window size n
  """

  ##############################################################################
  # TODO:                                                                      #
  # Write code to compute the moving average.
  ##############################################################################
  # Replace "pass" statement with your code
  m = []

  for i in range(len(x)-n+1):
    sum = 0
    sum = x[i]+x[i+1]+x[i+2]
    output = sum / n 
    m.append(output)
      



  ##############################################################################
  #                             END OF YOUR CODE                               #
  ############################################################################## 
  return m

x = np.array([1,3,5,7,9])
#y = np.array([8,5,1,6,1,2,5,6,9])

print("moving avarage(x, 3): " + str(moving_average(x, n = 3)))
print("moving avarage(y, 4): " + str(moving_average(y, n = 4)))

"""##Matplotlib

Matplotlib is a plotting library. In this section give a brief introduction to the `matplotlib.pyplot` module, which provides a plotting system similar to that of MATLAB.
"""

import matplotlib.pyplot as plt

"""By running this special iPython command, we will be displaying plots inline:"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""###Plotting

The most important function in `matplotlib` is plot, which allows you to plot 2D data. Here is a simple example:
"""

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)

"""With just a little bit of extra work we can easily plot multiple lines at once, and add a title, legend, and axis labels:"""

y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])

"""###Subplots

# 新增區段

You can plot different things in the same figure using the subplot function. Here is an example:
"""

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

"""You can read much more about the `subplot` function in the [documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)."""

