# This works from "Install packages with pip":
# https://godotengine.org/qa/37679/possible-use-python-modules-and-libraries-with-godot-yes-how?show=92938#a92938

# Better way, but couldn't get it to work:
# https://godotengine.org/qa/83554/how-to-import-python-modules-using-godot-python?show=83841#a83841

# uncomment "import" and "print" lines to test installation

from godot import exposed, export
from godot import *

#import numpy
#import tensorflow


@exposed
class test(Node2D):

	def _ready(self):

#		print("Numpy version: ")
#		print(numpy.__version__)
#		print("\nTensorflow version: ")
#		print(tensorflow.__version__)

		pass
