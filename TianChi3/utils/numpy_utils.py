"""
Utils for numpy.
"""

import numpy as np

def find_1D(condition):
    """
	find_1D(condition)

	Find indices in a 1-D vector.

	Parameters
	----------
	condition : conditions.
		e.g., a == 1

	Returns
	-------
	Indices of the search result.

	Examples
	--------
	>>> a = np.array([1,2,3,4,1,1,3,5,1,12])
	>>> b = numpy_utils.find_1D(a == 1)
	array([0, 4, 5, 8], dtype=int64)

    """
    return np.where(condition)[0]