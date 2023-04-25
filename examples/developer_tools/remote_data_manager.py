"""
=========================
Using Remote Data Manager
=========================

This is an example to show how to use remote data manager to
handle remote data in Sunpy.
"""

##############################################################################
# Remote data manager is used to handle the usage of remote files in the code
# base with file verification using hashes.

from sunpy.data import manager

##############################################################################
# Let's start by defining a function which uses some remote files.
# Let's assume the function requires a file,
# say http://data.sunpy.org/sample-data/predicted-sunspot-radio-flux.txt, which has
# the SHA256 hash '4c85b04a5528aa97eb84a087450eda0421c71833820576330bba148564089b11'.


@manager.require('test_file',
                 ['http://data.sunpy.org/sample-data/predicted-sunspot-radio-flux.txt'],
                 '4c85b04a5528aa97eb84a087450eda0421c71833820576330bba148564089b11')
def test_function_1():
    pass

##############################################################################
# To access the downloaded file inside the function, you can use
# `~sunpy.data.data_manager.manager.DataManager.get` function
# which returns a `pathlib.Path` object.


@manager.require('test_file',
                 ['http://data.sunpy.org/sample-data/predicted-sunspot-radio-flux.txt'],
                 '4c85b04a5528aa97eb84a087450eda0421c71833820576330bba148564089b11')
def test_function_2():
    return manager.get('test_file')

##############################################################################
# The first time the function is called, the file will be downloaded.
# During subsequent calls, no downloading will take place.


print(test_function_2())  # The file will be downloaded
print(test_function_2())  # No downloading here

##############################################################################
# In case the user wants to skip the hash check, there is a helper context manager
# `~sunpy.data.data_manager.manager.DataManager.skip_hash_check`.

with manager.skip_hash_check():
    print(test_function_2())

##############################################################################
# If the user knows the function is going to use a file and want to replace it with another version
# they can do that too.

with manager.override_file('test_file', 'http://data.sunpy.org/sample-data/AIA20110319_105400_0171.fits'):
    print(test_function_2())
