Setup 
=====

Dependencies
------------
See ``conda_environment.yml`` for Python dependencies. A new environment with
these dependencies can be created using::

   conda env create -f conda_environment.yml --prefix </path/to/environment/location>

Running the provided examples
-----------------------------
Global directory paths should be editted in ``armed.settings``:

1. ``RESULTSDIR``: where experimental results will be stored
2. ``DATADIR``: where downloaded and simulated datasets are stored

Add the repository root to the ``PYTHONPATH``. If using Visual Studio Code, this
can be done by modifying the ``.env`` file, which is read by the Python extension
when running any code interactively. 
