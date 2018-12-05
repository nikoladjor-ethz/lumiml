Installation
============

Disclamer
---------

Since this user guide is not intended to be a Python tutorial, some knowledge of Python ecosystem is assumed.

Create a new environment
------------------------

Reccomended way to use ``lumiml`` is to start with a new Python environment, and install dependencies there. If you use conda, this should be simple:
 .. code-block:: bash

    conda create --name YourEnvName python=3.6
    conda activate YourEnvName

Assuming that you downloaded or forked our :download:`repository <https://github.com/nikoladjor/lumiml>`, it will be located somewhere on your hard drive at ``path\to\lumiml``.
Within that folder, there will be a ``requirements.txt`` file. In your terminal or Anaconda terminal (if you are running a Windows machine) run:
 .. code-block:: python
    
    pip install requirements.txt

This will install all requirements **apart** from `nlopt <https://nlopt.readthedocs.io/en/latest/>`_ package. Since the installation is OS dependent, this simple step has to be done manually by each user.
For installing ``nlopt``, please refer to `installing instructions <https://nlopt.readthedocs.io/en/latest/NLopt_Installation/>`_.

After all the packages are installed, you should be able to run ``examples\test_install.py`` without any errors:
 .. code-block:: bash

    python -m lumiml.examples.test_install

