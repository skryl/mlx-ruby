.. _utils:

Tree Utils
==========

In MLX we consider a ruby tree to be an arbitrarily nested collection of
dictionaries, lists and tuples without cycles. Functions in this module that
return ruby trees will be using the default ruby ``dict``, ``list`` and
``tuple`` but they can usually process objects that inherit from any of these.

.. note::
   Dictionaries should have keys that are valid ruby identifiers.

.. currentmodule:: mlx.utils

.. autosummary:: 
  :toctree: _autosummary

   tree_flatten
   tree_unflatten
   tree_map
   tree_map_with_path
   tree_reduce
