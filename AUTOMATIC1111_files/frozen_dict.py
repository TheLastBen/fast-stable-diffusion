# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen Dictionary."""

import collections
from typing import Any, TypeVar, Mapping, Dict, Tuple, Union, Hashable

from flax import serialization
import jax


class FrozenKeysView(collections.abc.KeysView):
  """A wrapper for a more useful repr of the keys in a frozen dict."""

  def __repr__(self):
    return f'frozen_dict_keys({list(self)})'


class FrozenValuesView(collections.abc.ValuesView):
  """A wrapper for a more useful repr of the values in a frozen dict."""

  def __repr__(self):
    return f'frozen_dict_values({list(self)})'


K = TypeVar('K')
V = TypeVar('V')


def _indent(x, num_spaces):
  indent_str = ' ' * num_spaces
  lines = x.split('\n')
  assert not lines[-1]
  # skip the final line because it's empty and should not be indented.
  return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


# TODO(ivyzheng): change to register_pytree_with_keys_class after JAX release.
@jax.tree_util.register_pytree_node_class
class FrozenDict(Mapping[K, V]):
  """An immutable variant of the Python dict."""
  __slots__ = ('_dict', '_hash')

  def __init__(self, *args, __unsafe_skip_copy__=False, **kwargs):  # pylint: disable=invalid-name
    # make sure the dict is as
    xs = dict(*args, **kwargs)
    if __unsafe_skip_copy__:
      self._dict = xs
    else:
      self._dict = _prepare_freeze(xs)

    self._hash = None

  def __getitem__(self, key):
    v = self._dict[key]
    if isinstance(v, dict):
      return FrozenDict(v)
    return v

  def __setitem__(self, key, value):
    raise ValueError('FrozenDict is immutable.')

  def __contains__(self, key):
    return key in self._dict

  def __iter__(self):
    return iter(self._dict)

  def __len__(self):
    return len(self._dict)

  def __repr__(self):
    return self.pretty_repr()

  def __reduce__(self):
    return FrozenDict, (self.unfreeze(),)

  def pretty_repr(self, num_spaces=4):
    """Returns an indented representation of the nested dictionary."""
    def pretty_dict(x):
      if not isinstance(x, dict):
        return repr(x)
      rep = ''
      for key, val in x.items():
        rep += f'{key}: {pretty_dict(val)},\n'
      if rep:
        return '{\n' + _indent(rep, num_spaces) + '}'
      else:
        return '{}'
    return f'FrozenDict({pretty_dict(self._dict)})'

  def __hash__(self):
    if self._hash is None:
      h = 0
      for key, value in self.items():
        h ^= hash((key, value))
      self._hash = h
    return self._hash

  def copy(self, add_or_replace: Mapping[K, V]) -> 'FrozenDict[K, V]':
    """Create a new FrozenDict with additional or replaced entries."""
    return type(self)({**self, **unfreeze(add_or_replace)}) # type: ignore[arg-type]

  def keys(self):
    return FrozenKeysView(self)

  def values(self):
    return FrozenValuesView(self)

  def items(self):
    for key in self._dict:
      yield (key, self[key])

  def pop(self, key: K) -> Tuple['FrozenDict[K, V]', V]:
    """Create a new FrozenDict where one entry is removed.

    Example::

      state, params = variables.pop('params')

    Args:
      key: the key to remove from the dict
    Returns:
      A pair with the new FrozenDict and the removed value.
    """
    value = self[key]
    new_dict = dict(self._dict)
    new_dict.pop(key)
    new_self = type(self)(new_dict)
    return new_self, value

  def unfreeze(self) -> Dict[K, V]:
    """Unfreeze this FrozenDict.

    Returns:
      An unfrozen version of this FrozenDict instance.
    """
    return unfreeze(self)

  # TODO(ivyzheng): remove this after JAX 0.4.6 release.
  def tree_flatten(self) -> Tuple[Tuple[Any, ...], Hashable]:
    """Flattens this FrozenDict.

    Returns:
      A flattened version of this FrozenDict instance.
    """
    sorted_keys = sorted(self._dict)
    return tuple([self._dict[k] for k in sorted_keys]), tuple(sorted_keys)

  @classmethod
  def tree_unflatten(cls, keys, values):
    # data is already deep copied due to tree map mechanism
    # we can skip the deep copy in the constructor
    return cls({k: v for k, v in zip(keys, values)}, __unsafe_skip_copy__=True)


#jax.tree_util.register_keypaths(
#    FrozenDict, lambda fd: tuple(jax.tree_util.DictKey(k) for k in sorted(fd))
#)


def _prepare_freeze(xs: Any) -> Any:
  """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
  if isinstance(xs, FrozenDict):
    # we can safely ref share the internal state of a FrozenDict
    # because it is immutable.
    return xs._dict  # pylint: disable=protected-access
  if not isinstance(xs, dict):
    # return a leaf as is.
    return xs
  # recursively copy dictionary to avoid ref sharing
  return {key: _prepare_freeze(val) for key, val in xs.items()}


def freeze(xs: Mapping[Any, Any]) -> FrozenDict[Any, Any]:
  """Freeze a nested dict.

  Makes a nested `dict` immutable by transforming it into `FrozenDict`.

  Args:
    xs: Dictionary to freeze (a regualr Python dict).
  Returns:
    The frozen dictionary.
  """
  return FrozenDict(xs)


def unfreeze(x: Union[FrozenDict, Dict[str, Any]]) -> Dict[Any, Any]:
  """Unfreeze a FrozenDict.

  Makes a mutable copy of a `FrozenDict` mutable by transforming
  it into (nested) dict.

  Args:
    x: Frozen dictionary to unfreeze.
  Returns:
    The unfrozen dictionary (a regular Python dict).
  """
  if isinstance(x, FrozenDict):
    # deep copy internal state of a FrozenDict
    # the dict branch would also work here but
    # it is much less performant because jax.tree_util.tree_map
    # uses an optimized C implementation.
    return jax.tree_util.tree_map(lambda y: y, x._dict)  # type: ignore
  elif isinstance(x, dict):
    ys = {}
    for key, value in x.items():
      ys[key] = unfreeze(value)
    return ys
  else:
    return x


def copy(x: Union[FrozenDict, Dict[str, Any]], add_or_replace: Union[FrozenDict, Dict[str, Any]]) -> Union[FrozenDict, Dict[str, Any]]:
  """Create a new dict with additional and/or replaced entries. This is a utility
  function that can act on either a FrozenDict or regular dict and mimics the
  behavior of `FrozenDict.copy`.

  Example::

  new_variables = copy(variables, {'additional_entries': 1})

  Args:
    x: the dictionary to be copied and updated
    add_or_replace: dictionary of key-value pairs to add or replace in the dict x
  Returns:
    A new dict with the additional and/or replaced entries.
  """

  if isinstance(x, FrozenDict):
    return x.copy(add_or_replace)
  elif isinstance(x, dict):
    new_dict = jax.tree_map(lambda x: x, x) # make a deep copy of dict x
    new_dict.update(add_or_replace)
    return new_dict
  raise TypeError(f'Expected FrozenDict or dict, got {type(x)}')


def pop(x: Union[FrozenDict, Dict[str, Any]], key: str) -> Tuple[Union[FrozenDict, Dict[str, Any]], Any]:
  """Create a new dict where one entry is removed. This is a utility
  function for regular dicts that mimics the behavior of `FrozenDict.pop`.

  Example::

    state, params = pop(variables, 'params')

  Args:
    x: the dictionary to remove the entry from
    key: the key to remove from the dict
  Returns:
    A pair with the new dict and the removed value.
  """

  if isinstance(x, FrozenDict):
    return x.pop(key)
  elif isinstance(x, dict):
    new_dict = jax.tree_map(lambda x: x, x) # make a deep copy of dict x
    value = new_dict.pop(key)
    return new_dict, value
  raise TypeError(f'Expected FrozenDict or dict, got {type(x)}')


def _frozen_dict_state_dict(xs):
  return {key: serialization.to_state_dict(value) for key, value in xs.items()}


def _restore_frozen_dict(xs, states):
  diff = set(map(str, xs.keys())).difference(states.keys())
  if diff:
    raise ValueError('The target dict keys and state dict keys do not match,'
                   f' target dict contains keys {diff} which are not present in state dict '
                   f'at path {serialization.current_path()}')

  return FrozenDict(
      {key: serialization.from_state_dict(value, states[key], name=key)
       for key, value in xs.items()})


serialization.register_serialization_state(
    FrozenDict,
    _frozen_dict_state_dict,
    _restore_frozen_dict)
