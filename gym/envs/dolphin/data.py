def array(basetype, length):
  class Array:
    _type_ = basetype
    _length_ = length
    
    def __init__(self, objs):
      self.array = objs
    
    def __getitem__(self, index):
      return self.array[index]
    
    def __len__(self):
      return self._length_
    
    def __setitem__(self, index, value):
      assert(isinstance(value, self._type_))
      self.array[index] = value
    
    def __iter__(self, *args, **kwargs):
      return self.array.__iter__(*args, **kwargs)
  
  
  Array.__name__ = '%s[%d]' % (basetype, length)
  return Array

